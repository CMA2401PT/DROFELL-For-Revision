// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#include "q4_matrix.cuh"
#include <vector>
#include "util.cuh"
#include "matrix.cuh"
#include "qdq_util.cuh"

using namespace std;

const int UNSHUF_BLOCKSIZE_X = 64;

// const int RECONS_THREADS_X = 64;      // Block size and thread count along columns in out, each thread converts 1 column
// const int RECONS_THREADS_Y = 1;       // Block size and thread count along rows in x and out, each thread converts 8 rows

__global__ void shuffle_kernel
(
    uint32_t* __restrict__ b_q_weight,
    const int size_k,
    const int size_n
)
{
    int n = blockIdx.x * UNSHUF_BLOCKSIZE_X + threadIdx.x;
    if (n >= size_n) return;
    int k = 0;
    uint32_t* b_ptr = b_q_weight + n;
    while (k < size_k) { shuffle_4bit_8 (b_ptr, size_n); b_ptr += 1 * size_n; k +=  8; }
    while (k < size_k) { shuffle_3bit_32(b_ptr, size_n); b_ptr += 3 * size_n; k += 32; }
    while (k < size_k) { shuffle_2bit_16(b_ptr, size_n); b_ptr += 1 * size_n; k += 16; }
}


Q4Matrix::Q4Matrix
(
    const int _height,
    const int _width,
    const int _groups,

    uint32_t* _qweight,
    uint32_t* _qzeros,
    half* _scales,
    uint32_t* _g_idx,
    uint32_t* _cuda_x_map,
    const int _device,
    bool _v2_mode,
    cudaStream_t alt_stream
) :
    height(_height),
    width(_width),
    groups(_groups),
    device(_device),
    v2_mode(_v2_mode)
{
    cudaSetDevice(device);

    cuda_qweight = _qweight;
    cuda_qzeros = _qzeros;
    cuda_scales = _scales;

    groupsize = height / groups;

    if (_g_idx) {
        cuda_x_map=_cuda_x_map;
        make_sequential(_g_idx,alt_stream);
    };

    if (v2_mode){
        dim3 blockDim, gridDim;
        blockDim.x = UNSHUF_BLOCKSIZE_X;
        blockDim.y = 1;
        gridDim.x = DIVIDE(width, UNSHUF_BLOCKSIZE_X);
        gridDim.y = 1;

        shuffle_kernel<<<gridDim, blockDim,0,alt_stream>>>(cuda_qweight, height, width);
    }
}

Q4Matrix::~Q4Matrix()
{
}

// Make sequential

__global__ void make_sequential_kernel
(
    const uint32_t* __restrict__ w,
    uint32_t* __restrict__ w_new,
    const uint32_t* __restrict__ x_map,
    const int w_height,
    const int w_width
)
{
    const uint64_t* w2 = (uint64_t*) w;
    uint64_t* w_new2 = (uint64_t*) w_new;
    int w2_stride = w_width >> 1;

    int w2_column = UNSHUF_BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    if (w2_column >= w2_stride) return;
    int w_new2_row = blockIdx.y;

    int x_map_idx = w_new2_row << 3;

    uint64_t dst = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int source_row = x_map[x_map_idx++];

        int w2_row = source_row >> 3;
        int w2_subrow = source_row & 0x07;
        int w2_row_shift = w2_subrow << 2;
        int wnew2_row_shift = i << 2;

        uint64_t src = w2[w2_row * w2_stride + w2_column];
        src >>= w2_row_shift;
        src &= 0x0000000f0000000f;
        src <<= wnew2_row_shift;
        dst |= src;
    }

    w_new2[w_new2_row * w2_stride + w2_column] = dst;
}

void Q4Matrix::make_sequential(const uint32_t* cpu_g_idx,cudaStream_t alt_stream)
{
    uint32_t* cuda_new_qweight = NULL;
    cudaMalloc(&cuda_new_qweight, height / 8 * width * sizeof(uint32_t));
    // cudaMalloc(&cuda_x_map, height * sizeof(uint32_t));  // TODO: Should probably be allocated in PyTorch

    uint32_t* cpu_g_idx_map = (uint32_t*) calloc(groups, sizeof(uint32_t));
    uint32_t* cpu_x_map = (uint32_t*) malloc(height * sizeof(uint32_t));
    uint32_t* cpu_x_map_inv = (uint32_t*) malloc(height * sizeof(uint32_t));

    // Group histogram

    for (int i = 0; i < height; i++) cpu_g_idx_map[cpu_g_idx[i]]++;
    // Group map

    for (int i = 0, acc = 0; i < groups; i++)
    {
        short tmp = cpu_g_idx_map[i];
        cpu_g_idx_map[i] = acc;
        acc += tmp;
    }

    // X map (inverse)

    for (int row = 0; row < height; row++)
    {
        uint32_t target_group = cpu_g_idx[row];
        uint32_t target_row = cpu_g_idx_map[target_group];
        cpu_g_idx_map[target_group]++;
        cpu_x_map_inv[row] = target_row;
    }

    // X map

    for (int row = 0; row < height; row++) cpu_x_map[cpu_x_map_inv[row]] = row;

    // Move to CUDA

    cudaMemcpyAsync(cuda_x_map, cpu_x_map, height * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Rearrange rows in w

    dim3 threads(UNSHUF_BLOCKSIZE_X, 1, 1);
    dim3 blocks
    (
        (width + UNSHUF_BLOCKSIZE_X * 2 - 1) / (UNSHUF_BLOCKSIZE_X * 2),
        height / 8,
        1
    );

    make_sequential_kernel<<<blocks, threads,0,alt_stream>>>(cuda_qweight, cuda_new_qweight, cuda_x_map, height / 8, width);

    // Replace qweights

    cudaMemcpyAsync(cuda_qweight, cuda_new_qweight, height / 8 * width * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    // Cleanup

    cudaDeviceSynchronize();
    cudaFree(cuda_new_qweight);
    free(cpu_g_idx_map);
    free(cpu_x_map);
    free(cpu_x_map_inv);
}

__global__ void reconstruct_kernel
(
    const uint32_t* __restrict__ w,
    half* __restrict__ out,  // (y)
    const half* __restrict__ w_scales,
    const uint32_t* __restrict__ w_zeros,
    const int height,
    const int width,
    const int groupsize,
    const int reconstruct_threads_x,
    const int reconstruct_threads_y
)
{
    // Start of block

    int column = reconstruct_threads_x * blockIdx.x + threadIdx.x;
    int row = (reconstruct_threads_y * blockIdx.y + threadIdx.y) * 8;
    if (column >= width) return;

    // Views

    MatrixView_q4_column w_(w, height, width);
    MatrixView_half_rw out_(out, height, width);
    MatrixView_half w_scales_(w_scales, height / groupsize, width);
    MatrixView_q4_row w_zeros_(w_zeros, height / groupsize, width);

    // Groupsize version

    int group = row / groupsize;

    half w_scale = w_scales_.item(group, column);
    uint32_t w_zero = w_zeros_.item(group, column) + 1;

    uint32_t w_read = w_.item_uint32_t(row, column);
    half* out_ptr = out_.item_ptr(row, column);

    #pragma unroll
    for (int s = 0; s < 32; s += 4)
    {
        half w_item = __hmul(__int2half_rn((int)((w_read >> s) & 0x0f) - w_zero), w_scale);
        *out_ptr = w_item; out_ptr += out_.width;
    }
}

void Q4Matrix::reconstruct(
    half* out,
    const int reconstruct_threads_x,
    const int reconstruct_threads_y,
    cudaStream_t alt_stream
)
{
    dim3 threads(reconstruct_threads_x, reconstruct_threads_y, 1);

    dim3 blocks
    (
        (width + threads.x - 1) / threads.x,
        (height / 8 + threads.y - 1) / threads.y,
        1
    );

    reconstruct_kernel<<<blocks, threads,0, alt_stream>>>(cuda_qweight, out, cuda_scales, cuda_qzeros, height / 8, width, groupsize,reconstruct_threads_x, reconstruct_threads_y);
}



__global__ void reconstruct_nv_bfloat16_kernel
(
    const uint32_t* __restrict__ w,
    nv_bfloat16* __restrict__ out,  // (y)
    const half* __restrict__ w_scales,
    const uint32_t* __restrict__ w_zeros,
    const int height,
    const int width,
    const int groupsize,
    const int reconstruct_threads_x,
    const int reconstruct_threads_y
)
{
    // Start of block

    int column = reconstruct_threads_x * blockIdx.x + threadIdx.x;
    int row = (reconstruct_threads_y * blockIdx.y + threadIdx.y) * 8;
    if (column >= width) return;

    
    // Views

    MatrixView_q4_column w_(w, height, width);
    MatrixView_nv_bfloat16_rw out_(out, height, width);
    MatrixView_half w_scales_(w_scales, height / groupsize, width);
    MatrixView_q4_row w_zeros_(w_zeros, height / groupsize, width);

    // Groupsize version

    int group = row / groupsize;

    half w_scale = w_scales_.item(group, column);
    uint32_t w_zero = w_zeros_.item(group, column) + 1;

    uint32_t w_read = w_.item_uint32_t(row, column);
    nv_bfloat16* out_ptr = out_.item_ptr(row, column);

    #pragma unroll
    for (int s = 0; s < 32; s += 4)
    {
        half w_item = __hmul(__int2half_rn((int)((w_read >> s) & 0x0f) - w_zero), w_scale);
        *out_ptr = __float2bfloat16(__half2float(w_item)); out_ptr += out_.width;
    }
}

void Q4Matrix::reconstruct_nv_bfloat16(
    nv_bfloat16* out,
    const int reconstruct_threads_x,
    const int reconstruct_threads_y,
    cudaStream_t alt_stream
)
{
    dim3 threads(reconstruct_threads_x, reconstruct_threads_y, 1);

    dim3 blocks
    (
        (width + threads.x - 1) / threads.x,
        (height / 8 + threads.y - 1) / threads.y,
        1
    );

    reconstruct_nv_bfloat16_kernel<<<blocks, threads,0, alt_stream>>>(cuda_qweight, out, cuda_scales, cuda_qzeros, height / 8, width, groupsize,reconstruct_threads_x, reconstruct_threads_y);
}

__global__ void reconstruct_gptq_kernel
(
    const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales,
    const int size_k,
    const int size_n,
    const int groupsize,
    const int groups,
    half* __restrict__ b,
    const int block_kn_size
)
{
    MatrixView_half_rw b_(b, size_k, size_n);
    MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
    MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

    int offset_k = block_kn_size * blockIdx.y;
    int offset_n = block_kn_size * blockIdx.x * 4;

    int end_k = min(offset_k + block_kn_size, size_k);

    // Preload remapping table

    extern  __shared__ uint16_t perm[];
    int t = threadIdx.x;

    if (b_q_perm)
    {
        if (offset_k + t < size_k)
            perm[t] = uint16_t(b_q_perm[offset_k + t]);
    }

    // Column

    int n = offset_n + t * 4;
    if (n >= size_n) return;

    // Find initial group

    int group = offset_k / groupsize;
    int nextgroup = offset_k + groupsize;

    // b offset

    int qk = offset_k / (32 / 4);

    const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

    // Initial zeros/scale

    int zeros[4];
    half2 scales[4];
    half2 z1z16[4][2];
    half2 y1y16[4][2];
    b_gptq_qzeros_.item4(zeros, group, n);
    b_gptq_scales_.item4_h2(scales, group, n);
    dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
    dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
    dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
    dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);

    __syncthreads();

    int k = offset_k;
    int lk = 0;

    while (k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            nextgroup += groupsize;
            b_gptq_qzeros_.item4(zeros, group, n);
            b_gptq_scales_.item4_h2(scales, group, n);
            dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
            dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
            dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
            dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);
        }

        for (int p = 0; p < 4; p++)
        {
            half2 dq[4][4];
            const int4* b_ptr4 = (int4*) b_ptr;
            int4 load_int4 = *b_ptr4;

            dequant_4bit_8_gptq(load_int4.x, dq[0], z1z16[0], y1y16[0], size_n, false);
            dequant_4bit_8_gptq(load_int4.y, dq[1], z1z16[1], y1y16[1], size_n, false);
            dequant_4bit_8_gptq(load_int4.z, dq[2], z1z16[2], y1y16[2], size_n, false);
            dequant_4bit_8_gptq(load_int4.w, dq[3], z1z16[3], y1y16[3], size_n, false);

            b_ptr += size_n;
            //half* dqh = (half*)dq;
            if (b_q_perm)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
                    b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]), __low2half(dq[2][j]), __low2half(dq[3][j]));
                    b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]), __high2half(dq[2][j]), __high2half(dq[3][j]));
                }
            }
            else
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
                    b_.set4(offset_k + lk++, n, __low2half(dq[0][j]), __low2half(dq[1][j]), __low2half(dq[2][j]), __low2half(dq[3][j]));
                    b_.set4(offset_k + lk++, n, __high2half(dq[0][j]), __high2half(dq[1][j]), __high2half(dq[2][j]), __high2half(dq[3][j]));
                }
            }
        }
        k += 32;
    }
}

void Q4Matrix::reconstruct_exllama_v2(
    half* out,
    const int block_kn_size,
    cudaStream_t alt_stream
)
{
    dim3 blockDim, gridDim;
    blockDim.x = block_kn_size;
    blockDim.y = 1;
    gridDim.x = DIVIDE(width, block_kn_size*4);
    gridDim.y = DIVIDE(height, block_kn_size);

    int shared_mem = block_kn_size * sizeof(uint16_t);
    reconstruct_gptq_kernel<<<gridDim, blockDim, shared_mem,alt_stream>>>
    (
        cuda_qweight,
        cuda_x_map,
        cuda_qzeros,
        cuda_scales,
        height,
        width,
        groupsize,
        groups,
        out,
        block_kn_size
    );
}

__device__ __forceinline__ nv_bfloat16 half2fbloat16(const half in) {
    return __float2bfloat16(__half2float(in));
}

__global__ void reconstruct_gptq_kernel_nv_bfloat16
(
    const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales,
    const int size_k,
    const int size_n,
    const int groupsize,
    const int groups,
    nv_bfloat16* __restrict__ b,
    const int block_kn_size
)
{
    MatrixView_nv_bfloat16_rw b_(b, size_k, size_n);
    MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
    MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

    int offset_k = block_kn_size * blockIdx.y;
    int offset_n = block_kn_size * blockIdx.x * 4;

    int end_k = min(offset_k + block_kn_size, size_k);

    // Preload remapping table

    extern  __shared__ uint16_t perm[];
    int t = threadIdx.x;

    if (b_q_perm)
    {
        if (offset_k + t < size_k)
            perm[t] = uint16_t(b_q_perm[offset_k + t]);
    }

    // Column

    int n = offset_n + t * 4;
    if (n >= size_n) return;

    // Find initial group

    int group = offset_k / groupsize;
    int nextgroup = offset_k + groupsize;

    // b offset

    int qk = offset_k / (32 / 4);

    const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

    // Initial zeros/scale

    int zeros[4];
    half2 scales[4];
    half2 z1z16[4][2];
    half2 y1y16[4][2];
    b_gptq_qzeros_.item4(zeros, group, n);
    b_gptq_scales_.item4_h2(scales, group, n);
    dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
    dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
    dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
    dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);

    __syncthreads();

    int k = offset_k;
    int lk = 0;

    while (k < end_k)
    {
        if (k == nextgroup)
        {
            group++;
            nextgroup += groupsize;
            b_gptq_qzeros_.item4(zeros, group, n);
            b_gptq_scales_.item4_h2(scales, group, n);
            dequant_4bit_8_prep_zero(zeros[0] + 1, z1z16[0], y1y16[0]);
            dequant_4bit_8_prep_zero(zeros[1] + 1, z1z16[1], y1y16[1]);
            dequant_4bit_8_prep_zero(zeros[2] + 1, z1z16[2], y1y16[2]);
            dequant_4bit_8_prep_zero(zeros[3] + 1, z1z16[3], y1y16[3]);
        }

        for (int p = 0; p < 4; p++)
        {
            half2 dq[4][4];
            const int4* b_ptr4 = (int4*) b_ptr;
            int4 load_int4 = *b_ptr4;

            dequant_4bit_8_gptq(load_int4.x, dq[0], z1z16[0], y1y16[0], size_n, false);
            dequant_4bit_8_gptq(load_int4.y, dq[1], z1z16[1], y1y16[1], size_n, false);
            dequant_4bit_8_gptq(load_int4.z, dq[2], z1z16[2], y1y16[2], size_n, false);
            dequant_4bit_8_gptq(load_int4.w, dq[3], z1z16[3], y1y16[3], size_n, false);

            b_ptr += size_n;
            //half* dqh = (half*)dq;
            if (b_q_perm)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
                    b_.set4(perm[lk++], n, half2fbloat16(__low2half(dq[0][j])), half2fbloat16(__low2half(dq[1][j])), half2fbloat16(__low2half(dq[2][j])), half2fbloat16(__low2half(dq[3][j])));
                    b_.set4(perm[lk++], n, half2fbloat16(__high2half(dq[0][j])), half2fbloat16(__high2half(dq[1][j])), half2fbloat16(__high2half(dq[2][j])), half2fbloat16(__high2half(dq[3][j])));
                }
            }
            else
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
                    b_.set4(offset_k + lk++, n, half2fbloat16(__low2half(dq[0][j])), half2fbloat16(__low2half(dq[1][j])), half2fbloat16(__low2half(dq[2][j])), half2fbloat16(__low2half(dq[3][j])));
                    b_.set4(offset_k + lk++, n, half2fbloat16(__high2half(dq[0][j])), half2fbloat16(__high2half(dq[1][j])), half2fbloat16(__high2half(dq[2][j])), half2fbloat16(__high2half(dq[3][j])));
                }
            }
        }
        k += 32;
    }
}


void Q4Matrix::reconstruct_nv_bfloat16_exllama_v2(
    nv_bfloat16* out,
    const int block_kn_size,
    cudaStream_t alt_stream
)
{
    dim3 blockDim, gridDim;
    blockDim.x = block_kn_size;
    blockDim.y = 1;
    gridDim.x = DIVIDE(width, block_kn_size);
    gridDim.y = DIVIDE(height, block_kn_size);

    int shared_mem = block_kn_size * sizeof(uint16_t);
    reconstruct_gptq_kernel_nv_bfloat16<<<gridDim, blockDim, shared_mem,alt_stream>>>
    (
        cuda_qweight,
        cuda_x_map,
        cuda_qzeros,
        cuda_scales,
        height,
        width,
        groupsize,
        groups,
        out,
        block_kn_size
    );
}

// void QMatrix::reconstruct_exllama_v2(
//     half* out,
//     const int block_kn_size,
//     cudaStream_t alt_stream
// )
// {
//     dim3 blockDim, gridDim;
//     blockDim.x = block_kn_size;
//     blockDim.y = 1;
//     gridDim.x = DIVIDE(width, block_kn_size);
//     gridDim.y = DIVIDE(height, block_kn_size);

//     int shared_mem = block_kn_size * sizeof(uint16_t);
//     reconstruct_gptq_kernel<<<gridDim, blockDim, shared_mem,alt_stream>>>
//     (
//         cuda_qweight,
//         cuda_x_map,
//         cuda_qzeros,
//         cuda_scales,
//         //const uint16_t* __restrict__ b_q_groups,
//         height,
//         width,
//         groupsize,
//         groups,
//         out,
//         block_kn_size
//     );
// }
