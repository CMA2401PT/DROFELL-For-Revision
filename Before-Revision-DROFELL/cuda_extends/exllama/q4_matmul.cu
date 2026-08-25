#include "q4_matmul.cuh"
#include "column_remap.cuh"
#include "util.cuh"
#include "matrix.cuh"
#include "cuda_compat.cuh"
#include "qdq_util.cuh"
#include "q_gemm_kernel_gptq.cuh"
#include <cuda_runtime.h>
#include "cublas_v2.h"

typedef void (*fp_q4_matmul_kernel)
(
    const half*,
    const uint32_t*,
    half*,
    const half*,
    const uint32_t*,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int,
    const uint32_t*,
    bool
);

template<bool use_half2, bool use_groupsize, bool use_x_map>
__global__ void q4_matmul_kernel
(
    const half* __restrict__ x,
    const uint32_t* __restrict__ w,
    half* __restrict__ out,
    const half* __restrict__ w_scales,
    const uint32_t* __restrict__ w_zeros,
    const int height,
    const int dim,
    const int width,
    const int groupsize,
    const int threads_x,
    const int threads_y,
    const int block_size_z,
    const int groupstep,
    const uint32_t* __restrict__ x_map,
    bool no_zero
)
{
    extern __shared__ half2 x_cache[];
    half* x_cache_h = (half*)x_cache;
    // Start of block

    int x_column = block_size_z * blockIdx.z;
    int x_column_end = min(dim, block_size_z * (blockIdx.z + 1));

    int w_column = threads_x * blockIdx.x + threadIdx.x;
    int x_row = threads_y * blockIdx.y + threadIdx.y;

    int iterations = (x_column_end - x_column) / 8;

    // Views

    MatrixView_half x_(x, height, dim);
    MatrixView_half w_scales_(w_scales, dim / groupsize, width);
    MatrixView_q4_row w_zeros_(w_zeros, dim / groupsize, width);
    MatrixView_q4_column w_(w, dim, width);
    MatrixView_half_rw out_(out, height, width);

    // Zero output

    // threadIdx.x & 1 2 half share one float(uint32_4)
    if (!no_zero && blockIdx.z == 0 && (threadIdx.x & 1) == 0)
    {
        *((uint32_t*) out_.item_ptr(x_row, w_column)) = 0;
    }
    __syncthreads();

    // Loop over part of x row (and w column)

    half2 acc = {};
    half acc_h = {};

    if constexpr (use_groupsize)
    {
        // For quant matrices where groupsize divides BLOCK_SIZE_Z we always start on a group boundary, so this
        // could be slightly faster

        for (int k = x_column, group = x_column / groupsize; k < x_column + iterations * 8; group++, k += groupsize)
        {
                for (int i = threadIdx.x; i < groupsize; i += threads_x)
                {
                    if constexpr (use_x_map) x_cache_h[i] = *x_.item_ptr(x_row, x_map[k + i]);
                    else                     x_cache_h[i] = *x_.item_ptr(x_row, k + i);
                }
                __syncthreads();

                if constexpr (use_half2)
                {
                    half2 w_scale = w_scales_.item_half2half2(group, w_column);
                    uint32_t w_zero = w_zeros_.item(group, w_column) + 1;
                    acc = dot_product_8(acc, x_cache, w_, k, w_column, w_scale, w_zero, groupsize / 8);
                }
                else
                {
                    half w_scale = w_scales_.item(group, w_column);
                    uint32_t w_zero = w_zeros_.item(group, w_column) + 1;
                    acc_h = dot_product_8_h(acc_h, x_cache_h, w_, k, w_column, w_scale, w_zero, groupsize / 8);
                }
                __syncthreads();

                // if constexpr (use_half2)
                // {
                //     half2 w_scale = w_scales_.item_half2half2(group, w_column);
                //     uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

                //     if constexpr (use_x_map) acc = dot_product_8_x_map(acc, x_, x_row, k, w_, k, w_column, w_scale, w_zero, groupsize / 8, x_map);
                //     else                     acc = dot_product_8      (acc, (const half2*) x_.item_ptr(x_row, k), w_, k, w_column, w_scale, w_zero, groupsize / 8);
                // }
                // else
                // {
                //     half w_scale = w_scales_.item(group, w_column);
                //     uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

                //     if constexpr (use_x_map) acc_h = dot_product_8_x_map_h(acc_h, x_, x_row, k, w_, k, w_column, w_scale, w_zero, groupsize / 8, x_map);
                //     else                     acc_h = dot_product_8_h      (acc_h, x_.item_ptr(x_row, k), w_, k, w_column, w_scale, w_zero, groupsize / 8);
                // }

        }
    }
    else
    {
        // Otherwise assume groupsize is a multiple of GROUP_STEP, do GROUP_STEP columns per iteration and trust the cache
        for (int k = x_column; k < x_column + iterations * 8; k += groupstep)
        {
            for (int i = threadIdx.x; i < groupstep; i += threads_x)
            {
                if constexpr (use_x_map) x_cache_h[i] = *x_.item_ptr(x_row, x_map[k + i]);
                else                     x_cache_h[i] = *x_.item_ptr(x_row, k + i);
            }
            __syncthreads();

            if constexpr (use_half2)
            {
                int group = k / groupsize;
                half2 w_scale = w_scales_.item_half2half2(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;
                acc = dot_product_8(acc, x_cache, w_, k, w_column, w_scale, w_zero, groupstep / 8);
            }
            else
            {
                int group = k / groupsize;
                half w_scale = w_scales_.item(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;
                acc_h = dot_product_8_h(acc_h, x_cache_h, w_, k, w_column, w_scale, w_zero, groupstep / 8);
            }
            __syncthreads();
        }


        // for (int k = x_column; k < x_column + iterations * 8; k += 8)
        // {
        //     if constexpr (use_half2)
        //     {
        //         int group = k / groupsize;
        //         half2 w_scale = w_scales_.item_half2half2(group, w_column);
        //         uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

        //         if constexpr (use_x_map) acc = dot_product_8_x_map(acc, x_, x_row, k, w_, k, w_column, w_scale, w_zero, 1, x_map);
        //         else                     acc = dot_product_8      (acc, (const half2*) x_.item_ptr(x_row, k), w_, k, w_column, w_scale, w_zero, 1);
        //     }
        //     else
        //     {
        //         int group = k / groupsize;
        //         half w_scale = w_scales_.item(group, w_column);
        //         uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

        //         if constexpr (use_x_map) acc_h = dot_product_8_x_map_h(acc_h, x_, x_row, k, w_, k, w_column, w_scale, w_zero, 1, x_map);
        //         else                     acc_h = dot_product_8_h      (acc_h, x_.item_ptr(x_row, k), w_, k, w_column, w_scale, w_zero, 1);
        //     }
        // }
    }

    // Add to block result

    if constexpr (use_half2)
    {
        half result = __hadd(acc.x, acc.y);
        atomicAdd(out_.item_ptr(x_row, w_column), result);
    }
    else
    {
        atomicAdd(out_.item_ptr(x_row, w_column), acc_h);
    }
}


fp_q4_matmul_kernel q4_matmul_kernel_pick(bool no_half2, int block_size_z, int groupsize, uint32_t* x_map)
{
    // <bool use_half2, bool use_groupsize, bool use_x_map>
    if (no_half2) {
        if (block_size_z % groupsize == 0) {
            if (x_map) return q4_matmul_kernel<false, true,  true >;
            else       return q4_matmul_kernel<false, true,  false>;
        } else {
            if (x_map) return q4_matmul_kernel<false, false, true >;
            else       return q4_matmul_kernel<false, false, false>;
        }
    } else {
        if (block_size_z % groupsize == 0)
        {
            if (x_map) return q4_matmul_kernel<true,  true,  true >;
            else       return q4_matmul_kernel<true,  true,  false>;
        } else {
            if (x_map) return q4_matmul_kernel<true,  false, true >;
            else       return q4_matmul_kernel<true,  false, false>;
        }
    }
};

// Compute y = x @ w

void q4_matmul_cuda
(
    const half* x,
    const int x_height,
    const Q4Matrix* w,
    half* out,
    half* temp_x,
    bool no_zero,
    bool no_half2,
    bool no_fused_remap,
    const int threads_x, // 32 Block size and thread count along columns in w and out
    const int threads_y, // 1
    const int block_size_z,
    const int matmul_group_step,
    const int x_remap_block_size_x,
    const int x_remap_block_size_y,
    bool skip_remap,
    cudaStream_t alt_stream
)
{
    int height = x_height;
    int dim = w->height;
    int width = w->width;
    // printf("m=%d, n=%d, k=%d\n",width,height,dim);

    cudaSetDevice(w->device);

    uint32_t* x_map = w->cuda_x_map;
    const half* x_mapped = x;
    if (skip_remap){
        x_map = NULL;
    }
    if (x_map && no_fused_remap)
    {
        // printf("column_remap_cuda\n");
        column_remap_cuda(x, temp_x, x_height, dim, w->cuda_x_map,x_remap_block_size_x,x_remap_block_size_y, alt_stream);
        x_mapped = temp_x;
        x_map = NULL;
    }else{
        // printf("no column_remap_cuda\n");
    }
   

    // int block_size_z;
    // if (w->width == 4096) block_size_z = 384;           // 7B
    // else if (w->width == 11008) block_size_z = 256;
    // else if (w->width == 5120) block_size_z = 384;      // 13B
    // else if (w->width == 13824) block_size_z = 256;
    // else if (w->width == 6656) block_size_z = 256;      // 33B
    // else if (w->width == 17920) block_size_z = 128;
    // else block_size_z = 256;

    //if (!no_zero) cudaMemsetAsync(out, 0, x_height * w->width * sizeof(half));

    dim3 threads(threads_x, threads_y, 1);

    dim3 blocks
    (
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y,
        (dim + block_size_z - 1) / block_size_z
    );

    fp_q4_matmul_kernel kernel = q4_matmul_kernel_pick(no_half2, block_size_z, w->groupsize, x_map);
    int shared_mem = (block_size_z % w->groupsize == 0 ? w->groupsize : matmul_group_step) * sizeof(half);

    kernel<<<blocks, threads, shared_mem, alt_stream>>> (x_mapped, w->cuda_qweight, out, w->cuda_scales, w->cuda_qzeros, height, dim, width, w->groupsize, threads_x, threads_y,block_size_z,matmul_group_step, x_map, no_zero);
}

void q4_matmul_cudablas
(
    const half* x,
    const int x_height,
    Q4Matrix* w,
    half* out,
    half* temp_weight,
    half* temp_x,
    const cublasHandle_t handle,
    bool no_zero,
    const int x_remap_block_size_x,
    const int x_remap_block_size_y,
    const int reconstruct_threads_x,
    const int reconstruct_threads_y,
    bool skip_remap,
    bool skip_reconstruct,
    bool bfloat16,
    cudaStream_t alt_stream
)
{
    int height = x_height;
    int dim = w->height;
    int width = w->width;

    // qwidth [heigth,width]
    // q4 [height(rows)=x.width(cols)=qweight.height(rows)*8,width(cols)=qweight.width]
    // q4 height(rows) = x width(cols)

    cudaSetDevice(w->device);

    const half* x_mapped = x;

    if (w->cuda_x_map && !skip_remap)
    {
        // printf("cuda x map\n");
        column_remap_cuda(x, temp_x, x_height, dim, w->cuda_x_map,x_remap_block_size_x,x_remap_block_size_y,alt_stream);
        x_mapped = temp_x;
    }else{
        // printf("no cuda x map\n");
    }

    if (bfloat16){
        if (!skip_reconstruct){
            w->reconstruct_nv_bfloat16((nv_bfloat16 *)temp_weight,reconstruct_threads_x,reconstruct_threads_y,alt_stream);
        }
        const float alpha = 1.0f;
        const float beta = no_zero ? 1.0f : 0.0f;
        cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    width, height, dim,
                    &alpha, 
                    temp_weight, CUDA_R_16BF, width,
                    x_mapped, CUDA_R_16BF, dim, 
                    &beta, 
                    out, CUDA_R_16BF, width
                );
        // printf("returned value %s\n",cublasGetStatusString(val));
        
    }else{
        if (!skip_reconstruct){
            w->reconstruct(temp_weight,reconstruct_threads_x,reconstruct_threads_y,alt_stream);
        }
        const half alpha = __float2half(1.0f);
        const half beta = no_zero ? __float2half(1.0f) : __float2half(0.0f);
        cublasHgemm(handle, CUBLAS_OP_N,CUBLAS_OP_N,
            width, height, dim,
            &alpha, 
            temp_weight, width, 
            x_mapped, dim, 
            &beta, 
            out, width
        );
    }





    // cublasStatus_t cublasHgemm(cublasHandle_t handle,
    //                        cublasOperation_t transa, CUBLAS_OP_N/T
    //                        cublasOperation_t transb,
    //                        int m, number of rows of matrix op(A) and C.
    //                        int n, number of columns of matrix op(B) and C.
    //                        int k, number of columns of op(A) and rows of op(B).
    //                        const __half *alpha,
    //                        const __half *A, int lda,  leading dimension of two-dimensional array used to store the matrix A A 的主维, 通过指定为行数或者列数来表示存贮的顺序(行优先,列优先)
    //                        const __half *B, int ldb,
    //                        const __half *beta,
    //                        __half *C, int ldc
    //                        )
}


void gemm_half_q_half_cuda_part
(
    const half* a,
    const Q4Matrix* b,
    half* c,
    int block_kn_size,
    int size_m,
    int size_n,
    int size_k,
    int m_count,
    bool clear,
    cudaStream_t alt_stream
)
{
    dim3 blockDim, gridDim;
    blockDim.x = block_kn_size;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.x = DIVIDE(size_n, block_kn_size * 4);
    gridDim.y = DIVIDE(size_m, m_count);
    gridDim.z = DIVIDE(size_k, block_kn_size);

    fp_gemm_half_q_half_gptq_kernel kernel = pick_gemm_half_q_half_gptq_kernel(true, m_count);
    int shared_mem_size=block_kn_size*size_m*sizeof(half);
    kernel<<<gridDim, blockDim,shared_mem_size,alt_stream>>>
    (
        a,
        b->cuda_qweight,
        b->cuda_qzeros,
        b->cuda_scales,
        c,
        block_kn_size,
        size_m,
        size_n,
        size_k,
        b->groups,
        b->groupsize,
        b->cuda_x_map,
        b->height,
        clear
    );
}


void q4_matmul_cuda_v2
(
    const half* x, // a
    const int x_height,
    const Q4Matrix* w, // b
    half* out, // c 
    const int block_kn_size,
    const int block_m_size_max,
    bool no_zero,
    cudaStream_t alt_stream
)
{
    int height = x_height; // m
    int dim = w->height; //k
    int width = w->width; //n
    int max_chunks = height / block_m_size_max;
    int last_chunk = max_chunks * block_m_size_max;
    int last_chunk_size = height - last_chunk;

    if (max_chunks)
    {
        gemm_half_q_half_cuda_part(x, w, out, block_kn_size, last_chunk, width, dim, block_m_size_max, !no_zero,alt_stream);
    }

    if (last_chunk_size)
    {
        gemm_half_q_half_cuda_part(x + last_chunk * dim, w, out + last_chunk * width, block_kn_size, last_chunk_size, width, dim, last_chunk_size, !no_zero,alt_stream);
    }
}


void q4_matmul_cudablas_v2(
    const half* x,
    const int x_height,
    Q4Matrix* w,
    half* out,
    half* temp_weight,
    const cublasHandle_t handle,
    bool no_zero,
    const int block_kn_size,
    bool skip_reconstruct,
    bool bfloat16,
    cudaStream_t alt_stream
){
    int height = x_height;
    int dim = w->height;
    int width = w->width;

    cudaSetDevice(w->device);

    const half* x_mapped = x;

    if (bfloat16){
        if (!skip_reconstruct){
            w->reconstruct_nv_bfloat16_exllama_v2((nv_bfloat16 *)temp_weight,block_kn_size,alt_stream);
        }
        const float alpha = 1.0f;
        const float beta = no_zero ? 1.0f : 0.0f;
        cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    width, height, dim,
                    &alpha, 
                    temp_weight, CUDA_R_16BF, width,
                    x_mapped, CUDA_R_16BF, dim, 
                    &beta, 
                    out, CUDA_R_16BF, width
                );
        // printf("returned value %s\n",cublasGetStatusString(val));
    }else{
        if (!skip_reconstruct){
            w->reconstruct_exllama_v2(temp_weight,block_kn_size,alt_stream);
        }
        const half alpha = __float2half(1.0f);
        const half beta = no_zero ? __float2half(1.0f) : __float2half(0.0f);
        cublasHgemm(handle, CUBLAS_OP_N,CUBLAS_OP_N,
            width, height, dim,
            &alpha, 
            temp_weight, width, 
            x_mapped, dim, 
            &beta, 
            out, width
        );
    }
}