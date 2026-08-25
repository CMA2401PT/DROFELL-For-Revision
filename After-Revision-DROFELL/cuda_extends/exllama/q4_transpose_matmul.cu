#include "column_remap.cuh"
#include "util.cuh"
#include "matrix.cuh"
#include "cuda_compat.cuh"
#include "q4_transpose_matmul.cuh"

void q4_transpose_matmul_blas(
    const half *x,
    const int x_height,
    Q4Matrix *w,
    half *out,
    half *temp_weight,
    half *temp_x,
    const cublasHandle_t handle,
    bool no_zero,
    const int x_restore_block_size_x,
    const int x_restore_block_size_y,
    const int reconstruct_threads_x,
    const int reconstruct_threads_y,
    bool skip_remap,
    bool skip_reconstruct,
    bool bfloat16,
    cudaStream_t alt_stream)
{
    int n = x_height; // batch size
    int m = w->height;
    int k = w->width;

    cudaSetDevice(w->device);

    half *out_to_restore = out;

    if (w->cuda_x_map)
    {
        out_to_restore = temp_x;
    }

    if (bfloat16)
    {
        if (!skip_reconstruct)
        {
            w->reconstruct_nv_bfloat16((nv_bfloat16 *)temp_weight, reconstruct_threads_x, reconstruct_threads_y, alt_stream);
        }
        const float alpha = 1.0f;
        const float beta = no_zero ? 1.0f : 0.0f;
        cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                      m, n, k,
                      &alpha,
                      temp_weight, CUDA_R_16BF, w->width,
                      x, CUDA_R_16BF, w->width,
                      &beta,
                      out_to_restore, CUDA_R_16BF, m);
    }
    else
    {
        if (!skip_reconstruct)
        {
            w->reconstruct(temp_weight, reconstruct_threads_x, reconstruct_threads_y, alt_stream);
        }
        const half alpha = __float2half(1.0f);
        const half beta = no_zero ? __float2half(1.0f) : __float2half(0.0f);
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    m, n, k,
                    &alpha,
                    temp_weight, w->width,
                    x, w->width,
                    &beta,
                    out_to_restore, m);
    }
    // w->reconstruct(buffers->temp_dq);
    // const half alpha = __float2half(1.0f);
    // const half beta = no_zero ? __float2half(1.0f) : __float2half(0.0f);

    // A [rows,cols] q4
    // X

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
    // printf("m=%d, n=%d, k=%d",m,n,k);
    // cublasHgemm(handle,
    //     CUBLAS_OP_T,
    //     CUBLAS_OP_N,
    //     m, n, k,
    //     &alpha,
    //     buffers->temp_dq, w->width,
    //     x, w->width,
    //     &beta, out_to_restore,
    //     m
    // );
    //     const float alpha = 1.0f;
    //     const float beta = no_zero ? 1.0f : 0.0f;
    //     cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, height, dim, &alpha, buffers->temp_dq, CUDA_R_16F, width,
    //                 x_mapped, CUDA_R_16F, dim, &beta, out, CUDA_R_16F, width);
    if (w->cuda_x_map && !skip_remap)
    {
        // printf("reconstructe x restore\n");
        column_restore_cuda(out_to_restore, out, x_height, w->height, w->cuda_x_map, x_restore_block_size_x, x_restore_block_size_y, alt_stream);
    }
    else
    {
        // printf("no reconstructe x restore\n");
    }
}

void q4_transpose_matmul_cudablas_v2(
    const half *x,
    const int x_height,
    Q4Matrix *w,
    half *out,
    half *temp_weight,
    const cublasHandle_t handle,
    bool no_zero,
    const int block_kn_size,
    bool skip_reconstruct,
    bool bfloat16,
    cudaStream_t alt_stream)
{
    int n = x_height; // batch size
    int m = w->height;
    int k = w->width;

    cudaSetDevice(w->device);

    if (bfloat16)
    {
        if (!skip_reconstruct)
        {
            w->reconstruct_nv_bfloat16_exllama_v2((nv_bfloat16 *)temp_weight, block_kn_size, alt_stream);
        }
        const float alpha = 1.0f;
        const float beta = no_zero ? 1.0f : 0.0f;
        cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                      m, n, k,
                      &alpha,
                      temp_weight, CUDA_R_16BF, w->width,
                      x, CUDA_R_16BF, w->width,
                      &beta,
                      out, CUDA_R_16BF, m);
        // printf("returned value %s\n",cublasGetStatusString(val));
    }
    else
    {
        if (!skip_reconstruct)
        {
            w->reconstruct_exllama_v2(temp_weight, block_kn_size, alt_stream);
        }
        const half alpha = __float2half(1.0f);
        const half beta = no_zero ? __float2half(1.0f) : __float2half(0.0f);
        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    m, n, k,
                    &alpha,
                    temp_weight, w->width,
                    x, w->width,
                    &beta,
                    out, m);
    }
}

// void q4_t_matmul_cuda(
//     ExLlamaTuning *tuningParams,
//     const half *x,
//     const int x_batch_size,
//     const Q4Matrix *w,
//     half *out,
//     bool no_zero)
// {
//     int infeatures = w->height;
//     int outfeatures = w->width;
//     // printf("infeatures=%d, outfeatures=%d, x_batch_size=%d\n",infeatures,outfeatures,x_batch_size);

//     cudaSetDevice(w->device);
//     CudaBuffers *buffers = get_buffers(w->device);

//     const half *x_mapped = x;
//     int block_size_x = 16;
//     dim3 threads(1, THREADS_Y, T_THREADS_Z);

//     dim3 blocks(
//         (outfeatures + block_size_x - 1) / block_size_x,
//         (x_batch_size + threads.y - 1) / threads.y,
//         (infeatures + (T_Z_BLOCKSIZE * T_THREADS_Z) - 1) / (T_Z_BLOCKSIZE * T_THREADS_Z));

//     auto kernel = q4_t_matmul_kernel<true>;

//     if (tuningParams->matmul_no_half2)
//     {
//         kernel = q4_t_matmul_kernel<false>;
//     }

//     half *out_to_restore = out;
//     if (w->cuda_x_map)
//     {
//         out_to_restore = buffers->temp_state;
//     }

//     kernel<<<blocks, threads>>>(
//         x_mapped,
//         w->cuda_qweight,
//         out_to_restore,
//         w->cuda_scales, w->cuda_qzeros,
//         x_batch_size, infeatures, outfeatures,
//         w->groupsize,
//         block_size_x,
//         no_zero);

//     if (w->cuda_x_map)
//     {
//         // printf("cuda x restore\n");
//         column_restore_cuda(out_to_restore, out, x_batch_size, w->height, w->cuda_x_map);
//     }
//     else
//     {
//         // printf("no cuda x restore\n");
//     }
// }

// template <bool use_half2>
// __global__ void q4_t_matmul_kernel(
//     const half *__restrict__ x,
//     const uint32_t *__restrict__ w,
//     half *__restrict__ out,
//     const half *__restrict__ w_scales,
//     const uint32_t *__restrict__ w_zeros,
//     const int x_batch_size,
//     const int w_infeatures,
//     const int w_outfeatures,
//     const int groupsize,
//     const int block_size_x,
//     bool no_zero)
// {
//     // Start of block
//     // w -> w.T so x@w.T, x.cols==w.cols

//     // int x_column = block_size_z * blockIdx.z;
//     // int x_column_end = min(dim, block_size_z * (blockIdx.z + 1));
//     int col = blockIdx.x * block_size_x;
//     int col_end = min(w_outfeatures, col + block_size_x);

//     // int w_column = THREADS_X * blockIdx.x + threadIdx.x;
//     int w_row = (T_THREADS_Z * blockIdx.z + threadIdx.z) * T_Z_BLOCKSIZE;
//     int w_row_end = min(w_infeatures, w_row + T_Z_BLOCKSIZE);
//     int x_row = THREADS_Y * blockIdx.y + threadIdx.y;

//     // Views

//     MatrixView_half x_(x, x_batch_size, w_outfeatures);
//     MatrixView_half w_scales_(w_scales, w_infeatures / groupsize, w_outfeatures);
//     MatrixView_q4_row w_zeros_(w_zeros, w_infeatures / groupsize, w_outfeatures);
//     MatrixView_q4_column w_(w, w_infeatures, w_outfeatures);
//     MatrixView_half_rw out_(out, x_batch_size, w_infeatures);

//     // Zero output
//     if (!no_zero && blockIdx.x == 0)
//     {
//         for (int w_row_out_col = w_row; w_row_out_col < w_row_end; w_row_out_col += 2)
//         {
//             *((uint32_t *)out_.item_ptr(x_row, w_row_out_col)) = 0;
//         }
//         __syncthreads();
//     }

//     if constexpr (use_half2)
//     {
//         // half2
//         half2 out_cols_acc[T_Z_BLOCKSIZE / 2];
//         for (int i = 0; i < T_Z_BLOCKSIZE; i++)
//         {
//             *((uint32_t *)(&out_cols_acc[i])) = 0;
//         }
//         for (int col_out_acc = col; col_out_acc < col_end; col_out_acc++)
//         {
//             int group = w_row / groupsize;
//             const half2 w_scale = w_scales_.item_half2half2(group, col_out_acc);
//             uint32_t w_zero = w_zeros_.item(group, col_out_acc) + 1;
//             const half2 x_read = x_.item_half2half2(x_row, col_out_acc);
//             const uint32_t *w_ptr = (const uint32_t *)w_.item_uint32_ptr(w_row, col_out_acc);
//             // for (int w_row_out_col=w_row;w_row_out_col<w_row_end;w_row_out_col+=8){
//             for (int w_row_out_col_i = 0; w_row_out_col_i < w_row_end - w_row; w_row_out_col_i += 8)
//             {
//                 uint32_t w_read = *w_ptr;
//                 w_ptr += w_outfeatures;
//                 half w_0 = __int2half_rn((int)((w_read) & 0x0f) - w_zero);
//                 half w_1 = __int2half_rn((int)((w_read >> 4) & 0x0f) - w_zero);
//                 half w_2 = __int2half_rn((int)((w_read >> 8) & 0x0f) - w_zero);
//                 half w_3 = __int2half_rn((int)((w_read >> 12) & 0x0f) - w_zero);
//                 half w_4 = __int2half_rn((int)((w_read >> 16) & 0x0f) - w_zero);
//                 half w_5 = __int2half_rn((int)((w_read >> 20) & 0x0f) - w_zero);
//                 half w_6 = __int2half_rn((int)((w_read >> 24) & 0x0f) - w_zero);
//                 half w_7 = __int2half_rn((int)((w_read >> 28)) - w_zero);
//                 half2 w_01 = __halves2half2(w_0, w_1);
//                 half2 w_23 = __halves2half2(w_2, w_3);
//                 half2 w_45 = __halves2half2(w_4, w_5);
//                 half2 w_67 = __halves2half2(w_6, w_7);
//                 out_cols_acc[w_row_out_col_i / 2 + 0] = __hfma2(x_read, __hmul2(w_01, w_scale), out_cols_acc[w_row_out_col_i / 2 + 0]);
//                 out_cols_acc[w_row_out_col_i / 2 + 1] = __hfma2(x_read, __hmul2(w_23, w_scale), out_cols_acc[w_row_out_col_i / 2 + 1]);
//                 out_cols_acc[w_row_out_col_i / 2 + 2] = __hfma2(x_read, __hmul2(w_45, w_scale), out_cols_acc[w_row_out_col_i / 2 + 2]);
//                 out_cols_acc[w_row_out_col_i / 2 + 3] = __hfma2(x_read, __hmul2(w_67, w_scale), out_cols_acc[w_row_out_col_i / 2 + 3]);
//             }
//         }
//         for (int w_row_out_col_i = 0; w_row_out_col_i < w_row_end - w_row; w_row_out_col_i += 2)
//         {
//             atomicAdd(out_.item_ptr(x_row, w_row + w_row_out_col_i), out_cols_acc[w_row_out_col_i / 2].x);
//             atomicAdd(out_.item_ptr(x_row, w_row + w_row_out_col_i + 1), out_cols_acc[w_row_out_col_i / 2].y);
//         }
//     }
//     else
//     {
//         // no half2
//         half out_cols_acc[T_Z_BLOCKSIZE];
//         for (int i = 0; i < T_Z_BLOCKSIZE; i += 2)
//         {
//             *((uint32_t *)(&out_cols_acc[i])) = 0;
//         }
//         // memset((void*)out_cols_acc, 0, (w_row_end-w_row)*sizeof(half));
//         for (int col_out_acc = col; col_out_acc < col_end; col_out_acc++)
//         {
//             int group = w_row / groupsize;
//             half w_scale = w_scales_.item(group, col_out_acc);
//             uint32_t w_zero = w_zeros_.item(group, col_out_acc) + 1;
//             const half x_read = x_.item(x_row, col_out_acc);
//             const uint32_t *w_ptr = (const uint32_t *)w_.item_uint32_ptr(w_row, col_out_acc);
//             // for (int w_row_out_col=w_row;w_row_out_col<w_row_end;w_row_out_col+=8){
//             for (int w_row_out_col_i = 0; w_row_out_col_i < w_row_end - w_row; w_row_out_col_i += 8)
//             {
//                 uint32_t w_read = *w_ptr;
//                 w_ptr += w_outfeatures;
//                 half w_0 = __int2half_rn((int)((w_read) & 0x0f) - w_zero);
//                 half w_1 = __int2half_rn((int)((w_read >> 4) & 0x0f) - w_zero);
//                 half w_2 = __int2half_rn((int)((w_read >> 8) & 0x0f) - w_zero);
//                 half w_3 = __int2half_rn((int)((w_read >> 12) & 0x0f) - w_zero);
//                 half w_4 = __int2half_rn((int)((w_read >> 16) & 0x0f) - w_zero);
//                 half w_5 = __int2half_rn((int)((w_read >> 20) & 0x0f) - w_zero);
//                 half w_6 = __int2half_rn((int)((w_read >> 24) & 0x0f) - w_zero);
//                 half w_7 = __int2half_rn((int)((w_read >> 28)) - w_zero);
//                 out_cols_acc[w_row_out_col_i + 0] = __hfma(x_read, __hmul(w_0, w_scale), out_cols_acc[w_row_out_col_i + 0]);
//                 out_cols_acc[w_row_out_col_i + 1] = __hfma(x_read, __hmul(w_1, w_scale), out_cols_acc[w_row_out_col_i + 1]);
//                 out_cols_acc[w_row_out_col_i + 2] = __hfma(x_read, __hmul(w_2, w_scale), out_cols_acc[w_row_out_col_i + 2]);
//                 out_cols_acc[w_row_out_col_i + 3] = __hfma(x_read, __hmul(w_3, w_scale), out_cols_acc[w_row_out_col_i + 3]);
//                 out_cols_acc[w_row_out_col_i + 4] = __hfma(x_read, __hmul(w_4, w_scale), out_cols_acc[w_row_out_col_i + 4]);
//                 out_cols_acc[w_row_out_col_i + 5] = __hfma(x_read, __hmul(w_5, w_scale), out_cols_acc[w_row_out_col_i + 5]);
//                 out_cols_acc[w_row_out_col_i + 6] = __hfma(x_read, __hmul(w_6, w_scale), out_cols_acc[w_row_out_col_i + 6]);
//                 out_cols_acc[w_row_out_col_i + 7] = __hfma(x_read, __hmul(w_7, w_scale), out_cols_acc[w_row_out_col_i + 7]);
//             }
//         }
//         for (int w_row_out_col_i = 0; w_row_out_col_i < w_row_end - w_row; w_row_out_col_i++)
//         {
//             atomicAdd(out_.item_ptr(x_row, w_row + w_row_out_col_i), out_cols_acc[w_row_out_col_i]);
//         }
//     }
// }
