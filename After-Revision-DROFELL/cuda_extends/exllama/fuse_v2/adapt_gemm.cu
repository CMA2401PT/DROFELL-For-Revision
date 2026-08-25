#include "../q4_matmul.cuh"
#include "../util.cuh"
#include "adapt_gemm.cuh"

#define CLEAR_N_SIZE 256

__global__ void clear_kernel
(
    half* __restrict__ c,
    const int size_m,
    const int size_n
)
{
    int m = blockIdx.y;
    int n = (blockIdx.x * CLEAR_N_SIZE + threadIdx.x) * 8;
    if (n >= size_n) return;
    int4* c_ptr = (int4*)(c + m * size_n + n);
    *c_ptr = {};
}

void clear_tensor_cuda
(
    half* c,
    int size_m,
    int size_n,
    cudaStream_t alt_stream
)
{
    return;
    dim3 blockDim, gridDim;
    blockDim.x = CLEAR_N_SIZE;
    blockDim.y = 1;
    gridDim.x = DIVIDE(size_n / 8, CLEAR_N_SIZE);
    gridDim.y = size_m;
    clear_kernel<<<gridDim, blockDim,0,alt_stream>>>(c, size_m, size_n);
}



void q4_matmul_auto_select
(
    cublasHandle_t cublas_handle,
    const half* a,
    Q4Matrix* b,
    half* c,
    int x_height,

    bool no_zero,
    half* temp_weight,
    const int block_kn_size,
    const int block_m_size_max,

    bool must_blas,
    int switch_threshold,
    cudaStream_t alt_stream
)
{
    if (x_height > switch_threshold || must_blas)
    {
        q4_matmul_cudablas_v2(a,x_height,b,c,temp_weight,cublas_handle,no_zero,block_kn_size,false,false,alt_stream);
    }
    else
    {
        q4_matmul_cuda_v2(a,x_height,b,c,block_kn_size,block_m_size_max,false,alt_stream);
    }
}