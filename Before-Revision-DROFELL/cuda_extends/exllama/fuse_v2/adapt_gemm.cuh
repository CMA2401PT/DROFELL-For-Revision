#ifndef _adapt_gemm_cuh
#define _adapt_gemm_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#include "../q4_matrix.cuh"

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
    cudaStream_t alt_stream=NULL
);

void clear_tensor_cuda
(
    half* c,
    int size_m,
    int size_n,
    cudaStream_t alt_stream
);

#endif