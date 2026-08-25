// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#ifndef _q4_transpose_matmul_cuh
#define _q4_transpose_matmul_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#include "q4_matrix.cuh"

// const int T_THREADS_Z=32;
// const int T_Z_BLOCKSIZE=32;

// void q4_t_matmul_cuda
// (
//     const half* x,
//     const int x_batch_size,
//     const Q4Matrix* w,
//     half* out,
//     bool no_zero =false
// );

// void q4_t_matmul_recons_cuda
// (
//     const half* x,
//     const int x_height,
//     Q4Matrix* w,
//     half* out,
//     const cublasHandle_t handle,
//     bool no_zero = false
// );

void q4_transpose_matmul_blas
(
    const half* x,
    const int x_height,
    Q4Matrix* w,
    half* out,
    half* temp_weight,
    half* temp_x,
    const cublasHandle_t handle,
    bool no_zero,
    const int x_restore_block_size_x,
    const int x_restore_block_size_y,
    const int reconstruct_threads_x,
    const int reconstruct_threads_y,
    bool skip_remap,
    bool skip_reconstruct,
    bool bfloat16,
    cudaStream_t alt_stream=NULL
);

void q4_transpose_matmul_cudablas_v2(
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
    cudaStream_t alt_stream=NULL
);

#endif
