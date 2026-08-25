// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#ifndef _q4_matmul_cuh
#define _q4_matmul_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#include "q4_matrix.cuh"

// const int THREADS_X = 32;       // Block size and thread count along columns in w and out
// const int THREADS_Y = 1;        // Block size and thread count along rows in x and out

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
    const int threads_y, // 1 Block size and thread count along rows in x and out
    const int block_size_z,
    const int matmul_group_step,
    const int x_remap_block_size_x,
    const int x_remap_block_size_y,
    bool skip_remap,
    cudaStream_t alt_stream=NULL
);

void q4_matmul_cuda_v2
(
    const half* x, // a
    const int x_height,
    const Q4Matrix* w, // b
    half* out, // c 
    const int block_kn_size,
    const int block_m_size_max,
    bool no_zero,
    cudaStream_t alt_stream=NULL
);

    // int block_size_z;
    // if (w->width == 4096) block_size_z = 384;           // 7B
    // else if (w->width == 11008) block_size_z = 256;
    // else if (w->width == 5120) block_size_z = 384;      // 13B
    // else if (w->width == 13824) block_size_z = 256;
    // else if (w->width == 6656) block_size_z = 256;      // 33B
    // else if (w->width == 17920) block_size_z = 128;
    // else block_size_z = 256;

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
    cudaStream_t alt_stream=NULL
);

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
    cudaStream_t alt_stream=NULL
);

#endif
