#ifndef _half_matmul_cuh
#define _half_matmul_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

void half_matmul_cublas_cuda(
    bool use_half2,
    bool no_zero,
    const half *x,
    const half *w,
    half *out,
    const int height,
    const int dim,
    const int width,
    cublasHandle_t handle,
    cudaStream_t alt_stream);

#endif