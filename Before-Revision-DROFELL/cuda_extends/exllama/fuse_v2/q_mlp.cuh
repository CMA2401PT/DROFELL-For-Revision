#ifndef _q_mlp_cuh
#define _q_mlp_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#include "../q4_matrix.cuh"

void mlp_forward(
    half* layernorm,
    float norm_epsilon,
    Q4Matrix* gate,
    Q4Matrix* up,
    Q4Matrix* down,
    half* temp_rms_output,
    half* temp_gate_output,
    half* temp_up_output,
    half* temp_weight,
    half* x,
    int x_height,
    half* output,
    const int block_kn_size,
    const int block_m_size_max,
    const int switch_threshold,
    cublasHandle_t cublas_handle,
    cudaStream_t alt_stream=NULL
);

#endif