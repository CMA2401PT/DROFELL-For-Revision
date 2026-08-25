#ifndef _q_attn_cuh
#define _q_attn_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

#include "../q4_matrix.cuh"

void attention_forward_qkv_stage(
    half *layernorm,
    float norm_epsilon,
    Q4Matrix *q_proj,
    Q4Matrix *k_proj,
    Q4Matrix *v_proj,
    half *temp_rms_output,
    half *temp_q_output,
    half *temp_k_output,
    half *temp_v_output,
    half *temp_weight,
    half *x,
    const int block_kn_size,
    const int block_m_size_max,
    const int switch_threshold,
    int q_len,
    int batch_size,
    cublasHandle_t cublas_handle,
    cudaStream_t alt_stream = NULL);
void attention_forward_rope_stage(
    half *temp_q_output,
    half *temp_k_output,
    const half *sin,
    const half *cos,
    int q_len,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int past_len,
    const uint32_t *past_lens,
    cudaStream_t alt_stream);
#endif