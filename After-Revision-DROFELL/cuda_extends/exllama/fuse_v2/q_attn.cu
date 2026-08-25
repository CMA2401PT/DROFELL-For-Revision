#include "q_mlp.cuh"
#include "../q4_matmul.cuh"
#include "rms_norm.cuh"
#include "../util.cuh"
#include "../matrix.cuh"
#include "adapt_gemm.cuh"
#include "rope.cuh"
// #include "lora.cuh"

// void rope_cuda
// (
//     half* x,
//     const half* sin,
//     const half* cos,
//     const int batch_size,
//     const int rows_per_batch,
//     const int head_dim,
//     const int num_heads,
//     const int past_len,
//     const uint32_t* past_lens
// )

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
    cudaStream_t alt_stream)
{
    int x_height = q_len * batch_size;
    int columns = q_proj->height;
    // int intermediate_size=gate->width;
    // temp_rms_output = rms(x;layernorm, norm_epsilon)
    rms_norm_cuda(x, layernorm, temp_rms_output, norm_epsilon, x_height, columns, alt_stream);
    q4_matmul_auto_select(cublas_handle, temp_rms_output, q_proj, temp_q_output, x_height, false, temp_weight, block_kn_size, block_m_size_max, false, switch_threshold, alt_stream);
    q4_matmul_auto_select(cublas_handle, temp_rms_output, k_proj, temp_k_output, x_height, false, temp_weight, block_kn_size, block_m_size_max, false, switch_threshold, alt_stream);
    q4_matmul_auto_select(cublas_handle, temp_rms_output, v_proj, temp_v_output, x_height, false, temp_weight, block_kn_size, block_m_size_max, false, switch_threshold, alt_stream);

    // rope_cuda(temp_q_output, sin, cos, batch_size, q_len * num_heads,    head_dim, num_heads,    past_len, past_lens, alt_stream);
    // rope_cuda(temp_k_output, sin, cos, batch_size, q_len * num_kv_heads, head_dim, num_kv_heads, past_len, past_lens, alt_stream);
}

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
    cudaStream_t alt_stream)
{
    rope_cuda(temp_q_output, sin, cos, batch_size, q_len * num_heads, head_dim, num_heads, past_len, past_lens, alt_stream);
    rope_cuda(temp_k_output, sin, cos, batch_size, q_len * num_kv_heads, head_dim, num_kv_heads, past_len, past_lens, alt_stream);
}

// QMLP::QMLP
// (
//     half* _layernorm,
//     float _norm_epsilon,
//     Q4Matrix* _gate,
//     Q4Matrix* _up,
//     Q4Matrix* _down,
//     half* _temp_state,
//     half* _temp_a,
//     half* _temp_b,
//     half* _temp_dq,
//     int _max_rows
// ):
//     layernorm(_layernorm),
//     norm_epsilon(_norm_epsilon),
//     gate(_gate),
//     up(_up),
//     down(_down),
//     temp_state(_temp_state),
//     temp_a(_temp_a),
//     temp_b(_temp_b),
//     temp_dq(_temp_dq),
//     max_rows(_max_rows)
// {
// }

// void QMLP::forward_
// (
//     cublasHandle_t cublas_handle,
//     half* x,
//     int rows,
//     int columns,
//     const std::vector<uintptr_t>& loras,
//     half* lora_temp
// )
// {
//     bool use_half2 = true;
//     int intermediate_size = gate->width;
//     // temp_state=rms(x,layernorm)
//     rms_norm_cuda(x, layernorm, temp_state, norm_epsilon, rows, columns);
//     // temp a = temp_state @ gate
//     gemm_half_q_half_cuda(cublas_handle, temp_state, gate, temp_a, rows, intermediate_size, columns, true, temp_dq);
//     // temp b = temp_state @ up
//     gemm_half_q_half_cuda(cublas_handle, temp_state, up,   temp_b, rows, intermediate_size, columns, true, temp_dq);

//     // apply_loras_cuda(cublas_handle, gate_proj_lora, loras, gate, temp_state, temp_a, lora_temp, rows);
//     // apply_loras_cuda(cublas_handle, up_proj_lora,   loras, up,   temp_state, temp_b, lora_temp, rows);

//     dim3 blockDim, gridDim;
//     blockDim.x = THREADS_X;
//     blockDim.y = THREADS_Y;
//     gridDim.x = DIVIDE(up->width, THREADS_X) / (use_half2 ? 2 : 1);
//     gridDim.y = DIVIDE(rows, THREADS_Y);

//     fp_silu_mul_kernel kernel = pick_silu_mul_kernel(use_half2);
//     // temp_a = silu(temp_a) * temp_b
//     kernel<<<gridDim, blockDim>>>(temp_a, temp_b, rows, intermediate_size);
//     // temp_a = temp_a @ down
//     gemm_half_q_half_cuda(cublas_handle, temp_a, down, x, rows, columns, intermediate_size, false, temp_dq);

//     // apply_loras_cuda(cublas_handle, down_proj_lora, loras, down, temp_a, x, lora_temp, rows);
// }
