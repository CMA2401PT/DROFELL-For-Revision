// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// DROFELL-specific reduced registration unit. Operator schemas and signatures
// are derived from vLLM's csrc/libtorch_stable/torch_bindings.cpp and ops.h.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>

void silu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input);

void rms_norm(torch::stable::Tensor& out, torch::stable::Tensor& input,
              torch::stable::Tensor& weight, double epsilon);

void fused_add_rms_norm(torch::stable::Tensor& input,
                        torch::stable::Tensor& residual,
                        torch::stable::Tensor& weight, double epsilon);

void rotary_embedding(torch::stable::Tensor& positions,
                      torch::stable::Tensor& query,
                      std::optional<torch::stable::Tensor> key,
                      int64_t head_size,
                      torch::stable::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse);

STABLE_TORCH_LIBRARY_FRAGMENT(_drofell_vllm_ops, ops) {
  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  ops.def(
      "rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) "
      "-> ()");
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query, Tensor!? key, "
      "int head_size, Tensor cos_sin_cache, bool is_neox, "
      "int rope_dim_offset=0, bool inverse=False) -> ()");
  ops.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, SymInt size_k, "
      "SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");
  ops.def(
      "marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
      "Tensor? b_bias_or_none, Tensor b_scales, Tensor? a_scales, "
      "Tensor? global_scale, Tensor? b_zeros_or_none, Tensor? g_idx_or_none, "
      "Tensor? perm_or_none, Tensor workspace, int b_type_id, "
      "SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, "
      "bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_drofell_vllm_ops, CUDA, ops) {
  ops.impl("silu_and_mul", TORCH_BOX(&silu_and_mul));
  ops.impl("rms_norm", TORCH_BOX(&rms_norm));
  ops.impl("fused_add_rms_norm", TORCH_BOX(&fused_add_rms_norm));
  ops.impl("rotary_embedding", TORCH_BOX(&rotary_embedding));
}
