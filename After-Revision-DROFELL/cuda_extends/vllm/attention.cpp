#include <torch/extension.h>
#include <c10/util/Optional.h>

void paged_attention_v1(
    torch::Tensor &out,
    torch::Tensor &query,
    torch::Tensor &key_blocks_ptrs,
    torch::Tensor &value_blocks_ptrs,
    torch::Tensor &head_mapping,
    float scale,
    torch::Tensor &block_tables,
    torch::Tensor &context_lens,
    int block_size,
    int max_context_len,
    const c10::optional<torch::Tensor> &alibi_slopes,
    int kv_head_stride);

void paged_attention_v2(
    torch::Tensor &out,
    torch::Tensor &exp_sums,
    torch::Tensor &max_logits,
    torch::Tensor &tmp_out,
    torch::Tensor &query,
    torch::Tensor &key_blocks_ptrs,
    torch::Tensor &value_blocks_ptrs,
    torch::Tensor &head_mapping,
    float scale,
    torch::Tensor &block_tables,
    torch::Tensor &context_lens,
    int block_size,
    int max_context_len,
    const c10::optional<torch::Tensor> &alibi_slopes,
    int kv_head_stride);

void paged_attention_v1_simplify(
    torch::Tensor &out,
    torch::Tensor &query,
    torch::Tensor &key_blocks_ptrs,
    torch::Tensor &value_blocks_ptrs,
    torch::Tensor &head_mapping,
    float scale,
    int layer_bytes_offset,
    torch::Tensor &context_lens,
    int block_size,
    int max_context_len,
    const c10::optional<torch::Tensor> &alibi_slopes,
    int kv_head_stride);

void paged_attention_v2_simplify(
    torch::Tensor &out,
    torch::Tensor &exp_sums,
    torch::Tensor &max_logits,
    torch::Tensor &tmp_out,
    torch::Tensor &query,
    torch::Tensor &key_blocks_ptrs,
    torch::Tensor &value_blocks_ptrs,
    torch::Tensor &head_mapping,
    float scale,
    int layer_bytes_offset,
    torch::Tensor &context_lens,
    int block_size,
    int max_context_len,
    const c10::optional<torch::Tensor> &alibi_slopes,
    int kv_head_stride);

void paged_attention_v1_orig(
    torch::Tensor &out,
    torch::Tensor &query,
    torch::Tensor &key_cache,
    torch::Tensor &value_cache,
    torch::Tensor &head_mapping,
    float scale,
    torch::Tensor &block_tables,
    torch::Tensor &context_lens,
    int block_size,
    int max_context_len,
    const c10::optional<torch::Tensor> &alibi_slopes);

void paged_attention_v2_orig(
    torch::Tensor &out,
    torch::Tensor &exp_sums,
    torch::Tensor &max_logits,
    torch::Tensor &tmp_out,
    torch::Tensor &query,
    torch::Tensor &key_cache,
    torch::Tensor &value_cache,
    torch::Tensor &head_mapping,
    float scale,
    torch::Tensor &block_tables,
    torch::Tensor &context_lens,
    int block_size,
    int max_context_len,
    const c10::optional<torch::Tensor> &alibi_slopes);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "paged_attention_v1",
        &paged_attention_v1,
        "Compute the attention between an input query and the cached keys/values using PagedAttention.");
    m.def(
        "paged_attention_v2",
        &paged_attention_v2,
        "PagedAttention V2.");
    m.def(
        "paged_attention_v1_orig",
        &paged_attention_v1_orig,
        "Compute the attention between an input query and the cached keys/values using PagedAttention.");
    m.def(
        "paged_attention_v2_orig",
        &paged_attention_v2_orig,
        "PagedAttention V2.");
    m.def(
        "paged_attention_v1_simplify",
        &paged_attention_v1_simplify,
        "Compute the attention between an input query and the cached keys/values using PagedAttention.");
    m.def(
        "paged_attention_v2_simplify",
        &paged_attention_v2_simplify,
        "PagedAttention V2.");
}
