#include <torch/extension.h>

#include <map>
#include <vector>

void swap_blocks(
    torch::Tensor &src_blocks_ptr,
    torch::Tensor &dst_blocks_ptr,
    const std::map<int64_t, int64_t> &block_mapping,
    int direction,
    int bytes_in_block);

void swap_blocks_list(
    const std::vector<std::tuple<int64_t, int64_t>> &ptr_pairs,
    int direction,
    int bytes_in_block,
    uintptr_t cuda_stream_i,
    bool use_specific_stream);

void copy_blocks(
    torch::Tensor &key_blocks_ptr,
    torch::Tensor &value_blocks_ptr,
    torch::Tensor &block_mapping,
    int elements_per_block,
    py::object dtype);

void reshape_and_cache(
    torch::Tensor &key,
    torch::Tensor &value,
    torch::Tensor &key_blocks_ptr,
    torch::Tensor &value_blocks_ptr,
    torch::Tensor &slot_mapping,
    int block_size,
    int x);

void restore_from_cache(
    torch::Tensor &key,
    torch::Tensor &value,
    torch::Tensor &key_blocks_ptr,
    torch::Tensor &value_blocks_ptr,
    torch::Tensor &slot_mapping,
    int block_size,
    int x);

void swap_blocks_orig(
    torch::Tensor &src,
    torch::Tensor &dst,
    const std::map<int64_t, int64_t> &block_mapping);

void copy_blocks_orig(
    std::vector<torch::Tensor> &key_caches,
    std::vector<torch::Tensor> &value_caches,
    const std::map<int64_t, std::vector<int64_t>> &block_mapping);

void reshape_and_cache_orig(
    torch::Tensor &key,
    torch::Tensor &value,
    torch::Tensor &key_cache,
    torch::Tensor &value_cache,
    torch::Tensor &slot_mapping);

void gather_cached_kv_orig(
    torch::Tensor &key,
    torch::Tensor &value,
    torch::Tensor &key_cache,
    torch::Tensor &value_cache,
    torch::Tensor &slot_mapping);

void restore_from_cache_orig(
    torch::Tensor &key,
    torch::Tensor &value,
    torch::Tensor &key_cache,
    torch::Tensor &value_cache,
    torch::Tensor &slot_mapping);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "swap_blocks",
        &swap_blocks,
        "Swap in (out) the cache blocks from src to dst");

    m.def(
        "swap_blocks_list",
        &swap_blocks_list,
        "Swap in (out) the cache blocks from src to dst, ptr is List[int]");

    m.def(
        "copy_blocks",
        &copy_blocks,
        "Copy the cache blocks from src to dst");
    m.def(
        "reshape_and_cache",
        &reshape_and_cache,
        "Reshape the key and value tensors and cache them");
    m.def(
        "restore_from_cache",
        &restore_from_cache,
        "invert operation of reshape and cache");
    // orig
    m.def(
        "swap_blocks_orig",
        &swap_blocks_orig,
        "Swap in (out) the cache blocks from src to dst");
    m.def(
        "copy_blocks_orig",
        &copy_blocks_orig,
        "Copy the cache blocks from src to dst");
    m.def(
        "reshape_and_cache_orig",
        &reshape_and_cache_orig,
        "Reshape the key and value tensors and cache them");
    m.def(
        "restore_from_cache_orig",
        &restore_from_cache_orig,
        "invert operation of reshape and cache");
}
