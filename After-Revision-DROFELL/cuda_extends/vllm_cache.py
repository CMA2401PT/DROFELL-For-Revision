import os
import torch  # type: ignore

GPU_TO_CPU = 0
CPU_TO_GPU = 1
GPU_TO_GPU = 2

try:
    from .prebuilt_vllm_cache import swap_blocks, swap_blocks_list, copy_blocks, reshape_and_cache, restore_from_cache  # type:ignore
    from .prebuilt_vllm_cache import swap_blocks_orig, copy_blocks_orig, reshape_and_cache_orig, restore_from_cache_orig  # type:ignore
    # from .prebuilt_vllm_cache import reshape_and_cache_simplify, restore_from_cache_simplify  # type:ignore
except Exception as e:
    from torch.utils.cpp_extension import load
    print("load prebuilt_vllm_cache err: ", e)
    extension_name = "rebuilt_vllm_cache"
    verbose = True
    library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm")

    def lib_file_path(*ps: str) -> str:
        return os.path.join(library_dir, *ps)
    load(
        name=extension_name,
        sources=[
            lib_file_path("cache.cpp"),
            lib_file_path("cache_kernels.cu"),
        ],
        extra_include_paths=[library_dir, lib_file_path(".")],
        verbose=verbose,
        extra_ldflags=[],
        extra_cuda_cflags=["-lineinfo", "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"]  # ,"-ftime-report", "-DTORCH_USE_CUDA_DSA"],
    )
    from rebuilt_vllm_cache import swap_blocks, swap_blocks_list, copy_blocks, reshape_and_cache, restore_from_cache  # type: ignore
    from rebuilt_vllm_cache import swap_blocks_orig, copy_blocks_orig, reshape_and_cache_orig, restore_from_cache_orig  # type: ignore
    # from rebuilt_vllm_cache import reshape_and_cache_simplify, restore_from_cache_simplify  # type: ignore
