import os
import torch  # type: ignore
try:
    from .prebuilt_vllm_attention import paged_attention_v1, paged_attention_v2  # type: ignore
    from .prebuilt_vllm_attention import paged_attention_v1_simplify, paged_attention_v2_simplify  # type: ignore
    from .prebuilt_vllm_attention import paged_attention_v1_orig, paged_attention_v2_orig  # type: ignore
except Exception as e:
    from torch.utils.cpp_extension import load
    extension_name = "rebulit_vllm_attention"
    verbose = True
    library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm")

    def lib_file_path(*ps: str) -> str:
        return os.path.join(library_dir, *ps)
    load(
        name=extension_name,
        sources=[
            lib_file_path("attention.cpp"),
            lib_file_path("attention", "attention_kernels.cu"),
        ],
        extra_include_paths=[library_dir, lib_file_path(".")],
        verbose=verbose,
        extra_ldflags=[],
        extra_cuda_cflags=["-lineinfo", "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"]  # ,"-ftime-report", "-DTORCH_USE_CUDA_DSA"],
    )
    from rebulit_vllm_attention import paged_attention_v1, paged_attention_v2  # type: ignore
    from rebulit_vllm_attention import paged_attention_v1_simplify, paged_attention_v2_simplify  # type: ignore
    from rebulit_vllm_attention import paged_attention_v1_orig, paged_attention_v2_orig  # type: ignore
