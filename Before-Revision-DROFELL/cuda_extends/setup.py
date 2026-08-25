from setuptools import setup
import torch  # type: ignore
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

verbose = True
# python setup.py build_ext --inplace

CUDA_FLAGS = [
    "-lineinfo",
    "-gencode", "arch=compute_86,code=sm_86",
    "-gencode", "arch=compute_86,code=compute_86",
    "-gencode", "arch=compute_89,code=sm_89",
    "-gencode", "arch=compute_89,code=compute_89",
    "-std=c++17",
    "-O3"
]
C_FLAGS = ["-O3", "-std=c++17"]


def build_exllama_extend():
    library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exllama")
    lib_file_path = lambda *file: os.path.join(library_dir, *file)
    extension_name = "prebuilt_exllama_extend"
    setup(
        name=extension_name,
        ext_modules=[
            CUDAExtension(
                name=extension_name,
                sources=[
                    lib_file_path("entry.cpp"),
                    lib_file_path("q4_matrix.cu"),
                    lib_file_path("column_remap.cu"),
                    lib_file_path("q4_matmul.cu"),
                    lib_file_path("q4_transpose_matmul.cu"),
                    lib_file_path("fuse_v2", "rms_norm.cu"),
                    lib_file_path("fuse_v2", "adapt_gemm.cu"),
                    lib_file_path("fuse_v2", "q_mlp.cu"),
                    lib_file_path("fuse_v2", "q_attn.cu"),
                    lib_file_path("fuse_v2", "rope.cu"),
                ],
                extra_include_paths=[library_dir, lib_file_path("fuse_v2")],
                verbose=verbose,
                extra_ldflags=[],
                extra_cuda_cflags=CUDA_FLAGS,
                extra_cflags=["-O3"]  # ,"-ftime-report", "-DTORCH_USE_CUDA_DSA"],
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )


def build_vllm_attention():
    library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm")
    lib_file_path = lambda *file: os.path.join(library_dir, *file)
    extension_name = "prebuilt_vllm_attention"

    setup(
        name=extension_name,
        ext_modules=[
            CUDAExtension(
                name=extension_name,
                sources=[
                    lib_file_path("attention.cpp"),
                    lib_file_path("attention", "attention_kernels.cu"),
                ],
                extra_include_paths=[library_dir, lib_file_path(".")],
                verbose=verbose,
                extra_ldflags=[],
                extra_cuda_cflags=CUDA_FLAGS,
                extra_cflags=C_FLAGS,
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )


def build_vllm_cache():
    library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm")
    lib_file_path = lambda *file: os.path.join(library_dir, *file)
    extension_name = "prebuilt_vllm_cache"

    setup(
        name=extension_name,
        ext_modules=[
            CUDAExtension(
                name=extension_name,
                sources=[
                    lib_file_path("cache.cpp"),
                    lib_file_path("cache_kernels.cu"),
                ],
                extra_include_paths=[library_dir, lib_file_path(".")],
                verbose=verbose,
                extra_ldflags=[],
                extra_cuda_cflags=CUDA_FLAGS,
                extra_cflags=C_FLAGS,
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )


def build_serialize_helper():
    extension_name = "prebuilt_serialize_helper"
    library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensor_serialize")

    setup(
        name="prebuilt_"+extension_name,
        ext_modules=[
            CUDAExtension(
                name=extension_name,
                sources=[
                    os.path.join(library_dir, "serialize_helper.cpp"),
                ],
                extra_include_paths=[],
                verbose=True,
                extra_ldflags=[],
                extra_cuda_cflags=CUDA_FLAGS,
                extra_cflags=C_FLAGS,
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )


def build_hack():
    library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hack")
    lib_file_path = lambda *file: os.path.join(library_dir, *file)
    extension_name = "prebuilt_hack_extend"

    setup(
        name=extension_name,
        ext_modules=[
            CUDAExtension(
                name=extension_name,
                sources=[
                    lib_file_path("entry.cpp"),
                ],
                extra_include_paths=[library_dir],
                verbose=verbose,
                extra_ldflags=[],
                extra_cuda_cflags=CUDA_FLAGS,  # type:ignore
                extra_cflags=C_FLAGS,
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )


build_exllama_extend()
build_vllm_attention()
build_vllm_cache()
build_serialize_helper()
build_hack()
