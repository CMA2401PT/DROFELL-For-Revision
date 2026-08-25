from setuptools import setup
import torch  # type: ignore
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

verbose = True
# python setup.py build_ext --inplace

CUDA_FLAGS = [
    "-lineinfo",
    "-std=c++17",
    "-O3"
]
C_FLAGS = ["-O3", "-std=c++17"]
VLLM_STABLE_ABI_FLAGS = [
    "-DUSE_CUDA",
    "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
]
VLLM_CUDA_UNDEF_FLAGS = [
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-static-global-template-stub=false",
]


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
                    lib_file_path("column_remap.cu"),
                    lib_file_path("q4_matrix.cu"),
                    lib_file_path("q4_matmul.cu"),
                    lib_file_path("q4_transpose_matmul.cu"),
                    lib_file_path("half_matmul.cu"),
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


def build_marlin_extend():
    library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "marlin")
    lib_file_path = lambda *file: os.path.join(library_dir, *file)
    extension_name = "prebuilt_marlin_extend"
    setup(
        name=extension_name,
        ext_modules=[
            CUDAExtension(
                name=extension_name,
                sources=[lib_file_path("marlin_dequant.cu")],
                include_dirs=[library_dir],
                extra_compile_args={
                    "cxx": C_FLAGS,
                    "nvcc": CUDA_FLAGS,
                },
            )
        ],
        cmdclass={
            "build_ext": BuildExtension,
        },
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


def build_vllm_ops():
    """Build the project-owned subset of vLLM CUDA operators."""
    library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm")
    csrc_dir = os.path.join(library_dir, "csrc")
    libtorch_dir = os.path.join(csrc_dir, "libtorch_stable")
    marlin_dir = os.path.join(
        libtorch_dir,
        "quantization",
        "marlin",
    )
    extension_name = "prebuilt_vllm_ops"

    setup(
        name=extension_name,
        ext_modules=[
            CUDAExtension(
                name=extension_name,
                sources=[
                    os.path.join(library_dir, "ops_registration.cpp"),
                    os.path.join(libtorch_dir, "activation_kernels.cu"),
                    os.path.join(libtorch_dir, "layernorm_kernels.cu"),
                    os.path.join(libtorch_dir, "pos_encoding_kernels.cu"),
                    os.path.join(marlin_dir, "gptq_marlin_repack.cu"),
                    os.path.join(marlin_dir, "marlin.cu"),
                    os.path.join(
                        marlin_dir,
                        "sm80_kernel_float16_u4b8_float16.cu",
                    ),
                ],
                include_dirs=[csrc_dir, libtorch_dir],
                extra_compile_args={
                    "cxx": C_FLAGS + VLLM_STABLE_ABI_FLAGS,
                    "nvcc": (
                        CUDA_FLAGS
                        + VLLM_STABLE_ABI_FLAGS
                        + VLLM_CUDA_UNDEF_FLAGS
                    ),
                },
            )
        ],
        cmdclass={
            "build_ext": BuildExtension,
        },
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


BUILDERS = {
    "exllama": build_exllama_extend,
    "marlin": build_marlin_extend,
    "vllm_attention": build_vllm_attention,
    "vllm_cache": build_vllm_cache,
    "vllm_ops": build_vllm_ops,
    "serialize_helper": build_serialize_helper,
    "hack": build_hack,
}

selected_builder = os.environ.get("DROFELL_CUDA_EXT_ONLY")
if selected_builder is None:
    for builder in BUILDERS.values():
        builder()
elif selected_builder in BUILDERS:
    BUILDERS[selected_builder]()
else:
    raise ValueError(
        "unknown DROFELL_CUDA_EXT_ONLY value: "
        f"{selected_builder!r}; expected one of {sorted(BUILDERS)}"
    )
