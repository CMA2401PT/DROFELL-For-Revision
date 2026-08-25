"""Project-owned wrappers for the vendored vLLM-derived CUDA operators."""

from __future__ import annotations

import os
from pathlib import Path

import torch


# Mirrors vLLM's scalar_types.uint4b8.id. The encoding is defined by the
# vendored csrc/core/scalar_type.hpp and is part of the Marlin operator ABI.
UINT4B8_ID = 1125899907892224


def _sources() -> list[str]:
    root = Path(__file__).resolve().parent / "vllm"
    libtorch = root / "csrc" / "libtorch_stable"
    marlin = libtorch / "quantization" / "marlin"
    return [
        str(root / "ops_registration.cpp"),
        str(libtorch / "activation_kernels.cu"),
        str(libtorch / "layernorm_kernels.cu"),
        str(libtorch / "pos_encoding_kernels.cu"),
        str(marlin / "gptq_marlin_repack.cu"),
        str(marlin / "marlin.cu"),
        str(marlin / "sm80_kernel_float16_u4b8_float16.cu"),
    ]


def _load_prebuilt() -> str:
    candidates = sorted(
        Path(__file__).resolve().parent.glob("prebuilt_vllm_ops*.so")
    )
    if not candidates:
        raise ImportError("prebuilt_vllm_ops shared library was not found")
    library = str(candidates[0])
    torch.ops.load_library(library)
    return library


try:
    _extension = _load_prebuilt()
except (ImportError, OSError, RuntimeError):
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent / "vllm"
    _extension = load(
        name="rebuilt_drofell_vllm_ops",
        sources=_sources(),
        extra_include_paths=[
            str(root / "csrc"),
            str(root / "csrc" / "libtorch_stable"),
        ],
        extra_cflags=[
            "-O3",
            "-std=c++17",
            "-DUSE_CUDA",
            "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
        ],
        extra_cuda_cflags=[
            "-lineinfo",
            "-O3",
            "-std=c++17",
            "-DUSE_CUDA",
            "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-static-global-template-stub=false",
        ],
        extra_ldflags=[],
        verbose=os.environ.get("DROFELL_CUDA_EXT_VERBOSE") == "1",
        is_python_module=False,
    )


_ops = torch.ops._drofell_vllm_ops


def silu_and_mul(output: torch.Tensor, input: torch.Tensor) -> None:
    _ops.silu_and_mul(output, input)


def rms_norm(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> None:
    _ops.rms_norm(output, input, weight, epsilon)


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> None:
    _ops.fused_add_rms_norm(input, residual, weight, epsilon)


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    rope_dim_offset: int = 0,
    inverse: bool = False,
) -> None:
    _ops.rotary_embedding(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
        rope_dim_offset,
        inverse,
    )


def gptq_marlin_repack(
    qweight: torch.Tensor,
    permutation: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    return _ops.gptq_marlin_repack(
        qweight,
        permutation,
        size_k,
        size_n,
        num_bits,
        is_a_8bit,
    )


def marlin_gemm(
    a: torch.Tensor,
    c: torch.Tensor | None,
    qweight: torch.Tensor,
    bias: torch.Tensor | None,
    scales: torch.Tensor,
    a_scales: torch.Tensor | None,
    global_scale: torch.Tensor | None,
    zeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    permutation: torch.Tensor | None,
    workspace: torch.Tensor,
    quant_type_id: int,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    return _ops.marlin_gemm(
        a,
        c,
        qweight,
        bias,
        scales,
        a_scales,
        global_scale,
        zeros,
        g_idx,
        permutation,
        workspace,
        quant_type_id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )
