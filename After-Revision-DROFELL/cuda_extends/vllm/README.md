# Vendored vLLM CUDA operators

This directory contains two independently traced groups of vLLM-derived CUDA
code used directly by DROFELL.

## Legacy attention and cache kernels

The original attention/cache copy was based on `vllm-project/vllm` commit
`0d578228ca220c120bb73316c1d35d078a5bd7b1` (2023-11-09,
`v0.2.1-38-g0d57822`). Although the preserved checkout as a whole is dirty,
its `csrc/` tree has no uncommitted changes, so this commit identifies the
relevant source baseline.

The following files remain byte-for-byte identical to that baseline:

- `attention/attention_dtypes.h`
- `attention/attention_generic.cuh`
- `attention/attention_utils.cuh`
- `attention/dtype_bfloat16.cuh`
- `attention/dtype_float16.cuh`
- `attention/dtype_float32.cuh`
- `dispatch_utils.h`

`attention/attention_kernels.cu`, `attention.cpp`, `cache.cpp`, and
`cache_kernels.cu` were substantially modified for DROFELL's pointer-table,
dynamic-block, simplified/original compatibility, and cache restore paths.

## Current fused and Marlin operators

The newer retained operator subset is:

- `silu_and_mul`
- `rms_norm`
- `fused_add_rms_norm`
- `rotary_embedding`
- `gptq_marlin_repack`
- `marlin_gemm` (FP16 + GPTQ uint4b8 on SM80+)

This source snapshot was taken from `vllm-project/vllm` commit
`70b84f0bcbb6d0a35b74b1035673a1c934089dbb`. The Marlin selector and generated
instantiations were reduced to the FP16 GPTQ path used by this repository.

The vendored files remain licensed under Apache License 2.0. See `LICENSE` in
this directory. The license file is byte-for-byte identical in both traced
vLLM checkouts. Existing per-file copyright and attribution notices have been
retained. Marlin files also retain their original Marlin/Neural Magic notices.
