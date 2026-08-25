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

## Fused and Marlin operators

The newer retained operator subset is:

- `silu_and_mul`
- `rms_norm`
- `fused_add_rms_norm`
- `rotary_embedding`
- `gptq_marlin_repack`
- `marlin_gemm` (FP16 + GPTQ uint4b8 on SM80+)

The fused activation, normalization, and rotary sources use the vLLM
`v0.23.0` tag at commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`,
matching DROFELL's validated runtime environment. This matters for performance:
post-0.23 `main` generalized the activation and normalization kernels for APIs
that DROFELL does not use.

The stable-libtorch Marlin sources use commit
`70b84f0bcbb6d0a35b74b1035673a1c934089dbb`. Their selector and generated
instantiations were reduced to the FP16 GPTQ uint4b8 path used by this
repository. The reduced Marlin GEMM and repack outputs were validated against
the installed vLLM 0.23.0 operators.

The vendored files remain licensed under Apache License 2.0. See `LICENSE` in
this directory. The license file is byte-for-byte identical in both traced
vLLM checkouts. Existing per-file copyright and attribution notices have been
retained. Marlin files also retain their original Marlin/Neural Magic notices.
