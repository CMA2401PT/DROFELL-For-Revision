# Vendored ExLlama CUDA code

This directory contains modified and reorganized CUDA code derived from both
ExLlama generations preserved when DROFELL's kernels were developed.

## Source baselines

- `turboderp/exllama` commit
  `3b013cd53c7d413cf99ca04c7c28dd5c95117c0d` (2023-09-12).
- `turboderp/exllamav2` commit
  `31f31e1b08eeccf4a5ab31fd202ef3100dce8d22` (2023-10-16,
  `v0.0.6-10-g31f31e1`).

The root q4 matrix/matmul, column-remap, half-matmul, matrix, compatibility,
and utility code primarily traces to ExLlama V1. The `fuse_v2` RMSNorm/RoPE
code, GPTQ GEMM kernel, and quantization/dequantization utilities primarily
trace to ExLlama V2. The fused attention/MLP dispatch, adaptive GEMM,
transpose-matmul/backward support, BF16 handling, bindings, and stream/cache
behavior were substantially modified or added for DROFELL.

No current C++/CUDA file in this directory is byte-for-byte identical to the
preserved upstream copies; this directory must therefore be treated as a
modified derivative, not an unmodified vendored snapshot.

Both upstream projects use the MIT License and their preserved license texts
are byte-for-byte identical. They are included separately as
`LICENSE.exllama` and `LICENSE.exllamav2` so both source lineages remain
explicit.
