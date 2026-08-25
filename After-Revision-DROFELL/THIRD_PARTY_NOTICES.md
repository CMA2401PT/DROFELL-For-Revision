# Third-Party Notices

DROFELL includes modified source code derived from the projects listed below.
Unless a file or directory states otherwise, source code authored for DROFELL
is licensed under the Apache License 2.0 in the repository-root `LICENSE`.
Third-party and derivative components remain subject to their original
licenses and attribution requirements.

## vLLM and Marlin

- Project: <https://github.com/vllm-project/vllm>
- License: Apache License 2.0
- Local code: `cuda_extends/vllm/`, `scripts/qllm/infer/vllm_algo/`, and
  portions of `operators/core/sampler/`
- License text: `cuda_extends/vllm/LICENSE`
- Source baselines and modification scope for the CUDA operators:
  `cuda_extends/vllm/README.md`

The vendored operators have been reduced, reorganized, or modified for
DROFELL. Marlin-derived files retain their applicable Marlin and Neural Magic
copyright and attribution notices.

## ExLlama

- Project: <https://github.com/turboderp/exllama>
- License: MIT License
- Local code: portions of `cuda_extends/exllama/`
- License text: `cuda_extends/exllama/LICENSE.exllama`
- Source baseline and modification scope: `cuda_extends/exllama/README.md`

The source has been substantially modified for DROFELL.

## ExLlamaV2

- Project: <https://github.com/turboderp-org/exllamav2>
- License: MIT License
- Local code: portions of `cuda_extends/exllama/`
- License text: `cuda_extends/exllama/LICENSE.exllamav2`
- Source baseline and modification scope: `cuda_extends/exllama/README.md`

The source has been substantially modified for DROFELL.

## External Runtime Dependencies

DROFELL uses the following projects as external runtime dependencies. Their
source code is not vendored as part of DROFELL except where a third-party
derivative is separately identified above.

- PyTorch: <https://github.com/pytorch/pytorch> (BSD-style project license;
  packaged distributions also include separately licensed third-party
  components)
- FlashAttention: <https://github.com/Dao-AILab/flash-attention>
  (BSD 3-Clause License)

Names and trademarks belong to their respective owners. Their use here is
solely to identify upstream projects and dependencies and does not imply
endorsement of DROFELL.
