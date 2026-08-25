# `scripts` Test and Analysis Index

This directory is not one unified test suite. Most files are research-oriented entry points with parameters embedded in source. Many scripts allocate substantial GPU memory, read sharded models, or remain alive while waiting for network events. Read the target directory README and entry-point source before running anything.

## Directory levels

| Directory | Scope | Typical use |
|---|---|---|
| `base/` | complete model on one machine | basic inference and P-Tuning/LoRA fine-tuning correctness |
| `base_container/` | scheduling containers | compare synchronous, measurement, and pressure-prediction container strategies |
| `converts/` | data preparation | split GPTQ models and extract quantized-linear weights |
| `module/` | individual modules | qlinear and decoder speed/GPU-memory experiments |
| `qllm/` | single-machine qLLM | vLLM-style inference, KV-cache kernels, prefill, and fine-tuning stability |
| `offline/` | one process, many coroutines | Dealers, Bindings, STM, memory twin, recovery, and scheduling |
| `online/` | Orbit-distributed | multi-process/multi-node correctness, throughput, and failure benchmarks |

A useful reading/debugging order is `module -> base/qllm -> offline -> online`. Each later level involves more system components and deployment prerequisites.

## Common runtime environment

Commands assume the repository root as the current directory:

```bash
export DROFELL_ROOT="$PWD"
export PYTHONPATH="$DROFELL_ROOT"
export DROFELL_PY="${DROFELL_PY:-python}"
```

Legacy absolute imports in `offline/basic` additionally require:

```bash
export PYTHONPATH="$DROFELL_ROOT:$DROFELL_ROOT/scripts/offline"
```

Common prerequisites:

- `split_qllm_models` points to a valid sharded-model library, and datasets/user models configured under `data/` exist.
- CUDA, `cuda_extends`, `operators/helpers/cuda_async`, and `extend_asyncio` have been built according to the root README.
- GPU model, count, and memory satisfy `cuda:N`, model-size, block-count, and concurrency settings in the script.
- Inspect source-level constants and environment variables first. Most entry points have no CLI and do not validate the complete configuration before startup.

## Operating guidelines

1. Run syntax or import checks before an entry point that loads a model.
2. Start with 7B, one GPU, low concurrency, and short token counts; restore the intended scale only after validating output.
3. Use an independent output directory for each run and retain commands, environment variables, GPU/driver metadata, Git state, and stdout/stderr.
4. Online setup, Orbit, and Dealers may remain alive. Before the next experiment, clean old processes, port `7241`, GPU memory, and network namespaces.
5. Some analyzers are notebook-style and use relative or historical fixed paths. Do not run them in bulk before reading the source.

## Current static-check status

AST parsing with the designated Python succeeds for all 140 `scripts/**/*.py` files. This establishes syntax only, not imports, model data, CUDA extensions, or runtime configuration.

Known broken or high-risk entry points:

- `offline/benchmark-basic/infer_fast.py` and `infer_retrieve.py` import the removed `scripts.offline.speed_test`.
- `offline/benchmark-basic/train_fast.py` imports `scripts.offline.testbed.train_node`, which the current package does not provide.
- `online/speed/analysis.py` uses an unexpanded `~` and the old repository name `island`; update the path before reading the current `out.json`.
- `base_container/train.py` defaults to `mode="leak_detect"`, whose implementation branch is commented out; it raises `ValueError` unchanged.
- Several plotting scripts call `plt.show()`. In a headless environment set `MPLBACKEND=Agg` and add `savefig` where needed.
