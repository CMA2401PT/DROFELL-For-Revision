# Offline Testbed

This package is the shared foundation for `offline/basic`, `offline/benchmark`, and `memory_modeling`. It has no executable entry point.

## Components

| File | Contents |
|---|---|
| `model_config.py` | sharded-model selection, loaded-layer count, maximum sequence length, and decoder-grouping function |
| `pipeline_parallel_testbed.py` | place model segments on one or more Brains/Providers and create inference/training instances |
| `infer_instance.py` | inference-node lifecycle, Bindings, KV/user-model data, steps, and failure Breakers |
| `train_instance.py` | training-node lifecycle, forward/backward, optimizer, checkpoints, and failure Breakers |
| `__init__.py` | public configuration and instance classes |

## Usage notes

- `ModelConfig.__post_init__` mutates layer/maximum-length fields in a loaded model config. Sequentially constructing different configurations in one process may affect later ones.
- `PipelineParallelTestbed.__del__` removes providers, but destruction timing is uncertain. Long tests should remove instances explicitly and let the event loop complete resource release.
- `Breaker` creates interruption/recovery at configured stages. Correctness tests must check output, state cleanup, and whether subsequent requests continue, not only whether an exception was caught.
- `InferTestConfig.recovery_replay` defaults to `"chunk"`: with a complete KV checkpoint, it coalesces uncovered contiguous decode input into one execution. `"sequential"` retains the old per-token path. Recovery without KV always uses merged-prefill fallback.
- The `testbed.py` and `TestBed` compatibility alias has been removed. Legacy entry points using `from testbed import TestBed` fail to import. The current interface is `PipelineParallelTestbed(brains: Sequence[Brain], cfg)`, including `[brain]` for one GPU.

See `pipeline_parallel_testbed.py`, migrated 70B entry points, and `scripts/offline/benchmark` for current usage. Adjacent legacy 7B basic entry points still require this migration.
