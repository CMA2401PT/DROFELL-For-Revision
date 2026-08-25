# qLLM Inference and KV-Cache Experiments

## Core components

`infer_testbed.py` is the shared execution framework. `InferConfig` defaults to a sharded 7B model, P-Tuning weights, 640 cache blocks, the `exllama_fuse` decoder, and the `dual_table` cache kernel. It drives prefill and decode through the scheduler/manager under `vllm_algo/`.

Run two sample requests (base and user model) with:

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" scripts/qllm/infer/infer_testbed.py
```

## Benchmark entry points

| File | Configuration | Output |
|---|---|---|
| `kernel_our.py` | dual-table, concurrency 1/2/4/8, about 20K total tokens | `cache_kernel/kernel_dual_table.pkl` |
| `kernel_vllm.py` | contiguous vLLM cache, about 2,100 tokens | `cache_kernel/kernel_vllm_2k.pkl` |
| `kernel_our_memory.py` | dual-table, 128 tokens, CUDA memory history | `cache_kernel/kernel_dual_table_memory.pkl` |
| `70B_24L.py` | first 24 layers of 70B, P-Tuning, 1,024 blocks | `cache_kernel/kernel_dual_table_70B_24L.pkl` |
| `prefill_single.py` | `MULTI_PREFILL_STREAM=False` | `prefill/single.json` |
| `prefill_multi.py` | `MULTI_PREFILL_STREAM=True` | `prefill/multi.json` |

Example:

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" scripts/qllm/infer/kernel_our.py
```

Before the first run, reduce `total_tokens`, `blocks_count`, concurrency, or prompt length to fit the GPU. `kernel_our.py` and `kernel_vllm.py` currently use different total token counts; do not compare raw output without normalization and warmup review.

## Acceptance and risks

- `Report.generating_time_record` supplies throughput; memory mode stores a PyTorch allocator trace.
- These scripts have no golden-token comparison and primarily test completion plus performance/memory trends.
- `InferConfig.__post_init__` mutates decoder count in its model config; constructing different `max_layers_to_load` configurations in one process can contaminate shared state.
- `70B_24L.py` requires `ptuning_70B_24Layers.pt` and substantially more GPU memory and is not a smoke test.
