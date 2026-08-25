# KV-Cache Kernel Analysis Tools

This directory provides pickle analyzers for the parent benchmarks. Generated inputs and figures are not version-controlled.

## Files

- `analysis_utils.py`: throughput smoothing, interval means, speed figures, and CUDA allocator-trace parsing.
- `analysis_one.py`: read a selected pickle, print TPS for several ranges, and optionally plot speed/memory.
- `compare_kernel.py`: compare the contiguous vLLM cache with dual-table cache and export vLLM timing to `baseline.json`.

## Running

Both entry points use relative filenames in the current directory:

```bash
cd scripts/qllm/infer/cache_kernel
PYTHONPATH=../../../.. MPLBACKEND=Agg "${DROFELL_PY:-python}" analysis_one.py
```

Before running, inspect and correct:

- `target_file` in `analysis_one.py` defaults to `kernel_vllm_2k.pkl`;
- `compare_kernel.py` reads `kernel_vllm.pkl`, while the generator currently writes `kernel_vllm_2k.pkl`;
- `compare_kernel.py` requires both ordinary and memory dual-table outputs.

The scripts create matplotlib figures but do not call `savefig` consistently. Save explicitly in headless or archival use and record the generation configuration of the input pickle.
