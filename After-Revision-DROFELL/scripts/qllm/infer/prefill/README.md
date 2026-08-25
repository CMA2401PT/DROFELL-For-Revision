# Single-Stream and Multi-Stream Prefill Analysis

Parent scripts `prefill_single.py` and `prefill_multi.py` generate `single.json` and `multi.json`. Record keys encode:

```text
<P-Tuning prefix length>+<prompt length> <concurrent request count>
```

`analysis.py` parses both JSON files, compares TPS across prefix/prompt sizes, and can display allocator memory traces.

Run from this directory for relative paths:

```bash
cd scripts/qllm/infer/prefill
PYTHONPATH=../../../.. MPLBACKEND=Agg "${DROFELL_PY:-python}" analysis.py
```

The current script is exported notebook-cell code. It calls `plt.show()` several times, fixes the y-axis to 8,200-9,500 TPS, and skips concurrency 2. Remove or adjust these display assumptions for other hardware or configurations, or the figure may be empty or misleading.
