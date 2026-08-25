# Early Online Inference Throughput and Trace

`main.py` starts local Orbit with one Aux, two Dealers colocated in process 0 on `cuda:0`, and one Agent. The Agent first runs examples, then runs eight concurrent 128-token sequences while recording token times and detailed traces.

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" scripts/online/speed/main.py
```

Output is `scripts/online/speed/out.json`:

- `trace`: token-completion times for each sequence;
- `detail`: per-sequence, per-step event timeline.

`analysis_utils.py` provides TPS, timeline, blocking/parallel statistics, and plotting functions.

## Analysis

`analysis.py` reads `out.json` from its own directory:

```bash
PYTHONPATH="$PWD" MPLBACKEND=Agg "${DROFELL_PY:-python}" scripts/online/speed/analysis.py
```

The notebook-style script calls `fig.show()` / `plt.show()` and has no unified output writer.

Both Dealers share one GPU. The case is useful for observing scheduling concurrency but does not represent two-GPU or real-network throughput. Use adjacent `benchmark/` for controlled runs.
