# Container Scheduling and Pressure-Model Experiments

This directory compares execution behavior of several containers under the `extend_asyncio` blocking scheduler and builds a pressure-prediction model from operation-level CPU/CUDA times.

## Entry points and stages

| File | Purpose |
|---|---|
| `train.py` / `infer.py` | select a container mode and run training/inference workloads |
| `train_base.py` | construct the training workload; not a standalone entry point |
| `analysis.py` | aggregate operation time and waiting time from `results/*.json`, generate figures and pressure tables |
| `solve_model.py` | fit a linear model to `model_solve.json` and visualize it |
| `containers/` | implementations of `fast`, `measure`, `solve`, and `pressure` |

Expected workflow:

1. `measure`: collect CPU submission and CUDA synchronization times by operation, producing `train_profile.json` or `infer_profile.json`.
2. `solve`: collect predicted load and actual completion time, produce `model_solve.json`, and fit coefficients.
3. `pressure`: use profiles and fitted parameters to predict GPU pressure and insert sleep/synchronization.
4. `fast`: provide a low-synchronization control.

## Defaults that cannot run unchanged

- `train.py` defaults to `mode="leak_detect"`, but that branch is commented out and raises `ValueError("leak_detect")`.
- `infer.py` defaults to `mode="pressure"` and requires an existing `infer_profile.json`.
- Both entry points write `results/` but do not create it.

First set `mode="measure"` in source, create the directory, and run:

```bash
mkdir -p scripts/base_container/results
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" scripts/base_container/train.py
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" scripts/base_container/infer.py
```

Changing modes requires editing source. `train.py` runs 50 steps; `infer.py` runs one 128-token sequence. Keep warmup, repeat count, and model configuration identical for comparisons.

Analyze generated JSON with:

```bash
PYTHONPATH="$PWD" MPLBACKEND=Agg "${DROFELL_PY:-python}" scripts/base_container/analysis.py
```

`analysis.py` generates PNG files and `<name>_S.py` next to `results/`. `solve_model.py` calls `plt.show()`; set `MPLBACKEND=Agg` in a headless environment.
