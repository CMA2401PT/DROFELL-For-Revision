# P-Tuning Numerical-Stability Experiments

`train_testbed.py` assembles the quantized 7B foundation model with P-Tuning and records loss, speed, GPU memory, checkpoints, and NaN events. The other five files are fixed experiment configurations.

| Entry point | Primary difference |
|---|---|
| `err_eps.py` | float16, AdamW `eps=1e-8`, for observing post-convergence numerical collapse |
| `common.py` | conventional stable float16 baseline with `eps=1e-4` |
| `bfloat16.py` | bfloat16 throughout |
| `bfloat16_start.py` | bfloat16 initially, then float16 |
| `nan_remove.py` | float16 with periodic NaN-weight detection and handling |

Run one entry point from the repository root:

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" scripts/qllm/train/common.py
```

Each entry point performs a complete training run directly and has no CLI. Inspect dataset, batch, forced padding, learning rate, `bfloat16_end`, event periods, and output directory in `TrainConfig`. Experiments default to `NonAsyncContainer` and the current CUDA stream.

Output goes to `records/<experiment>/`, normally including `record.json` and checkpoints. When comparing dtype or epsilon, keep data, initialization, batch, step, and decoder implementation fixed. The scripts do not fix random seeds, so one loss curve is not a deterministic regression oracle.
