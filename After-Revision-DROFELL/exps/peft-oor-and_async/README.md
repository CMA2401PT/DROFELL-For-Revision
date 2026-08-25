# PEFT Out-of-Order Dispatch and Asynchronous Polling Ablation

## Purpose

Use an orthogonal 2x2 ablation to enable or disable out-of-order dispatch (OOO) and the asynchronous CUDA completion poller, measuring training throughput and outstanding work on one device.

## Matrix

| Case | OOO | Async poll | Runtime |
|---|---|---|---|
| `opt` | on | on | Go, busy wait |
| `no-ooo` | off | on | Go, busy wait |
| `no-async-poll` | on | off | Python, non-busy |
| `non-opt` | off | off | Python, non-busy |

The runner combines emission-delay, disable-OOO, and disable-async-poll patches from `patches/` in a temporary detached worktree without modifying product code in the main working tree.

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export RUN_ID="${RUN_ID:-peft-ooo-async}"

PYTHONPATH="$PWD" "$DROFELL_PY" exps/peft-oor-and_async/run.py all \
  --python "$DROFELL_PY" \
  --run-id "$RUN_ID" \
  --samples-per-client 256
```

Use fewer samples and a separate run ID for smoke. Resume only when arguments, patches, and the base commit are unchanged.

## Interpretation

- Training does not use CUDA graphs.
- The async switch changes both the runtime backend and the scheduling barrier around CUDA waits.
- `disable-ooo.patch` changes the request-emission window, not the runtime reorder controller.
- Overlapping CPU submission intervals in detailed traces establish multiple outstanding tasks, not necessarily simultaneous GPU-kernel execution on SMs.

## Output and acceptance

Generated worktrees, network specifications, logs, traces, runtime metrics, and figures are not version-controlled.

Check that:

- all four cases use the same model, topology, samples, and emission delay;
- benchmark and network sides use consistent async configuration;
- the temporary worktree contains the intended patches and is cleaned afterward;
- every case has a complete sample count;
- same-device overlap uses both device identity and time interval;
- known shutdown warnings remain distinct from fatal errors;
- one short smoke run is not extrapolated into stable performance.
