# Binding, Asynchronous Mode, and Merge-Limit Experiment

## Purpose

Under one decode-only workload, vary Binding size, asynchronous execution mode, and the Brain merge limit independently and measure their effects on execution behavior and resource use.

## Variables

- Binding: 32, 16+16, or 8+8+8+8 layers;
- merge limit: 4, 8, or 16;
- asynchronous path: synchronous Python, Go blocking, or Go polling;
- deployment: persistent Agent or combined Request-Agent;
- concurrent sequences: defined by the runner's fixed matrix.

Keep model, prompt, warmup, decode rounds, graph configuration, and device fixed outside the target variable.

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export RUN_ID="${RUN_ID:-binding-async-merge}"

"$DROFELL_PY" exps/binding-async-merge/run.py plan

"$DROFELL_PY" exps/binding-async-merge/run.py all \
  --python "$DROFELL_PY" \
  --run-id "$RUN_ID"
```

Replace `all` with `baseline`, `online`, or `analyze` for staged execution. After interruption, use identical parameters plus `--resume`.

## Output and acceptance

The runner writes arguments, logs, network specifications, runtime metrics, and analysis tables to local Git-ignored directories. Do not version-control generated files.

Check that:

- graph execution is active and captured batches match configuration;
- graph fallback is zero or recorded explicitly;
- every case/deployment combination is complete;
- CPU/GPU sampling excludes model loading;
- Linux context switches are not interpreted as event-loop wakeups;
- GPU power is not interpreted as CPU-package or whole-system power.

`py + non-busy + NonAsyncContainer` means CUDA work bypasses the blocking queue; it does not mean the event-loop source lacks dual-class structure.
