# PEFT Multi-Path Recovery Experiment

## Purpose

Establish several training paths inside a Backbone, fail the active winner, and observe valid minibatches from surviving paths, physical recovery, and state consistency.

## Profiles

- `clean`: zero artificial path latency;
- `delayed`: add one-way latency to the slow path;
- failure points supplied through `--failure-times-s`;
- `--attention-backend`, token lengths, learning rate, and checkpoint period recorded in argument snapshots.

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"

PYTHONPATH="$PWD" "$DROFELL_PY" \
  exps/recover-analysis-peft/multi/run.py \
  --python "$DROFELL_PY" \
  --profile clean \
  --run-id peft-multipath-clean \
  --repeats 1 --samples 960 \
  --submission-interval-ms 250 \
  --grad-accum-steps 8 --ckpt-period 10 \
  --seed 20260814
```

Use another run ID for the delayed profile and set `--path-latency-ms` and failure points explicitly.

## Acceptance

- Multiple paths receive real work before failure.
- The target is the source of the latest accepted output.
- Minibatches produced by surviving paths during recovery are traceable.
- Duplicate results do not advance the version twice.
- Normal and Recover align by semantic minibatch.
- Both loss and state fingerprints are checked.
- One recovery is not extrapolated to repeated recovery.
- Path slowdown and complete absence of output are measured separately.

All output is generated locally and not version-controlled.
