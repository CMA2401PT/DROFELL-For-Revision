# PEFT Single-Path Recovery Experiment

## Purpose

Without a redundant execution path, measure logical handoff, physical recovery, checkpoint replay, and training-state continuity after failure.

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export RESULT_ROOT="${RESULT_ROOT:-$PWD/tmp/peft-recovery-basic}"

PYTHONPATH="$PWD" "$DROFELL_PY" \
  exps/recover-analysis-peft/basic/run.py \
  --python "$DROFELL_PY" \
  --result-root "$RESULT_ROOT" \
  --repeats 1 \
  --samples 960 \
  --submission-interval-ms 250 \
  --minibatch-size 1 --grad-accum-steps 8 \
  --ckpt-period 10 \
  --adapter ptuning \
  --seed 20260814 \
  --case normal --case recover
```

`run_long_loss.py`, `run_sweep.py`, and `run_small_repro.py` package different durations and diagnostic settings. Read their source and `--help` first.

## Acceptance

- Normal and Recover use identical sample IDs and submission order.
- Failure target and timing take effect.
- Checkpoints cover complete handle state.
- Handoff, physical recovery, and ACK are recorded separately.
- Versions/minibatches have no gaps or duplicates.
- Loss and state fingerprints align by semantic minibatch.
- The first state divergence is identified by the earliest differing fingerprint, not scalar loss alone.
- A non-zero analysis exit status is distinguished from a run failure.

Store all generated files in Git-ignored directories.
