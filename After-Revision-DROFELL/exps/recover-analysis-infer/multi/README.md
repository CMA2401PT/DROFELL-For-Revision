# Multi-Path Inference Recovery Experiment

## Purpose

Observe whether established paths continue producing valid output after the active path fails and how path-latency differences appear in inter-token gaps.

## Profiles

- **clean:** one failure and zero artificial path latency, checking surviving-path continuity;
- **delayed/mix:** extra latency on the standby path and several failures, exposing path-switch plateaus and checkpoint spikes.

Use independent run IDs and analyzers and do not mix criteria across profiles.

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"

PYTHONPATH="$PWD" "$DROFELL_PY" \
  exps/recover-analysis-infer/multi/run_clean.py \
  --python "$DROFELL_PY" \
  --run-id infer-multipath-clean \
  --repeats 3 --drop-step 80 --max-tokens 160

PYTHONPATH="$PWD" "$DROFELL_PY" \
  exps/recover-analysis-infer/multi/run.py \
  --python "$DROFELL_PY" \
  --run-id infer-multipath-delayed \
  --repeats 3 --drop-steps 40 100 160 220 \
  --max-tokens 256 --standby-latency-ms 3
```

## Acceptance

- Multiple paths receive work before failure.
- The target is the source of the latest accepted output.
- Surviving output sources and deduplication state are traceable.
- Physical-recovery and token-completion windows are computed separately.
- Slowdown from path latency is not counted as a complete stall.
- The legacy one-ITI/recovery-duration ratio is not used as masking fraction.
- Normal and Recover token sequences agree.
- Graph fallback, checkpoint/replay mode, and failed attempts are explicit.

All output is generated locally and not version-controlled.
