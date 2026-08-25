# Minimal Reproduction of PEFT State Forking after Repeated Recovery

## Purpose

Use short training and a deterministic failure schedule to identify the first stage where Normal and Recover diverge. This entry point is diagnostic and not a performance measurement.

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"

PYTHONPATH="$PWD" "$DROFELL_PY" \
  exps/recover-analysis-peft/basic/run_small_repro.py \
  --python "$DROFELL_PY"
```

Consult `--help` for the complete interface. Use independent output directories for each backend, precision, and failure schedule.

## Comparison order

1. Sample ID and semantic minibatch.
2. Adapter-forward input/output.
3. Backward input and leaf gradients.
4. Parameters before and after optimizer update.
5. Optimizer/scheduler state.
6. Fingerprints before and after checkpoint dump/load.
7. Scalar loss.

In low precision, later scalar rounding can hide an early tensor difference, so loss is not the only criterion. Fingerprints and records are generated locally and not version-controlled.
