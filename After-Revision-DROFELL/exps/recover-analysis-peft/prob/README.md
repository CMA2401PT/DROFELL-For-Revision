# PEFT Training Determinism Isolation Experiment

## Purpose

Compare attention/backend, shape, and deterministic-algorithm configurations in independent single-machine training and locate the first difference in training state.

## Variables

- attention backend;
- query/response token lengths;
- gradient accumulation;
- learning rate and optimizer epsilon;
- `torch.use_deterministic_algorithms`;
- GPU placement and seed.

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"

PYTHONPATH="$PWD" "$DROFELL_PY" \
  exps/recover-analysis-peft/prob/run.py \
  --python "$DROFELL_PY" \
  --run-id peft-determinism \
  --seed 20260814 \
  --minibatches 16 --grad-accum-steps 8 \
  --learning-rate 1e-4 --optimizer-eps 1e-4 \
  --query-tokens 64 --response-tokens 64
```

Use an independent run ID for each variable combination. `train_probe.py` and `optimizer_probe.py` capture finer-grained fingerprints.

## Acceptance

- Control runs use identical initial weights, inputs, and seeds.
- Every stage records a stable SHA-256 or equivalent fingerprint.
- Fingerprint and scalar-loss differences remain separate.
- A non-failing deterministic-algorithm mode does not imply determinism of every custom kernel.
- Findings from one shape do not transfer automatically to another.
- Single-machine root-cause isolation does not establish online recovery-state agreement.

Generated JSON, fingerprints, logs, and figures are not version-controlled.
