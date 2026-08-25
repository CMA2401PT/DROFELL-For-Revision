# Inference Concurrency and Device-Scaling Experiment

## Purpose

Under a fixed total-token budget, vary client concurrency, GPU count, and Agent placement to measure aggregate token throughput, capacity boundaries, and graph behavior.

## Protocol

- model: `teared-Llama-2-7B-GPTQ`;
- backend: Marlin;
- CUDA graph: enabled;
- scale 1/2/4/8 corresponds to layer splits 32, 16x2, 8x4, and 4x8;
- clients: explicit through `--clients`;
- deployment: `combined-request-agent`, `persistent-agent`, or both;
- tokens per client: `min(total_tokens // clients, cap)`.

Do not combine absolute throughput across different GPU models.

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export RUN_ID="${RUN_ID:-infer-scale}"

"$DROFELL_PY" exps/infer_scale/run.py smoke \
  --python "$DROFELL_PY" --run-id "$RUN_ID" \
  --scales 1 2 --clients 1 128 \
  --smoke-tokens 64 --timeout 3600

"$DROFELL_PY" exps/infer_scale/run.py run \
  --python "$DROFELL_PY" --run-id "$RUN_ID" \
  --scales 1 2 4 8 \
  --clients 1 2 4 8 16 32 64 128 \
  --total-tokens 35072 \
  --tokens-per-client-cap 36864 \
  --timeout 21600
```

Reduce `--scales` when fewer physical GPUs are available. Resume only with identical parameters and `--resume`. Analysis is available through `run.py analyze`.

## Output and acceptance

The runner writes environment metadata, logs, network specifications, per-cell output, and summaries to Git-ignored directories.

For every cell, check complete target token counts, equal per-client counts, monotonic timestamps, `logprobs=0`, actual graph replay, fallback status, and absence of OOM/CUDA errors/tracebacks. Do not mix smoke output with the complete matrix.
