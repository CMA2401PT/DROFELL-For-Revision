# vLLM Local Decode Benchmark

## Purpose

Run vLLM GPTQ-Marlin decode on one GPU without failures or networking as an independent local execution reference. The experiment has no DROFELL topology, recovery-delay, or failure-probability dimension.

## Protocol

- use a local unsharded GPTQ checkpoint;
- FP16 and GPTQ-Marlin;
- CUDA graphs enabled;
- whole-request warmup before measuring each token-length class;
- exclude prefill and the first visible token from decode measurement;
- use one uclamp configuration for every benchmark process.

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export VLLM_MODEL="${VLLM_MODEL:?set VLLM_MODEL to an unsplit checkpoint}"

/usr/bin/uclampset -m 1024 -- "$DROFELL_PY" \
  exps/vllm_upper_bound/run.py \
  --python "$DROFELL_PY" \
  --model "$VLLM_MODEL" \
  --run-id vllm-local-decode \
  --gpu 0 --runs 10 --warmup-requests 2
```

After interruption, use `--resume` only when model, environment, and arguments are unchanged.

## Output and acceptance

Per-repeat records, logs, environment metadata, model hash, and automatic summaries are local generated files and are not version-controlled.

Check that:

- the engine actually selects GPTQ-Marlin;
- eager is disabled and graph capture size matches the request;
- warmup is outside measurement;
- every length completes the required repeats;
- there are no failed requests, fallback, OOM, or CUDA errors;
- the Python executable remains the intended environment and does not resolve to the system interpreter.
