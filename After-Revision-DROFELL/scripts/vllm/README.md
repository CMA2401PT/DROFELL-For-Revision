# vLLM decode-only benchmark

`decode_benchmark.py` directly uses the vLLM engine API. Model loading,
prefill, and decode warmup are outside the measured interval. `--cuda-graph`
enables capture sizes matching `--num-seqs`; without it the script sets
`enforce_eager=True`. Speculative decoding is not configured. The
implementation is intentionally independent from the three DROFELL benchmark
scripts.

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  "${DROFELL_PY:-python}" \
  scripts/vllm/decode_benchmark.py \
  --num-seqs 1 8 --runs 3 --warmup-requests 2 \
  --warmup-decode-tokens 8 \
  --decode-tokens 200 --max-model-len 256 \
  --cuda-graph \
  --output scripts/vllm/results/decode_graph_7b.json
```

The reported throughput is aggregate output tokens/s. Inspect the `graph`
section in the result for vLLM's effective CUDA graph mode and capture sizes.
Run the same command without `--cuda-graph` for the eager baseline.
