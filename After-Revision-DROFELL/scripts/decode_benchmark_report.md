# Llama-2-7B Single-Node Decode-Only Benchmark

This document defines one protocol for the base, offline, online, and vLLM decode-only paths. It stores no existing measurements; each run writes to an independent Git-ignored directory.

## 1. Comparison scope

| Path | Entry point | Included layers |
|---|---|---|
| base | `scripts/base/bench.py` | direct model execution |
| offline | `scripts/offline/benchmark/inference/benchmark.py` | Binding, Brain, and memory model; no Orbit |
| online | `scripts/online/benchmark/benchmark_infer.py` | complete Agent/Request, Dealer, and Orbit path |
| vLLM | `scripts/vllm/decode_benchmark.py` | independent vLLM engine control |

All paths should use the same model, input tokens, sequence count, decode-token count, warmup, and sampling semantics. Model loading, prefill, graph capture, and first allocator initialization must remain outside the decode measurement window.

## 2. Common environment

Run from the repository root:

```bash
export PYTHONPATH="$PWD"
export DROFELL_PY="${DROFELL_PY:-python}"
export DROFELL_MODEL="${DROFELL_MODEL:-teared-Llama-2-7B-GPTQ}"
export DROFELL_RESULT_ROOT="${DROFELL_RESULT_ROOT:-$PWD/tmp/decode-benchmark}"
```

`DROFELL_RESULT_ROOT` should be Git-ignored. Record GPU, driver, CUDA, PyTorch, CPU governor, and working-tree state before running.

## 3. Fixed parameters

Set explicitly on every path:

- `num_seqs`;
- `runs` and warmup requests;
- warmup decode-token count;
- measured decode-token count;
- `max_model_len` / `max_seq_len`;
- eager or CUDA graph;
- greedy sampling without speculative decoding;
- decoder backend;
- online placement and network conditions.

Change one independent variable at a time and place graph and eager output in separate directories.

## 4. Base

Inspect current options:

```bash
"$DROFELL_PY" scripts/base/bench.py --help
```

Direct output to:

```text
$DROFELL_RESULT_ROOT/base/<mode>.json
```

The base path excludes Bindings, the Brain, and Orbit and is useful for checking model/backend decode behavior itself.

## 5. Offline

```bash
"$DROFELL_PY" scripts/offline/benchmark/inference/benchmark.py \
  --measurement decode \
  --model "$DROFELL_MODEL" \
  --devices cuda:0 --decoder-splits 32 \
  --num-seqs 1 8 --runs 3 --warmup-requests 2 \
  --warmup-decode-tokens 8 --decode-tokens 200 \
  --max-concurrency 8 --max-seq-len 256 \
  --decoder-backend marlin \
  --async-backend golang --busy-wait --container nonasync \
  --output "$DROFELL_RESULT_ROOT/offline/eager"
```

For the graph case, keep every other parameter unchanged and add:

```text
--cuda-graph --graph-budget-mib 64
```

Decode measurement intentionally excludes adapters, retrieval, and recovery.

## 6. Online

Generate and launch a logical network first:

```bash
"$DROFELL_PY" scripts/online/benchmark/generate_network.py --help
"$DROFELL_PY" scripts/online/benchmark/launch_network.py --help
"$DROFELL_PY" scripts/online/benchmark/benchmark_infer.py --help
```

Use one 32-layer Dealer or an explicitly documented sharded topology and record:

- whether Agent and Request are combined;
- Dealer process/thread/device placement;
- whether colocated delivery is used;
- graph budget and batch buckets;
- Orbit address and external network shaping.

After measurement, stop the Request but optionally keep serving for the next case. Restart the serving network when graph/backend configuration changes so that caches do not cross cases.

## 7. vLLM

```bash
CUDA_VISIBLE_DEVICES=0 "$DROFELL_PY" \
  scripts/vllm/decode_benchmark.py \
  --model "$DROFELL_MODEL" \
  --num-seqs 1 8 --runs 3 --warmup-requests 2 \
  --warmup-decode-tokens 8 \
  --decode-tokens 200 --max-model-len 256 \
  --output "$DROFELL_RESULT_ROOT/vllm/eager.json"
```

Add `--cuda-graph` and use another output file for the graph case. Inspect the effective graph mode and capture size in output rather than inferring graph use only from the CLI.

## 8. Output validation

For each case, check at least:

- the parameter snapshot matches actual argv;
- timing begins only after warmup;
- every sequence generates the same number of decode tokens;
- greedy output-token hashes agree across compatible paths;
- graph cases have no silent fallback, or fallback is recorded explicitly;
- the aggregate tokens/s denominator covers only measured decode;
- repeat count is complete and failed retries are not extra samples.

## 9. Analysis boundaries

- Base, offline, online, and vLLM cover different execution layers; interpret absolute values with that scope.
- Same-host, same-GPU runs can still be affected by residual processes, allocator cache, CPU scheduling, and temperature.
- Min/max from a small repeat count describes that run and is not a confidence interval.
- Result directories, JSON, logs, and figures are generated locally and are not version-controlled.
