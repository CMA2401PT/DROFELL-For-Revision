# Offline Inference Benchmark

`benchmark.py` is the single node-form inference entry point without an Orbit network. Model, devices, decoder splits, memory policy, workload, adapter, recovery, and graph settings are explicit CLI arguments.

`--measurement request` (default) runs a general request benchmark. `--measurement decode` runs a fixed number of decode-only rounds and excludes model loading, instance construction, prefill, and decode warmup. Example with one Brain and one 32-layer Binding:

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" \
  scripts/offline/benchmark/inference/benchmark.py \
  --measurement decode \
  --model teared-Llama-2-7B-GPTQ \
  --devices cuda:0 --decoder-splits 32 \
  --num-seqs 1 8 --runs 3 --warmup-requests 2 \
  --warmup-decode-tokens 8 --decode-tokens 200 \
  --max-concurrency 8 \
  --max-seq-len 256 \
  --decoder-backend marlin --cuda-graph --graph-budget-mib 64 \
  --async-backend golang --busy-wait --container nonasync \
  --output scripts/offline/benchmark/inference/results/decode_graph_7b_1brain_size32
```

Remove `--cuda-graph` for eager. `--container nonasync` excludes `SharedBaseContainer` blocking-scheduler dispatch. Decode mode intentionally excludes adapters, retrieval, and recovery because they alter the measured path; study them separately in request mode. `--max-concurrency` fixes the Brain merge/concurrency cap. When omitted, the largest `--num-seqs` preserves legacy behavior.

## 7B one-GPU example

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" \
  scripts/offline/benchmark/inference/benchmark.py \
  --measurement request \
  --model teared-Llama-2-7B-GPTQ \
  --devices cuda:0 --decoder-splits 32 \
  --max-seq-len 2048 --num-seqs 8 --max-tokens 1050 \
  --memory-policy static --decoder-backend marlin \
  --prompt "once upon a time" \
  --output scripts/offline/benchmark/inference/results/7b-1gpu
```

## 7B two-GPU example

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" \
  scripts/offline/benchmark/inference/benchmark.py \
  --measurement request \
  --model teared-Llama-2-7B-GPTQ \
  --devices cuda:0 cuda:1 --decoder-splits 16 16 \
  --max-seq-len 256 --num-seqs 8 --max-tokens 200 \
  --memory-policy static --decoder-backend marlin --cuda-graph \
  --graph-budget-mib 64 \
  --prompt "once upon a time" \
  --output scripts/offline/benchmark/inference/results/7b-2gpu
```

## 70B two-GPU example

This command makes every topology and memory-policy parameter of the old 70B profile explicit:

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" \
  scripts/offline/benchmark/inference/benchmark.py \
  --measurement request \
  --model teared-Llama-2-70B-GPTQ \
  --devices cuda:0 cuda:1 --decoder-splits 40 40 \
  --max-seq-len 4096 --num-seqs 1 --max-tokens 2100 \
  --memory-policy percentage --major-percentage 0.5 \
  --operate-reserved-mib 100 --unmanaged-reserved-mib 10 \
  --minor-min-mib 0 --minor-max-mib 262144 \
  --decoder-backend marlin --prompt "once upon a time" \
  --output scripts/offline/benchmark/inference/results/70b-2gpu
```

P-Tuning/LoRA inference also requires `--adapter-state`. View all options with:

```bash
python scripts/offline/benchmark/inference/benchmark.py --help
```

Multiple recovery events use one schedule:

```bash
--recover 10 20 30:split 40
```

This recovers at decode steps 10, 20, 30, and 40 and simulates a complete node split only at step 30. Omit `--recover` to disable recovery. Inference recovery requires pre-existing recoverable state, so steps must be positive and step 0 is rejected. Concurrent sequences execute the schedule independently. Steps must be unique and strictly increasing, with at most one `:split`.

The output directory contains `args.json` and per-token completion times in `result.json`. Analyze with:

```bash
python scripts/offline/benchmark/inference/analysis_utils.py <output-dir>
```
