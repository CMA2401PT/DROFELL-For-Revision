# Offline Training Benchmark

`benchmark.py` is the general node-form training entry point without an Orbit network. `--grad-accum-steps` is the number of minibatches per optimizer update; the effective update batch for one instance is `minibatch-size × grad-accum-steps`.

## 7B one-GPU P-Tuning example

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" \
  scripts/offline/benchmark/train/benchmark.py \
  --model teared-Llama-2-7B-GPTQ \
  --devices cuda:0 --decoder-splits 32 --max-seq-len 256 \
  --instances 2 --samples-per-instance 100 \
  --minibatch-size 2 --grad-accum-steps 2 --sample-len 128 \
  --adapter ptuning --ptuning-prefix-len 128 \
  --optimizer AdamW --learning-rate 0.01 --optimizer-eps 0.0001 \
  --memory-policy static --decoder-backend marlin \
  --output scripts/offline/benchmark/train/results/7b-ptuning
```

## 70B two-GPU P-Tuning example

This command makes the old 70B profile's two-GPU topology and memory policy explicit:

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" \
  scripts/offline/benchmark/train/benchmark.py \
  --model teared-Llama-2-70B-GPTQ \
  --devices cuda:0 cuda:1 --decoder-splits 40 40 --max-seq-len 4096 \
  --instances 1 --samples-per-instance 64 \
  --minibatch-size 4 --grad-accum-steps 4 --sample-len 128 \
  --adapter ptuning --ptuning-prefix-len 32 \
  --optimizer AdamW --learning-rate 0.01 --optimizer-eps 0.0001 \
  --memory-policy percentage --major-percentage 0.5 \
  --operate-reserved-mib 3072 --unmanaged-reserved-mib 500 \
  --minor-min-mib 512 --minor-max-mib 1024 \
  --decoder-backend marlin \
  --output scripts/offline/benchmark/train/results/70b-ptuning
```

LoRA uses `--adapter lora` plus `--lora-r-q`, `--lora-r-k`, `--lora-r-v`, and `--lora-len-limit`. View all options with:

```bash
python scripts/offline/benchmark/train/benchmark.py --help
```

Multiple recovery events use:

```bash
--backup-period 2 --recover 10 20 30:split 40
```

This recovers at training steps 10, 20, 30, and 40 and simulates a complete node split only at step 30. Omit `--recover` for no recovery; step 0 means recovery before the first minibatch. Every training instance executes the schedule independently. Steps follow minibatch submission order, not asynchronous completion order, must be unique and strictly increasing, and may contain at most one `:split`. A non-empty schedule requires a positive `--backup-period`.

The output directory contains:

- `args.json`: effective CLI arguments and derived values;
- `results.json`: per-instance and aggregate throughput/loss records;
- `brain.json`: per-device operation timings.

Analyze with:

```bash
python scripts/offline/benchmark/train/analysis_utils.py <output-dir>
```
