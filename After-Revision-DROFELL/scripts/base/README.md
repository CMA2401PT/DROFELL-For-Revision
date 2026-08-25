# Basic Single-Machine Correctness Checks

These scripts directly assemble the head, decoder, tail, KV cache, and user models from `operators` to validate complete-model inference and fine-tuning before introducing Dealers or Orbit.

## Files

| File | Contents |
|---|---|
| `infer.py` | 7B inference, including no user model and P-Tuning/LoRA loading; prints sample output and approximate TPS |
| `bench.py` | Llama-2-7B-GPTQ eager decode-only benchmark |
| `train.py` | general training implementation and `TrainConfig`, including data, forward/backward, optimizer, checkpoints, loss, and memory records |
| `train_ptuning.py` | P-Tuning training configuration |
| `train_lora.py` | LoRA training configuration |
| `train_state_maintainer.py` | step/update counters, event triggers, and training statistics; not a standalone entry point |

## Running

From the repository root:

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" scripts/base/infer.py
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" scripts/base/bench.py --num-seqs 1 8
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" scripts/base/bench.py --num-seqs 1 8 --cuda-graph
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" scripts/base/train_ptuning.py
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" scripts/base/train_lora.py
```

Except for `bench.py`, these scripts have no CLI. Model scale, dtype, user-model files, dataset, and output directories are configured in source. Check:

- `target_model`, `num_blocks`, `use_ptuning`, generation length, and concurrency in `infer.py`;
- `bench.py` defaults to Seqs=1/8, one whole-request warmup per group, and three runs of `8 warmup + 200 measured` decode rounds; matching CLI options adjust these values and `--output` writes JSON;
- `bench.py` defaults to `--async-backend golang --busy-wait`; use `--no-busy-wait` to disable busy waiting, and the selected runtime is written to output JSON;
- `PtuningConfig`/`LoraConfig`, learning rate, batch, `event_dir`, and truncation length in training entry points;
- files referenced by `data.trained_model_path_of(...)` and `data.advertise_gen_train_path` exist.

## Output and acceptance

- Inference prints two sample sequences before a fixed-token throughput section; inspect semantic output manually.
- The benchmark reports aggregate decode throughput, per-token/per-batch-round latency, and median summaries. Timing excludes model loading, prefill, and decode warmup.
- Training writes `record.json`, periodic checkpoints, and a best checkpoint under each entry point's `event_dir`, according to `TrainConfig` event periods.
- Primary checks are decreasing loss, absence of NaN, and expected GPU-memory behavior.
- `infer.py` loads trained P-Tuning weights by default. To validate only the base model, modify `use_ptuning`, sample construction, and related data dependencies.
