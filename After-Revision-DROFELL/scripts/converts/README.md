# Model Conversion Tools

This directory prepares models and qlinear weights for testing; it is not a performance benchmark.

## Workflow

1. `model.py` reads one GPTQ `model.safetensors`, writes embedding, norm, LM head, and each decoder layer separately under `convert_output/`, then copies tokenizer files and `config.json`.
2. `qlinear.py` extracts one or more quantized projections from a selected decoder layer in any sharded model for `scripts/module/qlinear/test.py`. By default it reads layer 0 from `split_qllm_models/teared-Llama-2-7B-GPTQ` and extracts only `q_proj`.

## Required edit before running

`orig_model_dir` in `model.py` currently uses the repository-relative path:

```text
models/Llama-2-7B-GPTQ
```

It will normally fail unchanged. Set it to the actual GPTQ model directory and verify that the directory contains `model.safetensors`, `config.json`, and four tokenizer files. The script asserts that non-decoder weights are exactly `lm_head.weight`, `model.embed_tokens.weight`, and `model.norm.weight`, so differently named or multi-shard models are incompatible without changes.

## Commands

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" scripts/converts/model.py
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" scripts/converts/qlinear.py
```

`model.py` still executes at import/top level and has no CLI or overwrite protection. `qlinear.py` is parameterized and refuses to overwrite by default; use `--overwrite` explicitly when replacement is intended.

Extract only `q_proj` from layer 0:

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" scripts/converts/qlinear.py \
  --model-dir split_qllm_models/teared-Llama-2-7B-GPTQ \
  --layer 0 --projections q_proj \
  --output-dir scripts/converts/qlinear_data
```
