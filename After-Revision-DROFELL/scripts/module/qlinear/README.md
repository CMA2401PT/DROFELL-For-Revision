# QuantLinear Performance Experiments

`test.py` measures pure forward, PEFT forward, activation backward, and incremental CUDA allocator use for one local GPTQ projection. It supports DROFELL ExLlama cache/no-cache, Marlin, and legacy AutoGPTQ ExLlama/Triton/CUDA backends. Select a concrete path explicitly with `--method`.

Marlin has three forced diagnostic paths:

- `marlin`: packed INT4 direct forward, one reconstruction in backward;
- `marlin_dense_no_cache`: reconstruct + dense GEMM in forward, reconstruct again in backward;
- `marlin_dense_cache`: reconstruct + dense GEMM in forward and retain the weight for backward.

These modes also record `reconstruction` and `dense_forward_only` to separate reconstruction from cuBLAS GEMM cost. Ordinary decoders use `MarlinTuningParams` for token-count-based automatic dispatch.

The local Marlin reconstruction/dense-GEMM helper is prebuilt as `prebuilt_marlin_extend` by `cuda_extends/setup.py` and imported directly by `cuda_extends/marlin.py`; it does not compile dynamically through `TORCH_EXTENSIONS_DIR`. Before the first run:

```bash
cd cuda_extends
"${DROFELL_PY:-python}" setup.py build_ext --inplace
cd ..
```

AutoGPTQ `exllama_v1` and `exllama_v2` are forward-only. For PEFT backward use `our_cache`, `our_no_cache`, or one of the three Marlin diagnostic modes.

The input directory requires `config.json` and `<projection>.safetensors`. Extract them from a sharded model with:

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" scripts/converts/qlinear.py \
  --model-dir split_qllm_models/teared-Llama-2-7B-GPTQ \
  --layer 0 --projections q_proj \
  --output-dir scripts/converts/qlinear_data
```

Run one backend from the repository root:

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
  "${DROFELL_PY:-python}" scripts/module/qlinear/test.py \
  --input-dir scripts/converts/qlinear_data \
  --output scripts/module/qlinear/result.json \
  --method our_no_cache --projection q_proj \
  --seq-lens 1 16 128 512 2048 --warmup 5 --repeats 20
```

Timing uses CUDA Events. PEFT forward and backward are recorded separately; a separate memory sample does not enter timing. Run different backends in independent processes to avoid contamination from persistent cache, allocator state, and extension initialization.

`plot.py` reads only legacy `result.pkl` and remains for historical output. Analyze new JSON through the specific experiment, for example `exps/quant_linear_peft_bench/analyze.py`.
