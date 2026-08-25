# QuantLinear PEFT Backend Benchmark

## Purpose

On one GPTQ projection, compare ExLlama and Marlin inference, PEFT forward/backward, weight reconstruction, and GPU-memory lifecycles. The module-level data supports backend-dispatch and complete-decoder debugging.

## Paths

- `our_no_cache`: reconstruct weights during backward;
- `our_cache`: retain the dense weight from forward for backward;
- `marlin`: packed INT4 direct forward;
- `marlin_dense_no_cache`: reconstruct independently in forward and backward;
- `marlin_dense_cache`: reconstruct in forward and retain through backward.

Shared workspaces belong to the model-loading baseline. Dense weights retained after forward belong to the context lifecycle. Account for them separately.

## Running

Build the extension first:

```bash
(cd cuda_extends && python setup.py build_ext --inplace)
```

Run one projection:

```bash
export DROFELL_PY="${DROFELL_PY:-python}"

PYTHONPATH="$PWD" "$DROFELL_PY" exps/quant_linear_peft_bench/run.py \
  --run-id qlinear-qproj \
  --python "$DROFELL_PY" \
  --model-dir split_qllm_models/teared-Llama-2-70B-GPTQ \
  --projection q_proj \
  --methods our_no_cache our_cache marlin marlin_dense_no_cache marlin_dense_cache \
  --seq-lens 1 8 16 32 64 96 128 192 256 384 512 768 1024 1536 2048 \
  --warmup 5 --repeats 20 --seed 0
```

Use independent run IDs for `gate_proj` and `down_proj`. Run every backend in an independent process.

## Output and acceptance

Input weights, manifests, raw timings, memory records, summaries, and figures are local generated files and are not version-controlled.

Check that:

- weight manifests refer to the same model, layer, and projection;
- every method uses identical input and seed;
- warmup is outside timing;
- CUDA Event synchronization is consistent;
- forward/backward input gradients or checksums agree within tolerance;
- a dispatch crossover applies only to the measured GPU, shape, and quantization layout;
- a module-level crossover does not replace end-to-end decoder selection.
