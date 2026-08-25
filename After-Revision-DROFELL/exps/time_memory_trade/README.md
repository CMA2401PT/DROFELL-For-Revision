# Decoder Checkpoint/Swap Time-Memory Trade-Off

## Purpose

Under one decoder workload, compare forward, swap, backward, and incremental GPU-memory lifecycles for four checkpoint strategies and several backends.

## Strategies

| Name | Argument | Meaning |
|---|---|---|
| `each` | `each_decoder` | retain each layer input, recompute and backpropagate layer by layer |
| `2pass` | `first.2pass` | retain the first-layer input, replay once without gradients, then backpropagate layer by layer |
| `ckpt` | `first.grad` | retain the first-layer input and reconstruct the full autograd graph |
| `grad` | `grad` | retain the complete foundation-model gradient graph during forward |

Run every backend/strategy in an independent process so allocator history, workspaces, and extension state do not cross cases.

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export RUN_ID="${RUN_ID:-time-memory-trade}"

PYTHONPATH="$PWD" "$DROFELL_PY" exps/time_memory_trade/run.py \
  --run-id "$RUN_ID" \
  --python "$DROFELL_PY" \
  --backends exllama_fuse marlin \
  --strategies each_decoder first.2pass first.grad grad \
  --warmup 5 --repeats 100 \
  --no-reuse-reconstruct-cache
```

After interruption, use identical arguments plus `--resume`. The runner should skip only a case with both a complete pickle and JSON sidecar.

Reanalyze with:

```bash
PYTHONPATH="$PWD" "$DROFELL_PY" exps/time_memory_trade/analyze.py \
  --run-dir "exps/time_memory_trade/results/$RUN_ID"
```

Use `--no-plots` in a headless environment.

## Measurement semantics

- Timing covers complete stages and synchronizes CUDA at boundaries.
- Objects are released explicitly and GC/synchronization runs between warmup, timing, and memory capture.
- `Peak` is the peak during forward.
- `Res.` is context retained after forward for backward.
- `Bwd` is the complete peak through backward.
- Occupancy after swap-out cannot replace forward residual.
- Reconstruction-cache reuse extends dense-weight lifetime and must be a separate case.

## Output and acceptance

Allocator snapshots, raw timing arrays, sidecars, logs, summaries, and figures are generated locally and are not version-controlled. Load pickles only from trusted sources.

Check sample counts, synchronization boundaries, the model-loading baseline, empty-tensor/shared-storage handling, and OOM, fallback, or exceptions in logs.
