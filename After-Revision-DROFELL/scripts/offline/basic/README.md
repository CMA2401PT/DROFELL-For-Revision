# Basic Offline Correctness and Recovery Checks

These scripts validate that inference and training still complete after introducing the `Brain`, providers, model segmentation, Bindings, the memory twin, and the scheduler.

## Entry points

| File | Test |
|---|---|
| `infer_basic.py` | sharded 7B single-machine inference with P-Tuning/LoRA and selectable Brains |
| `infer_basic_70B.py` | 70B multi-GPU pipeline-parallel inference |
| `train_basic.py` | 7B offline fine-tuning |
| `train_basic_70B.py` | 70B multi-GPU offline fine-tuning |
| `infer_and_train_basic.py` | inference and training in one testbed |
| `infer_correctness.py` | compare retrieval overhead and interrupt/recover at configured decode steps; records token completion times |
| `train_correctness.py` | training interruption/recovery and checkpoint-state correctness |
| `throughoutput_basic.py` | lightweight provider/model data-path check |

## Running

`infer_correctness.py` uses `PipelineParallelTestbed` and shares the benchmark's `--block-retrieve` and `--recover STEP[:split]` argument design. For example, validate recovery with KV:

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" \
  scripts/offline/basic/infer_correctness.py \
  --max-tokens 64 --ignore-eos --block-retrieve --recover 40
```

With a complete KV checkpoint, default `--recovery-replay chunk` sends contiguous post-checkpoint input through the head and each decoder group as one decode chunk. `--recovery-replay sequential` preserves per-token decode for correctness/timing A/B. Without KV, both modes use merged prefill plus the final decode fallback. JSON field `recovery_gap_ms` records the adjacent-token completion gap at recovery.

As of 2026-08-12, some older 7B entry points still import the removed `TestBed` compatibility alias. The following `infer_basic.py` command therefore fails during import until the entry point is migrated to `PipelineParallelTestbed([brain], cfg)`:

```bash
PYTHONPATH="$PWD:$PWD/scripts/offline" \
  "${DROFELL_PY:-python}" scripts/offline/basic/infer_basic.py \
  --adapter base --decoder-backend marlin --parallel 8 --max-tokens 32
```

`infer_basic.py` accepts `--adapter base|lora|ptuning` for complete 7B inference. `train_correctness.py` accepts `--method lora|ptuning` and a bounded `--steps` for PEFT. Both default to Marlin and retain `--decoder-backend exllama|exllama_fuse` for regression.

Decode CUDA graphs are Marlin-only and cached by decoder Binding, batch, and cache engine:

```bash
PYTHONPATH="$PWD:$PWD/scripts/offline" \
  "${DROFELL_PY:-python}" scripts/offline/basic/infer_basic.py \
  --adapter base --parallel 8 --max-tokens 32 \
  --cuda-graph --graph-budget-mib 64
```

Base and P-Tuning can replay graphs; LoRA explicitly falls back to eager. Each Brain supports batch buckets `1..max_concurrency`; requests exceeding `--graph-max-context-len` or the Brain batch limit fall back. Graphs cover decoder generating forward only; head embedding and tail norm, LM head, and sampling remain independent eager stages.

Before running other legacy entry points, inspect in source:

- 7B/70B model, loaded layer count, decoder grouping, and GPU count;
- GPU-memory limits in `MemoryConfigStatic` / `MemoryConfigPercentage`;
- P-Tuning/LoRA configuration and state files;
- breaker/recovery phases, batch, token length, and iteration count.

The 70B entry points are multi-GPU and expensive by default and should not be smoke tests. Begin with `throughoutput_basic.py` or a reduced `infer_basic.py`.

## Output and acceptance

- Ordinary entry points print text, loss, speed, and resource information without one output schema.
- `infer_correctness.py` writes schema-version-2 `infer_with_recover_record.json` with three token-ID lists, text, completion timestamps, and per-node recovery records. It asserts identical token IDs, every requested recovery triggered, and no prefill during replay after KV injection. Records distinguish `decode` from `decode_chunk` and include chunk step ranges/counts. Top-level `fast`, `common`, and `recover` timestamp keys remain. See [Offline Design and Inference Execution Path](../../../docs/offline-design-and-inference-path.md#12-shadow-state-and-recovery).
- `train_correctness.py` writes `train_with_recover_correct_record.json`.
- Recovery checks should compare token/loss, state version, checkpoints, and final testbed cleanup before and after interruption. Exit status zero alone does not establish recovery semantics.
