# Memory-Twin Capture/Replay Experiment

## Purpose

Validate that traces from `scripts/offline/memory_modeling` replay MemoryBrain pools operation by operation and compare modeled memory with real CUDA allocator metrics. Coverage includes graph off/on, graph-variant replacement, PEFT training, preallocation, and concurrent minibatches.

## Matrix

| ID | Scenario |
|---|---|
| `infer_graph_off` | inference without graphs |
| `infer_graph_on` | graph inference |
| `infer_graph_triple` | replace head+decoder with head+decoder+tail variant |
| `train_graph_off` | P-Tuning without graphs |
| `train_graph_preallocate` | P-Tuning with graphs and preallocation |

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export CAPTURE_DIR="${CAPTURE_DIR:-$PWD/tmp/memory-twin/captures}"
export REPLAY_DIR="${REPLAY_DIR:-$PWD/tmp/memory-twin/results}"
mkdir -p "$CAPTURE_DIR" "$REPLAY_DIR"
```

Inference capture:

```bash
PYTHONPATH="$PWD" "$DROFELL_PY" \
  scripts/offline/memory_modeling/capture_infer.py \
  --output "$CAPTURE_DIR/infer_graph_on.pkl" \
  --seed 0 --method ptuning --max-tokens 100 --max-seq-len 256 \
  --max-concurrency 1 --decoder-group-size 8 \
  --pipeline-decoder-groups 1 \
  --cuda-graph --graph-budget-mib 16 --no-preallocate
```

Training capture:

```bash
PYTHONPATH="$PWD" "$DROFELL_PY" \
  scripts/offline/memory_modeling/capture_train.py \
  --output "$CAPTURE_DIR/train_graph_preallocate.pkl" \
  --seed 0 --method ptuning --steps 4 --batch-size 2 --max-in-flight 2 \
  --max-concurrency 1 --decoder-group-size 8 \
  --pipeline-decoder-groups 2 \
  --cuda-graph --graph-budget-mib 16 --preallocate
```

Replay:

```bash
PYTHONPATH="$PWD" "$DROFELL_PY" \
  scripts/offline/memory_modeling/simulate.py \
  "$CAPTURE_DIR/infer_graph_on.pkl" \
  --output "$REPLAY_DIR/infer_graph_on"
```

Generate other matrix cells by changing graph, decoder-group, and training arguments. Consult each script's `--help` for the full interface.

## Output and acceptance

Captures, JSON, and HTML are local generated files in Git-ignored directories. Load pickles only in trusted environments.

Check that:

- trace metadata matches runtime configuration;
- replay operation count and ordering are complete;
- same-policy pool mismatches are reported per operation;
- graph preparation, variant insertion/removal, and unload appear in the trace;
- modeled and CUDA metrics share one baseline;
- allocator reservation, workspaces, and preallocation differences are interpreted separately;
- counterfactual policy replay is not used to judge same-configuration consistency.
