# Memory-Twin Capture and Simulation

This directory captures operations, CUDA requested bytes, and digital-twin state from a real `Brain`, then uses `simulate.py` to replay with only `MemoryBrain`. See [`docs/memory-twin-capture-and-replay.md`](../../../docs/memory-twin-capture-and-replay.md) for protocol and responsibility boundaries.

## Entry points

| File | Purpose |
|---|---|
| `capture_infer.py` | capture inference, optionally with decode CUDA graphs |
| `capture_train.py` | capture sequential or concurrent-minibatch training |
| `simulate.py` | replay a capture and write per-operation comparisons |

`capture_train_parallel.py` has been consolidated into `capture_train.py`. Use `--max-in-flight` to control concurrently submitted minibatches in one training instance.

Capture files are pickles containing internal project objects. Load only trusted captures and replay with a code revision close to the capture revision.

## Inference capture

Minimal graph-off example:

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" \
  scripts/offline/memory_modeling/capture_infer.py \
  --output /tmp/memory_infer.pkl \
  --method ptuning --max-tokens 2 --no-cuda-graph
```

Graph-on example:

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" \
  scripts/offline/memory_modeling/capture_infer.py \
  --output /tmp/memory_infer_graph.pkl \
  --method ptuning --max-tokens 2 \
  --cuda-graph --graph-budget-mib 16 \
  --max-concurrency 1 --max-seq-len 256
```

Inference-specific arguments:

| Argument | Default | Meaning |
|---|---:|---|
| `--method {ptuning,lora}` | `ptuning` | PEFT type in inference Bindings |
| `--max-tokens N` | `32` | generated token count |
| `--pipeline-decoder-groups N` | `1` | decoder route groups in one inference instance |

## Training capture

One sequential minibatch:

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" \
  scripts/offline/memory_modeling/capture_train.py \
  --output /tmp/memory_train.pkl \
  --method lora --steps 1 --max-in-flight 1 --no-preallocate
```

Two concurrent minibatches with training-memory preallocation during Binding creation:

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" \
  scripts/offline/memory_modeling/capture_train.py \
  --output /tmp/memory_train_parallel.pkl \
  --method ptuning --steps 2 \
  --batch-size 2 --max-in-flight 2 --preallocate
```

Training-specific arguments:

| Argument | Default | Meaning |
|---|---:|---|
| `--method {ptuning,lora}` | `ptuning` | training PEFT type |
| `--steps N` | `4` | submitted minibatch count |
| `--minibatch-size N` | `1` | records in one minibatch |
| `--batch-size N` | `1` | minibatches in one optimizer update |
| `--max-in-flight N` | `1` | maximum concurrent submissions to one training instance |
| `--pipeline-decoder-groups N` | `2` | decoder route groups in one training instance |
| `--bfloat16` / `--no-bfloat16` | off | training-input dtype |
| `--individual-train-thread` | off | use a dedicated training execution thread |

`--max-seq-len` must cover the longest captured sequence. Artificially reducing training context to 256 or another small value can make RoPE position indexes overflow.

## Common capture arguments

| Argument | Default | Meaning |
|---|---:|---|
| `--model-name NAME` | `teared-Llama-2-7B-GPTQ` | foundation model |
| `--seed N` | `0` | PyTorch CPU/CUDA seed for model initialization and DataLoader |
| `--decoder-backend NAME` | `marlin` | decoder implementation |
| `--max-seq-len N` | model default | maximum context length |
| `--max-layers-to-load N` | all | load only the first N layers for reduced debugging |
| `--decoder-group-size N` | `8` | layers per base decoder group |
| `--max-concurrency N` | infer `1` / train `8` | Brain batch cap; graphs capture batches `1..N` |
| `--cuda-graph` / `--no-cuda-graph` | off | capture decode graphs during model loading |
| `--graph-budget-mib N` | `0` | hard graph budget; must be explicitly positive when graphs are enabled |
| `--preallocate` / `--no-preallocate` | off | pre-create PEFT optimizer/gradient state |
| `--preswap` / `--no-preswap` | off | preswap scalable KV cache |
| `--root-memory-gib GiB` | total GPU memory | override capacity used by Brain memory policy |

Graph count grows with `--max-concurrency`. For protocol checks, start with `--max-concurrency 1` and then expand. The budget is checked against measured requested bytes of all active graphs. Over-budget or capture OOM terminates; the process neither deletes graphs nor falls back to eager. Adjust the budget using the reported `required_minimum` and verify root-memory configuration.

## Replay

New captures record root memory and can replay directly:

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" \
  scripts/offline/memory_modeling/simulate.py \
  /tmp/memory_infer_graph.pkl \
  --output /tmp/memory_infer_graph_sim
```

| Argument | Meaning |
|---|---|
| `--output DIR` | write `DIR/result.json` |
| `--root-memory-gib GiB` | override captured root memory; required for old captures |
| `--preallocate` / `--no-preallocate` | replay one policy; default replays both |
| `--plot` / `--no-plot` | generate `compare.html`; enabled by default |
| `--embed-plotlyjs` | embed Plotly JS; default uses CDN to reduce file size |
| `--error-subplot-ratio RATIO` | vertical ratio of the prediction-error subplot in `[0, 1)`; default `0.28`, and `0` merges it into the main plot |

Without an explicit preallocation option, simulation replays both policies and plots both predictions and errors in one HTML document. Only the mode matching capture metadata establishes replay consistency; the other is counterfactual and normally has pool mismatches.

`result.json` contains requested/allocated bytes, twin predictions, errors, pools, graph-preparation/active-graph summaries, per-record `model_pools_match`, aggregate `model_pool_mismatch_count`, and speculative-revert/commit timings. CUDA comparison always uses allocator requested bytes, not allocated/reserved bytes.

`compare.html` contains CUDA requested/allocated, both preallocation-policy twin totals and errors, and stacked twin pools. It uses Times New Roman, distinct measured/predicted line styles, interactive operation hover, hidden x ticks, responsive sizing, and a separate error subplot by default. Pass `--embed-plotlyjs` for a self-contained offline file.

## Graph trace

Every outer operation carries:

- `graph_capture_preparations`: replayable logical effects of decoder graph capture on the KV allocator;
- `graph_runtime_snapshot`: live graph keys and measured requested-byte deltas after capture.

The simulator replays preparation and reconciles only graph entries in `MemoryBrain.runtime` with the measured snapshot. Ordinary tensor, KV, adapter, optimizer, and checkpoint pools remain digital-twin predictions.

Old graph-off six-tuple captures remain readable but require `--root-memory-gib`. Old graph-on six-tuples lack KV preparation and graph lifecycles and are rejected; recapture them.

Known persistent unmodeled Marlin memory after complete model unload is the cuBLAS workspace: 8.125 MiB for inference and 16.25 MiB for PEFT. It is outside the current twin's modeling scope.

`simulate.py` continues using the legacy visual style in `visualize_utils.py`.
