# Memory Digital Twin Capture and Replay

## 1. Goals and boundaries

`scripts/offline/memory_modeling` validates the `MemoryBrain` GPU-memory digital twin. The simulation stage must neither reload a CUDA model nor perform inference/fine-tuning. The workflow has two stages:

1. **Capture:** intercept operations around a real `Brain` and record inputs, return values, CUDA allocator changes, and snapshots of the real `Brain.memory_modeling` pools.
2. **Replay:** construct only `MemoryBrain`, perform speculative apply/revert followed by commit/merge/bind for the same operation, and compare predicted memory with captured CUDA requested bytes.

These are two distinct validation goals:

- **Replay consistency:** simulated pools should equal the pools in the real Brain's MemoryBrain during capture.
- **Modeling error:** only after replay is consistent should MemoryBrain totals be compared with CUDA requested bytes.

Until replay consistency holds, the difference must not be interpreted as digital-twin modeling error.

## 2. Current call path

```text
PipelineParallelTestbed
  -> Provider / InferNode / TrainNode
  -> Brain APIs or operation queues wrapped by AnalysisFrame
  -> ActualIORecorder
       records args/results
       reads CUDA allocator checkpoints
       records real Brain.memory_modeling pools
  -> pickle capture

pickle capture
  -> simulate.py
  -> pure MemoryBrain
  -> speculative apply + revert
  -> committed apply + merge + result bind
  -> result.json
```

`AnalysisFrame` wraps synchronous model make/remove/swap interfaces and every Brain operation mailbox. `OperateWithCallback` begins recording when an operation actually starts, then serializes its result and reads allocator state in `on_finish()`.

A version-2 capture is a metadata-bearing envelope:

```text
{
  schema_version: 2,
  metadata: {...},
  records: [{
    sequence, operation, args, result,
    cuda_memory_status, captured_model_pools,
    graph_capture_preparations,
    graph_runtime_snapshot
  }]
}
```

`captured_model_pools` is the replay-consistency oracle and must not be copied indiscriminately into the simulator. Doing so would hide modeling defects in ordinary tensors, KV, or optimizer state and defeat validation of the digital twin.

## 3. CUDA comparison metric and baseline

`simulate.py` uses `status.accumulate.requested`, not allocated or reserved allocator bytes. The digital twin models bytes requested by tensor/runtime objects, making requested bytes the appropriate primary metric.

Known persistent Marlin memory not modeled separately is the cuBLAS workspace:

- inference only: 8.125 MiB;
- PEFT: two workspaces totaling 16.25 MiB.

With graphs disabled and only temporary tolerance for `adjacent_decoder` in `MemoryBrain.make_model_head/tail()`, all 40 inference and 34 training records replay successfully; final/maximum residuals are 8.125 MiB and 16.25 MiB respectively. This indicates that the core non-graph model is intact and that the failure came from capture/replay protocol drift.

## 4. Resolved protocol drift

### 4.1 The `adjacent_decoder` interface

After fused decode graphs were introduced, real Brain `make_model_head()` / `make_model_tail()` calls received a new `adjacent_decoder` argument. `AnalysisFrame` stored the keyword unchanged, while `MemoryBrain` still accepted only `base_model`, causing simulation to fail on the first `make_model_head` record:

```text
TypeError: MemoryBrain.make_model_head() got an unexpected keyword argument
'adjacent_decoder'
```

MemoryBrain now accepts the argument and maintains protocol symmetry. The argument itself creates no head/tail tensor memory; graph side effects are replayed separately as described in Section 5.

### 4.2 Training capture entry point

Training capture now uses `PipelineParallelTestbed([brain], cfg)`. The old parallel script was essentially concurrent minibatches within one training instance and has been consolidated into `capture_train.py --max-in-flight N`.

### 4.3 Capture metadata

The version-2 envelope records the model, context, decoder backend, group size, maximum concurrency, graph batch sizes, memory policy, GPU, and PyTorch/CUDA versions. By default, simulation uses the context, backend, root memory, and preallocation policy from the capture.

### 4.4 Nested graph-capture side effects

Real `BasicBrainOperates._capture_decoder_graph_binding()` performs additional operations inside an intercepted `make_model_*`:

```text
MemoryBrain.prepare_decode_graph_capture(binding, temporary_block_count)
  -> temporarily allocate/provide/release KV blocks
  -> possibly create a persistent device buffer for scalable cache

MemoryBrain.track_runtime_memory(graph_key, observed_requested_bytes)
MemoryBrain.remove_runtime_memory(graph_key)  # during reconcile or unload
```

These calls mutate the real Brain's digital twin. `AnalysisFrame` still treats `make_model_*` as the outer operation, but associates internal KV preparation with that record; `ActualIORecorder` stores the active-graph snapshot after the operation.

Block allocation/release during graph preparation affects later block-ID evolution, so recording only graph bytes is insufficient. The simulator now replays preparation in both speculative and committed apply paths, preventing block-ID drift when an inference Binding is unloaded.

## 5. How graphs enter the trace

### 5.1 Design

Persistent requested bytes for a CUDA graph should enter the trace as **runtime calibration observed in the capture environment**, not be derived statically by MemoryBrain from tensor shapes alone.

Graph retention includes not only directly calculable input/output buffers, but also intermediate tensors in the capture pool, kernel/library workspaces, and PyTorch/CUDA implementation details. It varies with batch, context capacity, decoder backend, variant structure, GPU, and software stack. A static model fitted on one machine would not transfer robustly.

Immediately before and after each graph capture, the graph runner can synchronize the CUDA stream and read `requested_bytes.all.current`; the snapshot difference is the observed retained amount for that graph key. The current graph runner already uses this method, and capture carries the result into the offline trace.

Graph information must nevertheless be split into two classes:

- **Replayable logical side effect:** `prepare_decode_graph_capture(binding, block_count)` must be recorded and replayed because it changes the KV pool and block IDs.
- **Observed graph retention:** each outer operation records the live `{graph_runtime_key: observed_requested_bytes}` snapshot. Simulation reconciles only the runtime-graph pool against this snapshot.

`track_runtime_memory()` / `remove_runtime_memory()` are internal methods used by the real Brain to maintain the table and need not become user-visible trace concepts. A snapshot after every outer operation expresses graph creation, variant replacement, and deletion with a simpler protocol.

Ordinary tensor, KV, adapter, optimizer, and checkpoint pools remain predictions of MemoryBrain and are not filled from `captured_model_pools`.

The real scheduler places observed runtime memory in a fixed graph-reserved region of the constraint tree:

```text
reserved_total = operate_reserved + graph_budget
0 <= sum(active graph requested bytes) <= graph_budget
```

The regions do not borrow from one another. Total live graph retention is checked after every capture. Exceeding the budget or encountering CUDA OOM during capture is a fatal configuration error that terminates the process and requires an adjusted memory configuration. Graphs are not an evictable cache, so over-budget recovery must not delete a graph, silently fall back to eager, or consume operation reservation. Normal model unload and graph-variant replacement still delete obsolete graphs according to their lifecycle.

### 5.2 Implemented trace structure

The capture uses a versioned envelope:

```text
{
  schema_version,
  metadata: {
    model_name, max_seq_len, decoder_backend,
    max_concurrency, enable_cuda_graph, graph_batch_sizes,
    memory_config, device, torch_version, cuda_version
  },
  records: [
    {
      sequence, operation, args, result,
      cuda_memory_status, captured_model_pools,
      graph_capture_preparations: [
        (binding, temporary_block_count)
      ],
      graph_runtime_snapshot: {
        runtime_key: observed_requested_bytes
      }
    }
  ]
}
```

`graph_capture_preparations` contains logical operations that execute in both speculative and committed simulator applies. `graph_runtime_snapshot` contains observed calibration; during each apply the simulator modifies only graph entries under `MemoryBrain.runtime`, so both forms of state participate in `Env.revert_all()` / `merge_all()`.

A graph runtime key can be treated as an opaque stable identifier within one trace. Its cache-engine Python identity has no cross-process semantics; simulation must neither recompute nor interpret that value, only preserve references to the same identifier in later remove events.

### 5.3 Meaning of “real graph size”

The graph runner records the live difference in `requested_bytes.all.current` before and after capture. This is best described as the **observed retained requested-byte delta**, not a portable net graph size.

For example, the first tail-graph warmup lazily initializes a cuBLAS workspace. During investigation, batch-1 graph capture raised the runtime pool peak to about 8.282 MiB, most of which was the 8.125 MiB cuBLAS workspace. After unloading the graph, the workspace remained live and the residual returned to the expected 8.125 MiB unmodeled persistent amount.

The current protocol faithfully replays existing `Brain.memory_modeling` behavior. The cuBLAS workspace remains explicitly outside MemoryBrain's modeling boundary; simulation neither subtracts an empirical constant nor attempts to infer and separate it from the observed graph delta.

## 6. Backward compatibility and validation criteria

Compatibility rules:

1. Old six-tuple captures remain loadable.
2. If `captured_model_pools` in an old capture never contains non-zero `runtime`, replay it as a graph-disabled trace.
3. If an old capture contains non-zero `runtime` without graph preparation and runtime snapshots, reject it explicitly and require recapture. An aggregate runtime pool cannot reconstruct variant deletion or graph-KV preparation effects on block IDs.

Continuous validation criteria:

- graph off: final residual of 8.125 MiB for inference and 16.25 MiB for PEFT;
- graph on: simulated pools equal captured model pools for every record;
- graph-variant reconciliation: after removing `H+D` and adding `H+D+T`, runtime keys and bytes match capture;
- graph-enabled captures replay through all Binding/model unloads without block-ID drift;
- `revert_all()` after speculative apply restores every digital-twin state affected by graph runtime and KV graph preparation;
- simulation constructs the model from capture metadata by default; explicit CLI overrides are written to output rather than silently using unrelated defaults.

## 7. Current validation coverage

Under RTX 3090, PyTorch 2.11.0+cu130, Marlin, and four eight-layer decoder groups:

- graph-off inference: 40 operations, per-operation pool agreement, final residual 8.125 MiB;
- graph-on inference (batch graph 1, context 256): 40 operations, per-operation pool agreement, final residual 8.125 MiB;
- graph-on inference with one 32-layer decoder group: loading the tail replaces an `H+D` variant with `H+D+T`; all 16 operations agree and final residual is 8.125 MiB;
- graph-off P-Tuning: 34 operations, per-operation pool agreement, final residual 16.25 MiB;
- graph-on preallocated P-Tuning with two concurrent minibatches: 44 operations, per-operation pool agreement, final residual 16.25 MiB. Intermediate CUDA error can be substantially larger under preallocation; that is a current modeling-fidelity boundary, not a replay-pool inconsistency.

The cuBLAS workspace remains a known unmodeled persistent allocation and is not part of this protocol correction.
