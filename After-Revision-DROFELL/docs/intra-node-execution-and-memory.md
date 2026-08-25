# Intra-Node Execution and GPU-Memory Governance

## 1. Four-layer structure

```text
Compute Node
  -> Binding Shim / Provider
  -> Brain queues and scheduler
  -> backend / operators
```

| Layer | Responsible for | Not responsible for |
|---|---|---|
| Compute Node | step/version admission, `Navigate`, and local-segment advancement | direct CUDA-memory management |
| Binding Shim | translating logical tasks into `Operate*` objects | network-path selection |
| Brain | queueing, selection, merging, GPU-memory feasibility, and arbitration | defining the model's mathematical semantics |
| Backend/Operator | real tensor and kernel execution | job-level recovery |

## 2. Binding

A Binding can be written as `B = (*shards_group, S)`: one contiguous decoder shard group plus task state.

Configuration and runtime objects are separate:

- `InferBindingInit` / `TrainBindingInit` describe loading and initial state;
- `*BindingShim` exposes inference or training interfaces upward;
- `Provider` loads and caches foundation and user models;
- the Brain organizes mergeable operations by Binding key.

The compute node creates and releases Bindings. Paths that depend on Python object destruction should still use explicit shutdown and runtime metrics to verify actual release timing.

## 3. `shards_group` granularity

A shard group participates in:

- resource locations and path partitioning;
- model loading and Binding creation;
- failure recovery and splitting;
- merging adjacent intra-node operations;
- checkpoint and state migration.

Too fine a granularity increases handle, routing, and migration metadata; too coarse a granularity reduces placement and recovery flexibility. It is a deployment parameter, not one universally optimal value for every model and device.

## 4. Inference operation path

An inference Binding decomposes upper-level calls into head, decoder, and tail operations:

- prefill creates the initial KV state;
- generating/decode reads and extends KV state;
- the tail computes logits, sampling, and result packaging;
- KV checkpoint, swap, and recovery injection use separate operations.

When structures match exactly, head/decoder/tail can form a fused graph. Heterogeneous Bindings, missing variants, LoRA, non-greedy sampling, or an excessive context length use staged fallback.

## 5. Training operation path

A training Binding provides:

- forward;
- backward;
- gradient accumulation;
- optimizer/scheduler updates;
- training-state dump/load.

The reorder controller enforces version, minibatch, and direction relationships. Out-of-order safety has two layers:

1. ordering controllers prevent illegal phase and version execution;
2. leaf gradients within one version accumulate by addition in `dealers/real_basic/train_binding.py::do_weight_update`.

The weight update and version advancement share one execution path. Gradient commutativity cannot be attributed to the reorder buffer alone.

## 6. Brain queues and selection

The Brain maintains multiple semantic queues rather than one FIFO:

- inference head, decoder, and tail;
- prefill and generating;
- training forward, backward, and update;
- swap, checkpoint, and model-management operations.

`_operate()` centrally selects the next class of work. The backend is the sole arbiter of the device driver and memory state, preventing multiple coroutines from bypassing the model and mutating device objects independently.

## 7. Batch merging

Every operation type may declare merge semantics, but only some decoder/prefill/generating paths currently take multiple items from a queue at once. Merging requires:

- identical or compatible Bindings;
- adjacent operations in the ordered container;
- compatible tokens, shapes, and graph variants;
- feasibility of the complete candidate batch in the GPU-memory model.

The default `max_concurrency` is 8 and caps one selection. It is not a correctness constant; prefill is additionally constrained by token count and GPU memory.

## 8. Memory Digital Twin

### Object model

`dealers/real_basic/memory.py::MemoryBrain` mirrors the structure of real model objects. `operators/core/memory/env.py::Env` manages objects, pools, and change logs.

A candidate transaction:

1. creates a snapshot;
2. applies candidate lifecycle changes to modeled objects;
3. evaluates the restriction tree;
4. reverts if infeasible;
5. binds modeled results to a real operation and commits if feasible.

### Constraint tree

The tree includes pools for the foundation model, operation reservations, cross-operation in-flight objects, training contexts, KV cache, user models, gradients, and optimizers. Rules express reservations, upper bounds, parent-child sharing, and dynamic availability.

A candidate is infeasible when any required leaf has `available < 0`. This is a constraint-feasibility trigger, not a global occupancy threshold. `major_percentage`, reservation, and maximum parameters change pool budgets but do not replace object-level simulation.

### Fidelity boundary

The digital twin models Brain operations. Accuracy depends on structural symmetry between real and modeled objects. The following require additional calibration or reservation:

- allocator reservation and fragmentation;
- temporary kernel workspaces;
- external libraries and unmanaged allocations;
- aliases, shared storage, and graph-capture lifecycles.

`unmanaged` accounts for modeled in-flight carriers not assigned to a named application pool. It is not a universal fallback for arbitrary unknown CUDA allocations.

## 9. SwapperCore

Different operation types use different strategies when GPU memory is insufficient.

### Training context

The scheduler first downgrades the retention strategy incrementally and swaps training state only if the candidate remains infeasible. Checkpoint branches include combinations of full-gradient retention, per-layer state, two-pass replay, and more aggressive checkpointing. Configuration and feasibility jointly determine the selected branch.

### KV block

KV blocks are ranked for eviction with a weighted score containing generation/position encoding. This score and the training-context downgrade path are different policies and should not be combined into one sorting formula.

### State outside the graph

Some training state must be detached from the autograd graph so that swap or checkpoint does not leave it referenced by the graph. Related helpers are under `operators/core/swappable/`.

## 10. Dual-class asynchronous scheduling

Runtime dual-class scheduling lives in `extend_asyncio`:

- ordinary coroutines become ready work;
- blocking container/CUDA closures enter `BlockingScheduler`;
- the event loop drains ready work before advancing one blocking item.

The Brain's semantic operation queues, the CUDA completion poller, and selector backends are adjacent but distinct structures. Configure and measure them separately.

## 11. Debugging checklist

For ordering or memory problems, inspect in sequence:

1. the compute node's step/version/running state;
2. waiting and completed sets in the reorder controller;
3. compatibility of Binding keys in the selected batch;
4. lifecycle symmetry between modeled and real objects;
5. the first restriction-tree leaf with negative `available`;
6. whether downgrade/swap paths revert cleanly;
7. whether any device mutation bypasses the Brain;
8. whether graph capture/replay adds storage.

## 12. Key source files

- `dealers/common/provider.py`
- `dealers/common/reorder.py`
- `dealers/common/brain.py`
- `dealers/memory_schedule/brain.py`
- `dealers/memory_schedule/restriction.py`
- `dealers/memory_schedule/swapper_core.py`
- `dealers/real_basic/basic_operate.py`
- `dealers/real_basic/memory.py`
- `dealers/real_basic/train_binding.py`
- `operators/core/memory/env.py`
- `operators/core/swappable/`
- `extend_asyncio/`
