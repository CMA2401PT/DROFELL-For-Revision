# DROFELL Documentation

This directory is for developers who use, debug, or extend DROFELL. The documents describe the current implementation, operating procedures, and capability boundaries. Reproduce quantitative performance and correctness measurements with the corresponding experiments.

Code references use `file::symbol` rather than line numbers so that they remain stable as the source tree evolves.

## Document index

| Document | Contents |
|---|---|
| [reviewer-guide.md](reviewer-guide.md) | Dedicated reviewer entry point: architectural scope, stable concept-to-code references, and suggested inspection routes |
| [walkthrough/README.md](walkthrough/README.md) | Derives state semantics, resilience mechanisms, intra-node execution, and the asynchronous runtime from the limitations of a fixed chain |
| [architecture-overview.md](architecture-overview.md) | Overall layering, node roles, and the main inference, training, and recovery flows |
| [concept-to-code-map.md](concept-to-code-map.md) | Mapping from state semantics, Bindings, Shadows, the memory twin, and other concepts to source objects |
| [implementation-boundaries.md](implementation-boundaries.md) | Current capabilities, runtime verification requirements, and unimplemented boundaries |
| [inter-node-control-and-recovery.md](inter-node-control-and-recovery.md) | Provider discovery, paths, controls and handles, resource locations, and recovery |
| [network-abstraction-and-orbit-backend.md](network-abstraction-and-orbit-backend.md) | The Context/BaseNode contract and the default Orbit backend |
| [intra-node-execution-and-memory.md](intra-node-execution-and-memory.md) | Bindings, the Brain, batch merging, memory modeling, downgrade, and swap |
| [offline-design-and-inference-path.md](offline-design-and-inference-path.md) | Offline object ownership, prefill/decode, graphs, and recovery paths |
| [online-agent-request-design.md](online-agent-request-design.md) | Online Agent/Request protocol, placement, and timing boundaries |
| [decode-composite-eager-design.md](decode-composite-eager-design.md) | Composite head/decoder/tail decode execution and CUDA graphs |
| [memory-twin-capture-and-replay.md](memory-twin-capture-and-replay.md) | Memory-model capture, replay, and validation |
| [runtime-platform-and-experiments.md](runtime-platform-and-experiments.md) | Build instructions, runtime platforms, script levels, and experimental entry points |

## Suggested reading routes

### Build a system-level understanding

If you are new to DROFELL, read [`walkthrough/`](walkthrough/README.md) in order. It explains why each mechanism is needed before moving into the implementation references:

1. `walkthrough/01-fixed-chain-to-reconfigurable-layered-multipath-graph.md`
2. `walkthrough/02-why-path-changes-require-state-semantics.md`
3. `walkthrough/03-resilience-mechanisms-and-layered-computation-graph.md`
4. `walkthrough/04-from-one-task-to-many.md`
5. `walkthrough/05-why-cuda-needs-a-different-async-runtime.md`
6. `architecture-overview.md`
7. `concept-to-code-map.md`
8. `implementation-boundaries.md`

### Follow one inference request

```text
Agent InferControl
  -> MultiPathInferTask
  -> InferNodeHandle
  -> DealerNode
  -> inference compute node
  -> Binding shim
  -> Brain / MemoryBrain
  -> operator
```

Key source files:

- `node/infer/infer_ctrl.py`
- `node/dealer_node/infer_handle.py`
- `node/dealer_node/dealer_node.py`
- `node/compute_node/infer_node.py`
- `dealers/common/provider.py`
- `dealers/memory_schedule/brain.py`

### Follow one training request

```text
Agent TrainControl
  -> MultiPathTrainTask
  -> TrainNodeHandle
  -> DealerNode
  -> training compute node
  -> Binding shim
  -> forward / backward / weight update
  -> checkpoint
```

Key source files:

- `node/train/train_ctrl.py`
- `node/dealer_node/train_handle.py`
- `node/compute_node/train_node.py`
- `dealers/common/reorder.py`
- `dealers/real_basic/train_binding.py`
- `operators/assemble/fwd_ctx/`

### Follow recovery

Read `inter-node-control-and-recovery.md`, then inspect:

- `node/shadow_node_state/infer_state_data.py`
- `node/shadow_node_state/train_state_data.py`
- `node/agent/infer.py`
- `node/agent/train.py`
- `node/backbone/infer_backbone.py`
- `node/backbone/train_backbone.py`

### Follow a GPU-memory decision

Read `intra-node-execution-and-memory.md`, then inspect:

- `dealers/real_basic/memory.py`
- `operators/core/memory/env.py`
- `dealers/memory_schedule/restriction.py`
- `dealers/memory_schedule/swapper_core.py`
- `dealers/real_basic/basic_operate.py`

## Reading conventions

- **Implementation fact:** directly supported by a current class, field, branch, or call relationship.
- **Runtime property:** requires a test or experiment in a fixed environment.
- **Design boundary:** explicitly absent from the default implementation or available only through a replaceable backend.

Common pitfalls:

- A recovery call path does not imply that end-to-end output has no gap.
- Shadow state does not imply an idle standby GPU.
- Duplicate suppression does not imply consensus under a network partition.
- A memory model does not imply exact modeling of every third-party allocator behavior.
- A training recovery path does not imply bitwise reproducibility for every low-precision backend.

## Terminology notes

- `dealer_node/` handles process-level network I/O and routing; `compute_node/` advances one task inside a node.
- A Shadow is control-side state attached to a handle, not a separately running node process.
- Resource-location maps are normally merged by union. Stale entries can cause an invalid send; disconnect handling, barriers, and path reconstruction repair the graph.
- GPU-memory policy is triggered by constraint feasibility, not by one global occupancy percentage.
- Training reorder safety depends on both phase ordering and additive leaf-gradient accumulation.
