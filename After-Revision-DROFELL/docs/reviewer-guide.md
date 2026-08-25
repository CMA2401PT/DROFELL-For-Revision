# DROFELL Artifact Guide for Reviewers

This page is the English entry point for inspecting the implementation that accompanies the
revision. It is a navigation aid, not an additional source of performance or correctness claims.
Quantitative claims and their experimental conditions remain in the revised manuscript and the
response letter; this guide points to the code that carries the corresponding mechanisms.

## 1. Architectural objective and present scope

DROFELL targets a state-centric, reconfigurable, layered multi-path computation graph. The graph
retains the logical order of model shards rather than becoming an arbitrary network mesh, while
allowing more than one physical realization of a logical transition.

The broader objective contains three capabilities whose present status differs:

| Architectural capability | Present realization |
|---|---|
| performance-informed path use | candidate probing during graph construction and replacement selection; first-legal-arrival competition among already selected copies; no continuous globally fastest-path routing |
| continuity and reconstruction under partial failure | concurrent selected realizations, STM-constrained admission, Promise/shadow reconstruction, and reactive rebinding; this is the capability evaluated end to end |
| topology malleability | failure- or explicit-relinquishment-driven replacement, graph rewrite, and split recovery; no general policy-driven online path insertion/removal or repartitioning |

## 2. Semantics, mechanisms, and policy are separate

| Level | Question answered | Present components |
|---|---|---|
| semantic safety substrate | Which input is consumed, how does logical state advance, and what frontier must another holder realize? | STM, `canon`, Promise, Binding |
| implemented runtime mechanisms | How can selected physical realizations coexist or be replaced safely? | control-side shadow material, concurrent selected Bindings, departure-driven graph rewriting |
| adaptation policy | Which paths should be selected, and when should paths be added, removed, or migrated? | construction/recovery probing, configured redundancy, and reactive replacement; continuous optimization is outside the current system |

The semantic substrate determines whether a physical change is valid. It does not decide when that
change should occur or which new path is optimal.

## 3. Concept-to-code map

References use files and symbols rather than line numbers so that they remain stable as the source
tree evolves.

| Concept | Primary implementation carriers | What to inspect |
|---|---|---|
| STM and the admission-induced `canon` map | `dealers/common/reorder.py::InferTaskReorderAndDropCtrl`, `dealers/common/reorder.py::TrainTaskReorderAndDropCtrl`, and the Binding shims in `dealers/common/provider.py` | rejection of stale/duplicate work, parking and ordered redemption of future work, and inference/training phase constraints |
| Promise | a semantic frontier contract; current redemption through `node/shadow_node_state/infer_state_data.py::InferStateData.gen_init_cfg` and `node/shadow_node_state/train_state_data.py::TrainStateData.gen_init_cfg` | the configuration, optional snapshot, and boundary-input trace from which another holder reaches the contract-specified frontier |
| Binding | `node/compute_node/infer_meta.py::InferBindingInit`, `node/compute_node/train_meta.py::TrainBindingInit`, and the Binding shims in `dealers/common/provider.py` | the rebindable boundary between one logical shard role/task state and its current physical realization |
| shadow state | `node/shadow_node_state/infer_state_data.py::InferStateData` and `node/shadow_node_state/train_state_data.py::TrainStateData` | control-side material used to redeem a Promise; this is data, not an idle GPU worker |
| concurrent selected realizations | inference/training controls in `node/infer/infer_ctrl.py` and `node/train/train_ctrl.py`, together with Agent controllers in `node/agent/` | dispatch to selected handles and admission of the first legal result while later duplicates are absorbed |
| reactive graph rewriting | `node/agent/infer.py::AgentInferController`, `node/agent/train.py::AgentTrainController`, and the recovery domains in `node/backbone/` | replacement selection, handle installation/removal, local recovery ownership, and split recovery |
| self-healing node | `node/backbone/infer_backbone.py` and `node/backbone/train_backbone.py` | a local redundant subgraph exposed outward as one logical capability |
| memory digital twin | `dealers/real_basic/memory.py::MemoryBrain`, `operators/core/memory/env.py::Env`, and `dealers/memory_schedule/` | rollback-capable prospective memory changes, constraint-feasibility admission, downgrade, and swap |
| dual-class asynchronous runtime | `extend_asyncio/common.py::BlockingScheduler` and the event loops in `extend_asyncio/std_asyncio.py`, `extend_asyncio/lite_asyncio.py`, and `extend_asyncio/chimera_asyncio.py` | ready non-blocking work is advanced before one potentially blocking device launch is attempted |
| transport boundary | `orbit/defines.py::Context` and `node/base/__init__.py::BaseNode` | the reachability, direction, bandwidth/latency, and failure-observation properties exposed to the algorithms |

No runtime class is literally named `STM`, `Promise`, or `canon`; these are semantic contracts
realized jointly by the carriers above.

## 4. Suggested inspection routes

### Inter-node state and recovery

1. `dealers/common/reorder.py`
2. `node/shadow_node_state/`
3. `node/infer/infer_ctrl.py` or `node/train/train_ctrl.py`
4. `node/agent/infer.py` or `node/agent/train.py`
5. `node/backbone/`

This route connects legal state advancement, retained frontier material, and departure-driven
replacement. A surviving redundant compute peer is not queried for the replacement state.

### Binding and intra-node execution

1. `node/compute_node/infer_meta.py` and `node/compute_node/train_meta.py`
2. `dealers/common/provider.py`
3. `dealers/common/brain.py`
4. `dealers/real_basic/basic_operate.py`

This route shows how task-scoped state is attached to a shard role and translated into mergeable
device operations.

### Memory feasibility and asynchronous execution

1. `dealers/real_basic/memory.py`
2. `operators/core/memory/`
3. `dealers/memory_schedule/`
4. `extend_asyncio/common.py`
5. one of the event-loop implementations under `extend_asyncio/`

This route separates prospective memory feasibility from the runtime confinement of potentially
blocking CUDA work.

### Agent/Request placement

For the English design note covering colocated and remote Agent/Request deployment, cancellation,
and timing boundaries, see [`online-agent-request-design.md`](online-agent-request-design.md).

## 5. Boundaries to keep in mind

- A first legal arrival among selected copies is passive winner selection, not continuous adaptive
  routing.
- Graph rewriting is triggered by detectable failure or explicit relinquishment; the current system
  does not perform general policy-driven topology optimization.
- End-to-end service continuity requires a surviving selected path. Without one, reconstruction can
  produce a visible output gap.
- The default backend does not provide general NAT traversal, congestion control, split-brain
  fencing, or recovery of the outermost per-task Agent.
- Memory safety is conditional on modeled allocations and configured reservations for bounded
  external allocation classes.
- Numerical recovery is defined modulo the stated numerical-equivalence relation, not bitwise
  identity across nondeterministic kernels or different accumulation orders.

