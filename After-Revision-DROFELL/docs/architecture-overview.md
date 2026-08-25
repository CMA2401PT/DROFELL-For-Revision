# DROFELL Architecture Overview

## 1. Overview

DROFELL's broader objective is to evolve a fixed pipeline, where logical progress is bound to one physical path, into a **state-centric, reconfigurable multi-path computation graph**. The graph still follows the model's logical layer order rather than becoming an arbitrary network mesh, but more than one physical realization may implement the same logical state transition.

The complete architectural objective includes:

1. selecting better paths from runtime observations;
2. continuing along surviving paths after partial failure and reconstructing lost physical realizations;
3. dynamically adding, removing, migrating, or reoptimizing paths as resources change.

The current system fully implements and validates a bounded form of the second capability. For the first, it provides probing during graph construction and recovery plus passive competition among selected copies. For the third, it provides replacement, graph rewriting, and split recovery driven by detectable failure or explicit relinquishment.

### Three architectural levels

| Level | Responsibility | Current components |
|---|---|---|
| **Semantic safety substrate** | Define which input is consumed, how state advances, and what a new holder must reconstruct | STM, `canon(·)`, Promise, Binding |
| **Implemented runtime mechanisms** | Allow multiple physical realizations to coexist or be replaced safely | shadow redemption material, concurrent selected Bindings, departure-driven graph rewriting |
| **Adaptation policy** | Select paths and decide when to add, remove, or migrate them | graph-construction/recovery probing, configured redundancy, and departure-driven replacement; no continuous fastest-path routing or general topology optimization |

The semantic layer determines what remains the same legal task progress after a physical realization changes. The policy layer decides when a change should occur and where execution should move.

### Two DRM control axes

DROFELL's implementation is organized along two connected dynamic-resource-management axes:

- **Inter-node control:** organizes task graphs, selects paths, maintains resource locations, and handles disconnects and recovery.
- **Intra-node control:** binds tasks to model shards, orders operations, merges batches, and governs GPU memory.

These axes organize the implementation. They must not be confused with the two inter-node runtime mechanisms: concurrent multi-path execution and shadow-assisted rebinding.

```text
Request / Agent
  -> job Control
  -> execution Handle
  -> DealerNode
  -> Compute Node
  -> Binding Shim
  -> Brain / MemoryBrain
  -> Operator
```

## 2. Inter-node roles

### Orbit

The default deployment and communication backend. It handles service registration, node locations, connection setup, cross-process calls, and logging. Orbit's central service does not perform model computation, resource scheduling, or recovery.

### BaseNode

The shared communication base for node roles. It uses `Context` for location resolution, send/receive, and connection events. Upper-level logic does not depend directly on a particular socket implementation.

### AuxNode

Aggregates metadata about providers, throughput, and Backbones. Dealers query it during initialization or loading and decide autonomously which models to load. Aux does not retain a complete execution graph for every job or schedule requests step by step.

### AgentNode and RequestNode

An Agent owns a job-level control. A Request is a temporary client role that submits inference or training work and receives boundary outputs. Colocated roles can use local delivery; remote roles communicate through Orbit links.

### DealerNode

One Dealer process manages network receive, handle routing, and local compute nodes. `node/dealer_node/` and `node/compute_node/` are separate layers: the former manages process/network concerns, while the latter advances one task's execution state.

### Compute Node

Each logical execution instance has its own inference or training compute node. It uses `Navigate`, step/version state, and Bindings to determine locally executable segments, then returns work that leaves the node to the Dealer for forwarding.

### Backbone

A Backbone appears as one logical node externally and contains recursive controls, virtual nodes, and a local Aux internally. Local failures are recovered internally first. If internal recovery fails, the Backbone exits and is replaced by its outer control.

## 3. Core data objects

### STM and `canon(·)`

STM defines legal state advancement across steady-state execution, competition among redundant copies, out-of-order waiting, and recovery replay. `canon(·)` is not a materialized runtime function; it is the mapping induced by admission logic from possibly duplicated, reordered, or delayed physical arrivals to the logical history actually consumed by a component.

### Promise

A Promise is an externally specified state-and-input frontier contract defining the logical frontier that any valid holder must realize. Shadow state stores the configuration, snapshot, and boundary-input trace used by the current implementation to redeem that contract.

### Control and Handle

A Control manages one logical job and its task graph. A Handle represents a segment of remote execution capability and maintains its connection, resource locations, shadow state, and recovery entry point.

### Navigate

`Navigate` identifies the task's current model position and next execution target. Inference advances by step; training additionally carries version, minibatch, and direction.

### Resource-location map

The resource-location map associates logical nodes or model segments with physical execution locations and travels with the task. It is a job-scoped computational-location hint, **not a network routing table**. Root controls and Backbones append known locations. Entries may become briefly stale, so send failures and disconnect handling still repair the graph.

### Binding

A Binding combines a contiguous decoder shard group with task state and forms the rebindable boundary between a logical shard role/task state and its current physical placement. Configuration objects are separate from runtime shims; a shim translates upper-level tasks into operations understood by the Brain.

### Shadow state

Shadow state is an `InferStateData` or `TrainStateData` object attached to the control side of a handle. It retains the configuration, committed snapshot, and boundary-input trace needed to redeem a Promise. It is a control-plane mechanism, not the Promise itself, an independent worker, or a reserved GPU performing steady-state work.

## 4. Inference flow

1. An Agent creates an inference control and obtains candidate resources from Aux and known Backbones.
2. The control probes candidate paths and selects one or more execution handles from the candidates and probe samples in that construction round.
3. The prompt becomes a task carrying `Navigate` and a resource-location map.
4. A handle updates the shadow material used to redeem the Promise before sending the task.
5. A Dealer routes the task to the corresponding compute node.
6. The compute node applies STM admission to step, duplicate, and reorder state before executing the locally covered segment.
7. A Binding shim translates prefill/decode work into Brain operations.
8. The Brain trials candidate operations in the GPU-memory model and invokes real operators only after constraint checks pass.
9. Work continues forwarding when it leaves the local coverage range; the tail produces a boundary output and returns it to the Agent.

When structures match exactly, decode can execute head, decoder, and tail as a composite. Otherwise it uses staged eager/graph fallback.

With configured redundancy, the same logical input enters multiple selected realizations when it is sent. The first legal arrival advances state; subsequent duplicates are absorbed by STM/`canon(·)`. This is passive competition among selected copies, not continuous global fastest-path search.

## 5. Training flow

Training shares the control, handle, Dealer, compute-node, and Binding layers with inference but additionally carries:

- forward/backward direction;
- version and minibatch;
- gradient accumulation;
- optimizer/scheduler state;
- periodic checkpoints.

The reorder controller enforces phase and version constraints. Leaf gradients within one version accumulate by addition in `train_binding.py::do_weight_update`; the version advances only after the weight update.

## 6. Departure-driven topology change and recovery

A detectable disconnect or explicit relinquishment enters the recovery path of a job control or Backbone. Both indicate that a state-bearing physical realization has left the executable set:

1. Mark the failed handle and determine the affected range.
2. Query or select replacement resources.
3. Create a replacement handle and compute node.
4. Reconstruct resource locations and the admission state machine.
5. Redeem the Promise from the shadow's configuration, snapshot, and trace.
6. Replay required work after the checkpoint or retained input boundary.
7. Continue admitting subsequent tasks.

Multi-path redundancy submits one logical input to several selected realizations when it is sent. Shadow-assisted rebinding reconstructs a lost realization after its departure is observed. A surviving path can continue producing results, but it does not replace the physical-resource and state reconstruction performed by rebinding. This is reactive topology change, not general policy-driven topology optimization.

## 7. Intra-node execution

Intra-node execution has four layers:

1. compute node: decides whether a logical task should execute locally;
2. Binding shim: converts the logical request into operations without network semantics;
3. Brain: maintains multiple operation classes and selects, merges, and arbitrates work;
4. backend/operator: performs real device execution.

The Brain is the unified arbiter of device operations. `Provider` loads and caches model objects; `DecodersShim` maps logical Bindings to concrete operations.

## 8. GPU-memory digital twin

Candidate operations first modify a rollback-capable memory model:

```text
candidate operation
  -> modeled-object lifecycle changes
  -> restriction tree computes availability
  -> feasible: bind and execute real work
  -> infeasible: revert and attempt downgrade / swap
```

The trigger is whether a restriction-tree leaf has negative available bytes, not a universal GPU-occupancy percentage. Pool ratios and reservation parameters affect the constraints but do not replace object-level feasibility checks.

## 9. Asynchronous runtime

Ordinary coroutine work and potentially blocking CUDA/container work use different advancement paths. The event loop processes ready work first and then advances managed work from the blocking scheduler. The Brain's semantic queues are a higher-level operation classification and are not the runtime's two work classes.

## 10. Source-reading entry points

### Inter-node

- `node/agent/`
- `node/infer/infer_ctrl.py`
- `node/train/train_ctrl.py`
- `node/dealer_node/`
- `node/compute_node/`
- `node/backbone/`
- `node/shadow_node_state/`

### Intra-node

- `dealers/common/provider.py`
- `dealers/common/reorder.py`
- `dealers/common/brain.py`
- `dealers/memory_schedule/brain.py`
- `dealers/memory_schedule/restriction.py`
- `dealers/memory_schedule/swapper_core.py`
- `dealers/real_basic/`
- `operators/core/memory/env.py`

### Runtime entry points

- `scripts/offline/`
- `scripts/online/benchmark/`
- `exps/`

## 11. Capability boundaries

The source establishes control flow and state-update behavior, but concrete throughput, recovery time, memory-model error, power use, and numerical reproducibility require validation in a fixed environment.

| Architectural capability | Current implementation | Not implemented |
|---|---|---|
| Performance-informed path use | probing during graph construction and recovery; first-admissible-arrival competition among selected copies | continuous reprobe, a global-fastest guarantee, congestion/load-aware rerouting, proactive migration |
| Continuity and reconstruction under partial failure | concurrent execution constrained by STM/`canon(·)`, Promise/shadow recovery, and reactive rebinding | uninterrupted output without a surviving redundant path; failures outside the current failure model |
| Topology malleability | failure/explicit-relinquishment-driven replacement, graph rewriting, and split recovery | proactive path insertion/removal for performance, online repartitioning, continuous topology optimization |

Other capabilities absent from the default implementation include network-partition consensus, split-brain fencing, general NAT relaying, shared-bottleneck congestion control, automatic migration of persistently overloaded nodes, and recovery of the outermost Agent.
