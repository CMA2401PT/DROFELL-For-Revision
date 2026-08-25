# Core Concepts and Their Code Carriers

This document maps DROFELL runtime concepts to source objects. Some concepts are jointly realized by several fields and controllers; do not expect every concept to have a class with the same name.

## 1. Quick index

| Concept | Primary implementation |
|---|---|
| Legal state evolution | task step/version, `Navigate`, and reorder/drop controllers |
| Final-state contract | admission state machines, result waiting, shadow commits, and recovery replay |
| Semantic canonicalization | shared inference/training state-update and admission logic |
| Binding | `*BindingInit`, `*BindingShim`, and `Provider` |
| Resource-location map | a control's `res_location`, propagated with multi-path tasks |
| Shadow state | handle-side `InferStateData` / `TrainStateData` |
| Self-healing node | recursive controls and a local Aux under `node/backbone/` |
| GPU-memory digital twin | `MemoryBrain`, `Env`, the restriction tree, and `SwapperCore` |
| Dual-class asynchronous scheduling | ready work and the blocking scheduler in `extend_asyncio` |

## 2. State evolution

There is no single source object named `STM`. Inference and training compute nodes decide whether to execute, wait for, or drop a task using:

- inference: task step, Binding step, running state, and the reorder buffer;
- training: version, minibatch, forward/backward phase, and completion sets;
- path state: `Navigate`, which identifies the current model position and the next target;
- recovery: shadow state containing the minimum material needed to reconstruct the admission state machine.

Primary entry points:

- `node/compute_node/infer_node.py`
- `node/compute_node/train_node.py`
- `dealers/common/reorder.py`
- `dealers/common/provider.py`

## 3. Final-state contract

There is no standalone `Promise` class. The final-state contract is maintained jointly by the following mechanisms:

1. A control creates a logical task and waits for its boundary output.
2. A compute node admits only inputs that satisfy the step/version constraints.
3. Duplicate or stale tasks are dropped; early tasks wait.
4. A handle updates control-side recovery state before sending work to remote execution.
5. On failure, committed recovery material reconstructs a handle and replays the required inputs.

This contract neither predicts future paths nor requires every physical node to survive. It describes how a logical task can continue under the stated failure model and available resources.

## 4. Semantic canonicalization

There is no general `canon()` function in the source tree. Canonicalization arises because live and shadow inference state, and live and shadow training state, apply the same admission and update rules:

- steps and versions advance monotonically;
- duplicate input does not redeem a transition twice;
- out-of-order input executes only when its dependencies are satisfied;
- recovery resumes only from a committed boundary state;
- training leaf gradients accumulate by addition, and the version advances only after the weight update.

Training reorder safety cannot be attributed to `reorder.py` alone. Ordering enforces phase and version constraints; commutative gradient combination within one version is implemented by `dealers/real_basic/train_binding.py::do_weight_update`.

## 5. Binding

A Binding connects one contiguous decoder shard group to task state and can be written as `B = (*shards_group, S)`:

- `*BindingInit` describes configuration and initial state;
- `*BindingShim` translates a task into operations independent of the upper-level network path;
- `Provider` loads and caches model objects;
- the Brain/backend selects and executes operations.

`shards_group` is the primary granularity for migration, recovery, resource location, and intra-node merging. The default `max_concurrency` is 8; it is a batch-selection cap, not a correctness constant or a theoretical optimum.

## 6. Shadow state and recovery

There is no independent `ShadowNode` worker. Recovery material is attached to the control side of each remote execution handle:

- inference: `node/shadow_node_state/infer_state_data.py`
- training: `node/shadow_node_state/train_state_data.py`

These objects retain task metadata, committed boundary state, required inputs, and checkpoint information. They do not perform ordinary model computation and are not idle standby GPUs.

Multi-path redundancy sends the same logical input along several paths from the outset. Shadow recovery installs a replacement handle, updates resource locations, and replays required state after failure detection. The mechanisms can cooperate, but they have different lifecycles and responsibilities.

## 7. Backbone

A Backbone appears as one logical node to an outer control, while internally running virtual nodes and a local Aux:

- the outward virtual-node name remains stable;
- local failures and handle reconstruction are managed internally;
- cached parent trace/resource-map information compresses internal multi-hop structure back into one outer hop;
- if internal recovery fails, the Backbone exits and is recovered by the outer control.

The implementation is under `node/backbone/`.

## 8. GPU-memory digital twin

The GPU-memory digital twin is not a single occupancy threshold. Candidate operations first act on a rollback-capable object model:

1. `MemoryBrain` constructs modeled objects corresponding to real model objects.
2. `Env` records pools, object relationships, and snapshot/revert state.
3. The restriction tree computes each leaf's `available` value.
4. If any required leaf has negative availability, the candidate is infeasible.
5. `SwapperCore` then attempts downgrade, training-state swap, or KV-block swap.
6. Real work is committed only after a feasible modeled state is found.

Parameters such as `major_percentage` participate in pool constraints but are not a universal policy-switch percentage. Model fidelity depends on structural symmetry between real and modeled lifecycles, together with explicit handling of allocator reservations, external workspaces, and untracked allocations.

## 9. Dual-class asynchronous scheduling

`extend_asyncio` separates ordinary ready work from potentially blocking container/CUDA calls. The event loop drains ordinary work before advancing work from the blocking scheduler. The Brain's multiple semantic queues are a different layer and must not be conflated with the runtime's two work classes.

Busy polling is configurable. Go and Python backends, and blocking versus polling configurations, have different waiting behavior and must be selected explicitly for a deployment.

## 10. Network abstraction

Upper-level controls use communication through `Context` / `BaseNode`. The default Orbit implementation provides deployment, discovery, connection setup, and logging, while direct data links may be established between nodes. A replacement backend must preserve the contracts for node registration, send/receive, disconnect notifications, and location resolution.

The default implementation is not a fully decentralized network and does not provide network-partition consensus, general NAT relaying, or shared-link congestion control.
