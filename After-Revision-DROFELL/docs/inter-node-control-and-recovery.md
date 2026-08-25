# Inter-Node Control, Forwarding, and Recovery

## 1. Control structure

```text
Aux metadata
     |
Agent/Request Control
     |
execution Handle(s)
     |
DealerNode -> Compute Node -> next Dealer / result
     |
optional Backbone-local control
```

Each job graph is managed by its corresponding control. Aux supplies resource metadata, a Dealer performs network routing, a compute node advances one task's state, and a Backbone can recursively organize the same structure internally.

## 2. Communication foundation

`BaseNode` accesses the following capabilities through `Context`:

- node-location registration and resolution;
- connection establishment and reuse;
- message send, receive, and deserialization;
- connection-up and disconnect notifications;
- one interface for colocated and remote placements.

Orbit provides the default Context. An upper-level control must not depend on implementation details of a particular tracker, socket, or NAT scheme when the backend is replaced.

## 3. Provider discovery and path selection

### Aux metadata

Aux aggregates:

- decoder groups loaded by Dealers;
- provider/TPS information;
- registered Backbones;
- node metadata needed for path selection.

Aux neither retains real-time task state for every request nor decides each forwarding step.

### Dealer loading decisions

A Dealer queries aggregate information, chooses model shards to load, and announces them to Aux and selected Backbones. This occurs primarily during initialization and loading; it is not a continuous CPU/GPU-utilization feedback loop that automatically migrates models.

### Path probing

A control enumerates candidate providers or Backbones and probes reachability, latency, or bandwidth. Several viable paths may receive handles concurrently. Probe values are selection inputs, not guarantees of shared-bottleneck fairness or long-term stability.

## 4. Control, Handle, and resource-location map

A Control stores job-level indexes, task futures, the logical graph, and handle sets. A Handle:

- communicates with one remote execution instance;
- updates shadow state before sending;
- maintains task barriers and reorder state;
- receives remote results or checkpoints;
- triggers upper-level recovery after a disconnect.

`resource_loc` travels with a task and locates logical nodes. A root control normally merges and deduplicates locations from its handles; a Backbone can append internally visible locations. A stale entry may cause one invalid send, while disconnect handling, barriers, and reconstruction repair the execution graph.

## 5. Inference forwarding

1. The root control creates a `MultiPathInferTask`.
2. Each candidate handle retains the input and current boundary state.
3. The Dealer dispatches the task through its handle/compute-node index.
4. The compute node applies `should_drop`, reorder/wait, and running-state checks.
5. Local Bindings execute the covered head/decoder/tail segments.
6. When `Navigate` points outside the local placement, the task returns to the Dealer for forwarding.
7. The tail result returns to the root control; admission state deduplicates repeated results.

Multi-path execution submits work to several paths when it is sent. It does not wait for a failure and then switch to a path that never received the input.

## 6. Training forwarding

Training tasks additionally carry direction, version, minibatch, checkpoint, and model state. A forward path turns into a backward path at the tail and returns along the path. A training handle promotes a version into ready recovery state only after all required checkpoints in its coverage range are complete.

Primary training entry points:

- `node/train/train_ctrl.py`
- `node/dealer_node/train_handle.py`
- `node/compute_node/train_node.py`
- `node/shadow_node_state/train_state_data.py`

## 7. Shadow state

There is no independent Shadow worker. Each remote handle owns a control-side state object:

- inference: initialization metadata, boundary input, confirmed output, KV checkpoints, and replay material;
- training: versions, inputs, completed items, and complete checkpoint sets.

StateData does not perform steady-state model computation and does not reserve a GPU. One process may manage multiple jobs, but recovery objects remain isolated by handle/job.

## 8. Commit barriers

The source contains several distinct barrier layers:

1. task logical-progress barriers;
2. Brain operation-execution barriers;
3. receive/commit barriers for shadow base state;
4. deferred-send barriers for training checkpoints.

Shadow state first receives candidate updates and shrinks its retained set only after the corresponding output or checkpoint meets commit conditions. Recovery continues from a committed boundary, not merely the last packet received but not yet confirmed.

## 9. Recovery call path

Recovery begins after a disconnect, explicit failure injection, or Backbone-internal failure:

1. Locate the failed handle and affected logical range.
2. Freeze or serialize related Dealer operations.
3. Query a replacement provider.
4. Create and register a replacement handle.
5. Update resource locations and the logical graph.
6. Install admission state and committed recovery material.
7. Send required checkpoints and historical input.
8. Release barriers and continue execution.

The Agent side may submit subsequent logical work after the replacement graph is installed, while necessary waits still exist on the Dealer side. Measure failure detection, replacement-graph installation, logical handoff, physical model recovery, and the first boundary output separately.

## 10. Inference replay

### No available KV checkpoint

Recovery coalesces confirmed history through the penultimate input and replays the final decode to reconstruct boundary KV and output state.

### Available KV checkpoint

Recovery injects the checkpoint, then replays later input in chunks. The confirmed output prefix determines which tokens must not be redeemed again.

Recovery configuration selects the mode. Both paths obey step admission and duplicate-suppression rules.

## 11. Training replay

Training recovery uses a complete handle checkpoint plus later inputs:

- the checkpoint must cover all required user models in the handle;
- tasks older than the ready checkpoint are not re-executed;
- forward/backward and version state are reconstructed;
- replay-completion state is managed separately from ordinary execution state.

Low-precision kernels, attention backends, and floating-point accumulation order may affect bitwise reproducibility. Validate with state fingerprints, parameter differences, and task metrics.

## 12. Backbone-local recovery

Internal virtual nodes retain stable outward names. Recovery packets use cached parent trace and resource-location information rather than exposing the current internal multi-hop topology; `trace_override` compresses the internal route into one outer hop.

If internal recovery fails, the Backbone exits and lets the outer layer replace it as an ordinary failed node. This implements local-first recovery with failure escalation.

## 13. Failure model and limitations

The current recovery path primarily handles node disconnects and explicit failure events:

- no resource-map version, lease, quorum, term, or fencing;
- no split-brain arbitration under network partitions;
- a highly loaded but connected node does not automatically trigger general migration;
- the default network backend does not guarantee NAT relaying;
- path probing does not implement shared-link congestion control;
- Aux and Orbit tracker have no default replication or leader election.

## 14. Key source files

- `node/base_node.py`
- `node/aux_node/`
- `node/path_utils.py`
- `node/infer/infer_ctrl.py`
- `node/train/train_ctrl.py`
- `node/dealer_node/`
- `node/compute_node/`
- `node/shadow_node_state/`
- `node/backbone/`
- `node/recovery_metrics.py`
- `utils/runtime_metrics.py`
