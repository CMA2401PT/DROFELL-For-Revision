# Implementation Capabilities and Boundaries

This document summarizes behavior directly supported by the current code, properties that require runtime validation, and capabilities absent from the default implementation.

## 1. Behavior directly visible in the source

- The roles and call relationships among Agent/Request, Dealer, Compute, Aux, and Backbone.
- Task-graph, path, and remote-execution-instance management by controls and handles.
- Step/version admission, out-of-order waiting, and duplicate or stale input rejection.
- Multi-path dispatch, control-side shadow state, handle reconstruction, and task replay.
- Binding creation, execution, deletion, and shard-group granularity.
- The Brain's multi-class operation selection, default merge cap, and unified backend.
- Modeled objects, pools, rollback, and the constraint tree in the Memory Digital Twin.
- Checkpoint, downgrade, training-state swap, and KV-block swap branches.
- Separation of ready work from the blocking scheduler.
- The Context/BaseNode communication abstraction and responsibilities of the default Orbit backend.

## 2. Properties that require runtime validation

The presence of a code path alone does not establish:

- throughput, latency, peak GPU memory, or energy use on given hardware;
- time spent on failure detection, logical handoff, physical reconstruction, or the first valid output;
- whether multi-path execution masks a recovery window under specific network conditions;
- Memory Digital Twin error relative to the real allocator or its OOM-avoidance rate;
- stepwise numerical agreement, convergence speed, or final model quality after dynamic recovery;
- optimal merge limits, Binding granularities, or memory-policy parameters across models;
- CPU cost or whole-system power use of polling modes.

Measure these properties with the corresponding `scripts/` or `exps/` entry points while fixing the commit, inputs, random seeds, topology, and environment.

## 3. Shadow boundaries

Shadow state is control-side recovery material, not a standby compute worker:

- it does not perform steady-state model forward/backward;
- it does not reserve one GPU for each task;
- its lifetime is normally tied to one remote handle/job;
- recovery still requires discovery of replacement resources and physical model preparation.

Consequently, “no global stop-the-world in the control flow,” “logical handoff can proceed after failure detection,” and “end-to-end output has no gap” are three different statements.

## 4. Recovery boundaries

The current recovery paths handle disconnects or explicit failure events and reconstruct the task graph from control-side state. Note that:

- the Dealer side contains necessary serialization waits and barriers;
- top-level Agent recovery and Backbone-internal recovery do not obtain confirmed state in exactly the same way;
- without a checkpoint, historical input can be coalesced and the boundary decode replayed;
- with a checkpoint, KV state can be injected before chunked replay;
- a persistently overloaded but connected node is not automatically treated as failed;
- there is no cross-control consensus or split-brain arbitration under a network partition.

## 5. Network boundaries

The central service in the default Orbit backend handles deployment control, discovery, connection setup, and logging. It does not perform model scheduling or recovery. The default implementation:

- supports direct node links after central coordination;
- supports simple message-level latency and bandwidth simulation;
- does not model complete shared-bottleneck queuing or fair competition;
- does not provide general NAT traversal or a relay;
- does not replicate the Aux metadata service or elect its leader;
- does not provide consensus under network partitions.

Use Linux namespaces, `tc qdisc`, or an independent network testbed for stricter network experiments.

## 6. GPU-memory model boundaries

The digital twin models Brain operations rather than mirroring every internal CUDA allocator event. Its credibility comes from structural symmetry between real and modeled objects, not an empirical percentage. Sources of divergence include:

- CUDA/PyTorch allocator reservation and fragmentation;
- temporary workspaces in third-party kernels;
- external allocations outside managed object lifecycles;
- aliases, shared storage, and additional graph-capture lifecycles.

Deployments should retain both real CUDA metrics and modeled-pool metrics rather than checking only one side.

## 7. Numerical reproducibility boundaries

Dynamic replay can change floating-point accumulation order. Semantically identical input does not automatically imply bitwise identity, especially with custom attention, low-precision training, and repeated recovery. Validation should fix at least:

- model weights and initial adapter state;
- data order, random seeds, and sampling policy;
- attention/backend choice and precision;
- optimizer, scheduler, and gradient accumulation;
- failure points, checkpoint period, and replay mode.

Compare boundary outputs, per-version state fingerprints, loss and parameter differences, and final task metrics together.

## 8. Engineering boundaries

- Environment snapshots may contain machine-specific dependencies and should be reduced for the target deployment.
- Some early scripts require explicit model paths and device parameters.
- Exact upstream commits and per-file modification records for third-party code still need consolidation.
- Automated tests do not yet cover every online, recovery, and low-precision training combination.

Any capability description should distinguish “a call path exists in the source,” “the path passed under one configuration,” and “the property holds for every supported configuration.”
