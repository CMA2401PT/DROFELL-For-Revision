# Two Resilience Mechanisms and the Layered Computation Graph

## 1. State semantics still need runtime actions

STM, `canon(·)`, Promise, and Binding allow different physical executions to reduce to the same logical progress, but they do not create paths or recover nodes by themselves. The current runtime uses two cooperating mechanisms:

1. **concurrent coexistence and competition among selected realizations**;
2. **shadow-assisted reactive rebinding**.

The first lets a surviving copy continue; the second reconstructs the lost physical execution.

## 2. Concurrent selected realizations

Suppose logical layer `L1` is currently implemented by both physical nodes `B` and `C`:

```text
          -> B ->
L0 output          L2 admission
          -> C ->
```

The same logical input enters several selected handles when it is sent, not only after a failure:

- the first legal arrival may advance downstream state;
- an equivalent late result is recognized as a duplicate;
- if one path fails, the other already owns the input and execution state and does not wait for a reactive switchover.

Competition occurs only among a finite set of already selected copies. The system does not continuously scan all providers for every token, nor does it guarantee that each step uses the globally fastest path.

## 3. Why the surviving copy does not replace recovery state

When `B` fails while `C` survives, it is tempting to initialize a new node `E` from `C`. That makes recovery depend on another compute peer:

- `C` may be processing a different step;
- `C` may fail during transfer;
- another protocol is needed to identify which committed frontier the copied state represents;
- without redundancy, no `C` exists.

The state source for a new holder is therefore not a surviving copy. It is the Shadow state continuously maintained for the original Binding on the control side.

## 4. Shadow is a state structure, not a standby node

In the source, each remote handle owns a control-side:

- `InferStateData` object;
- `TrainStateData` object.

It retains configuration, committed snapshots, boundary input, and confirmed history needed to redeem a Promise. Important properties are:

- it performs no steady-state model forward/backward;
- it does not represent a reserved idle GPU;
- it resides with the control, and updates are normally local state operations;
- each handle/job has its own recovery state rather than mixing frontiers from different jobs.

Before sending real remote work, a handle first updates the corresponding Shadow material. If the remote side fails before or after receipt, the control side still has a recovery boundary constrained by commit rules.

## 5. Why “received” is not the same as “recoverable”

The control side may receive base state, boundary input, and output confirmation at different times. Deleting old material immediately would leave a gap if failure occurs while confirmation is propagating. Shadow state therefore separates receipt from commit:

1. New material enters candidate state.
2. The corresponding output or checkpoint satisfies commit conditions.
3. The new frontier is promoted into recoverable state.
4. Only then is the old retained set reduced.

The source also contains task-progress barriers, Brain execution barriers, and deferred-send barriers for training checkpoints. These solve different problems and do not form one global distributed transaction.

## 6. Reactive rebinding after departure

A disconnect, explicit failure, or voluntary departure enters the corresponding control's recovery path:

```text
observe departure
  -> locate affected Handle / Binding
  -> probe or query replacement resources
  -> create replacement Handle and Compute Node
  -> reconstruct resource locations and admission state
  -> redeem the Promise from Shadow state
  -> replay required input after the checkpoint boundary
  -> continue admitting work
```

This is reactive graph rewriting: the trigger is departure of a realization from the executable set, not continuous cost-based topology optimization.

### State snapshot available

Inference may inject a KV checkpoint; training may load a complete handle checkpoint before replaying later boundary input.

### No state snapshot

Inference coalesces confirmed history into fewer replay operations and retains the unconfirmed boundary decode. Training reconstructs an available version from configuration and retained input.

### Split recovery

The shard groups covered by one failed Binding may be split across several replacement holders. The state object is partitioned for the new ranges while logical roles and admission frontiers are reconstructed.

## 7. Three timing boundaries must remain distinct

Recovery contains at least three different questions.

### Does the whole control flow stop?

Recovery is owned by the directly related Agent or nearest Backbone and does not require every job and node to enter a graph-wide barrier.

### Does logical handoff wait for physical recovery?

After the replacement graph is installed and recovery packets enter the local send path, the control can complete logical handoff without waiting for the new Dealer to finish loading and acknowledge it.

### Does the user continue receiving output?

Only unbroken concurrent redundancy can keep producing results for the affected logical layer during physical recovery. Without a surviving path, the first valid output may still be delayed.

“No graph-wide stop-the-world,” “non-blocking logical handoff after failure detection,” and “continuous end-to-end output” are therefore three different properties.

## 8. Why a Backbone is needed

As a flat multi-path graph grows, an Agent faces many provider combinations, recovery domains, and internal network details. A Backbone encapsulates a local group as one logical realization at the outer layer:

```text
outer graph:  L0 -> [Backbone logical node] -> L2
                         |
inner graph:          B1  B2  B3
```

A Backbone runs its own control and local Aux:

- internal realizations may coexist concurrently;
- internal holder failure first triggers local rebinding;
- the outward virtual-node name remains stable;
- an internal multi-hop trace is compressed into one outer hop;
- if local recovery fails, the Backbone exits and the outer layer replaces the logical node.

This creates layered failure domains: the nearest recovery domain absorbs a local failure when possible and escalates it outward as an ordinary departure otherwise.

A Backbone also reduces the outer candidate space and can encapsulate local direct links, private networks, or jump-host environments behind one logical node. It is not a general topology optimizer; configuration, probing, and reactive recovery still determine its internal organization.

## 9. Why the resource-location map is only a hint

`resource_loc` identifies currently visible compute locations for logical nodes. It guides forwarding, but is neither an IP routing table nor a strongly consistent membership view:

- a root control merges known locations;
- a Backbone appends internally visible locations;
- mappings travel with tasks and may become briefly stale;
- a stale entry may cause one invalid send;
- disconnect handling, barriers, and handle reconstruction repair the graph.

The current implementation has no map lease, quorum, term, or network-partition fencing. Ordinary node departure and split-brain are different failure models.

## 10. One inference request through the graph

```text
Agent InferControl
  -> MultiPathInferTask
  -> one or more InferNodeHandle
  -> Shadow update
  -> DealerNode
  -> per-task Compute Node
  -> Binding shim
  -> Brain / MemoryBrain
  -> operator
  -> next resource_loc or final result
```

After failure, the old handle's Shadow is partitioned or transferred to replacement handles. Replay uses the same STM admission path, so old in-flight packets and recovery packets can be waited for, accepted, or absorbed at their convergence point.

## 11. Current boundaries

- A persistently overloaded but connected holder is not automatically treated as failed.
- There is no general continuous fastest-path routing.
- There is no shared-link congestion control.
- The default network backend does not provide network-partition consensus or general NAT relaying.
- Without a surviving redundant path, affected requests are not guaranteed continuous end-to-end output.

## Further reading

- [From One Task to Many](04-from-one-task-to-many.md)
- [Inter-Node Control, Forwarding, and Recovery](../inter-node-control-and-recovery.md)
- [Network Abstraction and the Default Orbit Backend](../network-abstraction-and-orbit-backend.md)
- [Implementation Capabilities and Boundaries](../implementation-boundaries.md)
