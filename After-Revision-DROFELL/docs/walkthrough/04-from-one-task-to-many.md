# From One Task to Many

## 1. A single-task graph hides intra-node problems

The previous chapters described a holder as `D1` or `D2`, as if each node served one task and owned one model segment exclusively. In the real system:

- several tasks may traverse the same Dealer;
- each task has independent KV cache, training state, and admission progress;
- tasks may share the same foundation weights;
- one Dealer may load several contiguous model segments;
- inference and training may coexist over one set of foundation weights.

A Dealer therefore cannot be modeled as “a model plus one task state.” A more accurate structure is:

```text
Dealer = (foundation model shards, Bindings)

Binding = (*shards_group, task_state)
```

Foundation weights belong to the provider/holder, while task state belongs to a Binding.

## 2. Why use shard groups

If each decoder layer were an independent migratable node, path, handle, checkpoint, and resource-location metadata would grow rapidly. If the entire model were one unit, placement and local recovery would lose flexibility.

DROFELL uses a shard group as the primary compromise:

- a contiguous sequence of decoder layers;
- the head;
- or the tail.

This granularity participates in model loading, Bindings, resource locations, intra-node merging, checkpoints, and recovery splitting. It is a deployment parameter rather than one universally optimal value.

## 3. Why multiple tasks must merge

One decode step for one sequence is small. If every task launches kernels independently, the GPU sees many small batches and scheduling gaps. Different tasks have independent state but may request the same operation over the same shard group:

```text
task A: Binding(*group_2, state_A) -> decoder step
task B: Binding(*group_2, state_B) -> decoder step
```

They cannot share task state, but the backend can execute them in one batch. This forces intra-node execution to separate logical task handling from device operations.

## 4. Deriving the four-layer execution structure

### Holder / Provider

Retains loaded foundation-model objects and creates Binding shims for tasks. Foundation weights are reused through caching while task state remains isolated.

### Router / Compute Node

Reads `Navigate` and step/version state, locates the current Binding, and decides whether the task should execute locally. It does not launch CUDA kernels directly.

### Mailboxes

Receive `Operate*` objects created by shims. Operations are inserted in compatible-key order so that work for identical or compatible Bindings becomes adjacent and eligible for batch selection.

### Backend / Brain

Selects candidate operations from mailboxes, checks GPU-memory feasibility, restructures, downgrades, or swaps when necessary, and invokes the real operator.

```text
logical task
  -> compute node/router
  -> Binding shim
  -> semantic mailbox
  -> Brain selection
  -> memory trial
  -> real operator
```

The backend is the unified arbiter of device state. Upper-level coroutines may arrive concurrently, but one execution loop chooses the next device-operation group, preventing multiple controls from mutating device state outside one memory model.

## 5. “Supports merging” versus “takes a batch”

Every mailbox has comparison and ordered-insertion logic, so compatible operations can be placed together structurally. In the current implementation, the primary paths that actually take several operations at once are inference head, tail, prefill, and generating; training queues normally choose one logical operation per round.

Merging also requires:

- compatible Bindings and operation types;
- compatible shapes, token counts, and adapter conditions;
- an available graph variant or staged eager support;
- feasibility of the complete batch under memory constraints;
- selection count no greater than `max_concurrency`.

The default merge limit of 8 is configurable scheduling policy, not a semantic constant.

## 6. Why static GPU-memory quotas are insufficient

A task's GPU-memory state changes during execution:

- KV cache grows with token count;
- training context survives between forward and backward;
- checkpoint policy determines retained intermediates;
- graph capture and kernel workspaces add memory;
- several candidate operations may pressure different pools simultaneously.

Checking after real execution is too late, while reserving every worst case wastes capacity. Before committing work, the Brain therefore asks:

> If this operation group executes, do all constrained memory pools remain feasible?

## 7. Memory Digital Twin

The Memory Digital Twin is a rollback-capable model parallel to real object lifecycles:

```text
snapshot
  -> apply candidate operation to modeled objects
  -> recompute restriction tree
  -> feasible: bind and commit real operation
  -> infeasible: revert and try another strategy
```

### Objects and pools

`MemoryBrain` creates modeled objects corresponding to real model, KV, training-context, and user-model objects. `Env` records object relationships, pools, and transaction logs.

### Constraint tree

The tree organizes foundation model, operation reservation, KV, training context, user model, gradient, and optimizer pools. A required leaf with `available < 0` makes a candidate infeasible.

Parameters such as `major_percentage` affect pool budgets, but the system does not switch policy at one global GPU-occupancy percentage.

### Fidelity boundary

The digital twin models Brain operations and relies on structural symmetry between real and modeled objects. Allocator reservation, fragmentation, third-party kernel workspaces, aliases, and graph-capture lifecycles still require measurement or explicit reservation.

`unmanaged` is the default accounting location for modeled in-flight objects not yet assigned to a named application pool. It is not a universal sink for every unknown CUDA allocation.

## 8. Why infeasibility has several escape paths

Memory objects have different reconstruction costs, so “free GPU memory” is not one operation.

### Downgrade

Change training-context retention strategy, exchanging recomputation for lower residency.

### Training-state swap

Move migratable training state to CPU and load it again later.

### KV-block swap

Select KV blocks for eviction with a score containing generation/position encoding. This is a different policy from training-context downgrade.

### Checkpoint branches

Branches retain a full gradient graph, per-layer input, first-layer input, or reconstruction seeds, trading forward time, backward recomputation, and GPU memory.

The Brain trials these branches in the rollback-capable model and executes real work only after finding a feasible state.

## 9. Who coordinates tasks across nodes

Intra-node execution has one Brain, but there is no global controller scheduling every Dealer operation.

- Each Agent manages its own job graph.
- Aux aggregates only non-real-time metadata such as providers, TPS, and Backbones.
- Dealers decide autonomously what to load from metadata.
- Tasks carry `Navigate` and job-scoped `resource_loc` for forwarding.
- Backbones run recursive controls in local domains.

Aux is therefore a thin discovery point, not a global execution scheduler. Orbit's central service is a lower-level deployment and connection facility, not the same role.

## 10. One execution-layer problem remains

Even after the Brain chooses the next operation group, Python's event loop still faces CUDA synchronization, launch backpressure, thread-local contexts, and streams. Ordinary asyncio assumptions do not directly cover these behaviors.

The next chapter explains why intra-node execution additionally needs dual-class scheduling, Containers, and dedicated CUDA waiting paths.

## Further reading

- [Why CUDA Execution Needs a Different Asynchronous Runtime](05-why-cuda-needs-a-different-async-runtime.md)
- [Intra-Node Execution and GPU-Memory Governance](../intra-node-execution-and-memory.md)
- [Memory Digital Twin Capture and Replay](../memory-twin-capture-and-replay.md)
