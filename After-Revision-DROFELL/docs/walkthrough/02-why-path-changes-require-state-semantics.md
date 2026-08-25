# Why Path Changes Require State Semantics First

## 1. The most obvious recovery arrangements

Return to `R -> D1 -> D2 -> D3`. A conservative recovery design sends every intermediate result through `R`:

```text
R -> D1 -> R -> D2 -> R -> D3 -> R ...
```

`R` now retains replay input, but the network-hop count grows substantially. A second design preserves the short path while sending side copies to the control side:

```text
R -> D1 -> D2 -> D3 -> R
       \-> control
             ^
       D2 ----/
```

The critical path is shorter, but the design exposes a problem deeper than whether a backup exists.

## 2. Ambiguity left by in-flight packets

At step `i`:

- `D1 -> D2` sends input `H_i_in`;
- `D1 -> control` retains the same input;
- `D2 -> D3` sends output `H_i_out`;
- `D2 -> control` retains the same output.

If `D2` fails in the middle, surviving parties cannot know how far each transfer progressed. Packets already emitted may arrive after failure or even after a replacement is installed.

Recovery now faces two symmetric risks.

### Resend

If the old `D2 -> D3` transfer succeeded, a replacement that sends again executes the same logical step twice.

### Do not resend

If the old packet never arrived, a replacement that refuses to resend leaves the task permanently stalled.

More accurate failure detection does not solve the problem. The last local action occurred at the node that disappeared, and there is no observable global instant that reconstructs the true state of every node and in-flight packet.

Recovery must therefore avoid reproducing one unobservable physical execution point. It needs different physical arrival histories to reduce to the same meaningful logical state.

## 3. STM: which input may advance state

The State Transition Matrix (STM) is not one matrix object in the runtime. It is a set of admission rules answering:

> Given the state already accepted by a component, should the current arrival execute, wait, or be dropped?

### Inference

- Prefill establishes step 0.
- Decode steps advance monotonically.
- Duplicate arrivals for completed steps are dropped.
- Future steps wait until dependencies are satisfied.
- Only a legal running step may execute.

### Training

- A version advances only at a complete batch/update boundary.
- Backward must correspond to an already completed forward.
- Minibatch, direction, and version jointly determine admission.
- The logical version advances only after the weight update.

Primary carriers are step/version state in compute nodes and the admission controllers in `dealers/common/reorder.py`.

## 4. `canon(·)`: from physical arrivals to logical history

An actual arrival stream may contain:

```text
step 3, step 2, duplicate step 2, early step 4, late step 1
```

The component nevertheless consumes one legal history. `canon(I)` denotes the semantic mapping from physical arrivals `I` to the consumed logical history.

It is not a source-level `canon()` function. It is induced jointly by:

- rejecting stale arrivals;
- absorbing duplicates;
- parking future arrivals;
- executing in logical order after dependencies hold;
- advancing state only after commit conditions hold.

Two important consequences follow:

1. Reordering irrelevant packets does not change the consumed history.
2. Adding an already consumed duplicate does not change the consumed history.

Packets from an old node, a redundant copy, and a replacement can therefore meet at one admission state machine without first electing a “true original.”

## 5. Promise: what a new holder must reconstruct

Admission rules answer what happens after an arrival, but a replacement holder still needs a starting point. A Promise is a transferable state-and-input frontier contract described by:

- initial configuration;
- retained boundary-input trace;
- an optional committed state snapshot.

Redeeming a Promise does not reproduce the old holder's last machine instruction. It reconstructs a logical frontier equivalent under STM/`canon(·)`.

Recovery must not query the failed holder and should not depend on a surviving compute copy for current state. Required material must already have entered a committable recovery state on the control side.

## 6. Binding: the seam between a logical role and physical placement

A physical node may load several foundation-model shards and serve many tasks concurrently. Logical task state cannot simply belong to “this machine.” A Binding separates them:

```text
Binding = (*shards_group, task_state)
```

- `shards_group` identifies a logical model role;
- `task_state` identifies this task's state at that role;
- the physical holder is only the current placement realizing the Binding.

When a holder changes, recovery reconstructs the logical role and its Promise frontier rather than redefining the task graph's logical meaning.

## 7. The four concepts form one loop

```text
physical arrivals
      |
      v
STM admission ------> wait / drop / execute
      |
      v
canon-consumed logical history
      |
      v
committed Promise frontier
      |
      v
Binding can be realized by another holder
```

In short:

- STM defines legal advancement.
- `canon(·)` defines which physical histories are semantically equivalent.
- Promise defines the frontier a replacement holder must reconstruct.
- Binding provides the rebindable boundary between a logical role and its current placement.

These concepts form a semantic substrate. They neither select the fastest path nor decide when to scale. They make path competition and replacement safe runtime operations.

## 8. Training reorder requires one additional fact

Training ordering control guarantees legal forward, backward, version, and minibatch relationships, but does not by itself prove that gradient combination is commutative. Leaf gradients within one version accumulate by addition in `dealers/real_basic/train_binding.py::do_weight_update`.

Training reorder safety therefore has three parts:

1. admission state machines prevent illegal phases and versions;
2. legal leaf contributions accumulate with commutative addition;
3. the optimizer step completes before the logical version advances.

Floating-point addition order may still affect bitwise reproducibility. Semantic legality and bitwise identity are different requirements.

## 9. Next: move semantics into the runtime

Path-independent task progress is now defined. The next chapter uses it to construct two cooperating mechanisms: concurrent selected physical realizations and control-side reconstruction after one realization fails.

## Further reading

- [Two Resilience Mechanisms and the Layered Computation Graph](03-resilience-mechanisms-and-layered-computation-graph.md)
- [Core Concepts and Their Code Carriers](../concept-to-code-map.md)
- [Inter-Node Control, Forwarding, and Recovery](../inter-node-control-and-recovery.md)
