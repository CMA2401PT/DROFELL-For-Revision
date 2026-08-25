# From a Fixed Chain to a Reconfigurable Layered Multi-Path Graph

## 1. Start with the simplest sharded execution

Suppose a large language model is split into three logically ordered segments hosted by `D1`, `D2`, and `D3`, and requests enter through `R`. The most direct autoregressive chain is:

```text
R -> D1 -> D2 -> D3 -> R -> D1 -> D2 -> D3 -> R ...
```

This resembles an ordinary pipeline, but Internet and edge nodes expose three fundamental problems.

### Every token pays for the complete network circuit

Decode is autoregressive: the next token depends on the previous token's output. The chain is therefore traversed for every generated token rather than once per request. A slow link, excessive hop count, or one slow holder repeatedly enters the critical path.

### A disappearing node takes task history with it

Foundation-model weights can normally be reloaded from another copy. KV cache, training context, and task version are different: they are accumulated by the current request. Losing them does not merely require rerunning one operator; it loses the history that defines current state.

### A physical chain cannot naturally absorb resource changes

Even when a closer, faster, or newly available holder appears, an in-flight task remains bound to the original physical chain. Removing one holder, adding a copy, or migrating one segment changes the structure on which in-flight work depends.

The problem is therefore not only how to make a fixed chain faster, but how to keep logical work from being defined by one fixed physical chain.

## 2. The target remains a layered graph

DROFELL targets a **state-centric, reconfigurable, layered multi-path computation graph**:

- **State-centric:** progress is defined by accepted logical state and input frontiers, not a physical node's local instruction pointer.
- **Reconfigurable:** a new physical holder can assume the same logical role.
- **Layered:** edges still follow the model's logical layer order; the system is not an arbitrary network mesh.
- **Multi-path:** one logical transition may have several physical realizations at the same time.

Conceptually:

```text
logical layer 0      logical layer 1       logical layer 2
     [L0]  ------->       [L1]  ------->        [L2]
       |                   /  \                    |
       |              physical B  physical C      |
       |                   \  /                    |
       +-------------- logical transition --------+
```

`B` and `C` may both realize the `L1` transition, but their physical packets must ultimately produce only one legal logical advancement.

## 3. The complete objective contains three capabilities

A complete online adaptation system would eventually need to:

1. select higher-quality paths from runtime observations;
2. continue through surviving paths after partial failure and reconstruct lost physical realizations;
3. add, remove, migrate, or reoptimize paths as resources change.

The current implementation covers these goals to different degrees:

| Capability | Current implementation |
|---|---|
| Runtime-quality-informed path selection | probe candidates during graph construction and recovery; passive competition among selected copies |
| Continuation and reconstruction after partial failure | multi-path survival, control-side recovery state, replacement handles, graph updates, and replay |
| General online topology optimization | replacement, split recovery, and graph rewriting driven by failure or explicit relinquishment |

The current system does not continuously scan all providers for every token, guarantee a globally fastest path, or run a general policy layer that continually rewrites topology according to long-term load, cost, or resource changes.

## 4. Why an “intelligent router” cannot come first

Assume a policy can move logical layer `L1` from `B` to `C` at any time. Three unanswered questions remain:

- A packet already emitted by `B` may arrive after the switch.
- `B` and `C` may both produce output for the same logical step.
- `C` does not know from which state and input boundary it should resume.

Without answers, a more aggressive policy only creates more duplicates, reorderings, and state forks. Path optimization is not a substitute for state semantics; it depends on them.

This requirement separates the architecture into three levels.

### State semantics

Define which input may advance state, when different physical arrivals are equivalent, and what a replacement holder must reconstruct.

### Runtime mechanisms

Use those semantics to let physical realizations coexist safely and to replace failed realizations.

### Adaptation policy

Use quality, cost, and resource information to decide which realizations should exist.

The current system first closes the loop across the first two levels and exposes bounded primitives for the third: construction-time probing, redundancy configuration, and reactive graph rewriting.

## 5. Two implementation control axes

Once the task graph can change, the system must manage two scales simultaneously.

### Inter-node control

It answers:

- Which holders can serve a logical model segment?
- Which paths currently realize one job?
- Where should a packet be forwarded?
- Who creates a replacement path after failure?
- How does a new holder obtain task state?

### Intra-node control

It answers:

- How do many tasks share one set of foundation weights?
- Which operations can merge?
- Which operation executes next?
- How does the runtime downgrade or swap when GPU memory is insufficient?
- How does CUDA work coexist with ordinary asynchronous tasks?

Together these axes implement the reconfigurable computation graph. They are not two recovery mechanisms: inter-node control manages the graph, while intra-node control manages execution inside a holder.

## 6. Next: define path-independent progress

Before two paths may race or one logical role may move to a new holder, the system must answer:

> When duplicates, reordering, late arrival, and replay coexist, what counts as one forward step of the task?

The next chapter begins with in-flight packets left by a failed node and derives STM, `canon(·)`, Promise, and Binding.

## Further reading

- [Why Path Changes Require State Semantics First](02-why-path-changes-require-state-semantics.md)
- [Architecture Overview](../architecture-overview.md)
- [Implementation Capabilities and Boundaries](../implementation-boundaries.md)
