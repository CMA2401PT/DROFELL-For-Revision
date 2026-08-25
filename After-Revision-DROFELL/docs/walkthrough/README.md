# From the Problem to the System: A Progressive DROFELL Walkthrough

This series begins with the physical constraints of sharded LLM execution over a loosely coupled network rather than with class names or module tables. Each chapter first exposes an unresolved problem and then introduces the state semantics, runtime mechanism, or engineering structure required to solve it.

Read the chapters in order:

1. [From a Fixed Chain to a Reconfigurable Layered Multi-Path Graph](01-fixed-chain-to-reconfigurable-layered-multipath-graph.md)
2. [Why Path Changes Require State Semantics First](02-why-path-changes-require-state-semantics.md)
3. [Two Resilience Mechanisms and the Layered Computation Graph](03-resilience-mechanisms-and-layered-computation-graph.md)
4. [From One Task to Many](04-from-one-task-to-many.md)
5. [Why CUDA Execution Needs a Different Asynchronous Runtime](05-why-cuda-needs-a-different-async-runtime.md)

## Relationship to the reference documentation

- This series explains **why the system has its current shape**.
- [`../architecture-overview.md`](../architecture-overview.md) describes the current layers and roles.
- [`../concept-to-code-map.md`](../concept-to-code-map.md) maps concepts to their source-code carriers.
- [`../inter-node-control-and-recovery.md`](../inter-node-control-and-recovery.md) and [`../intra-node-execution-and-memory.md`](../intra-node-execution-and-memory.md) provide mechanism details.
- [`../implementation-boundaries.md`](../implementation-boundaries.md) consolidates capability boundaries and runtime verification requirements.
- [`../../scripts/README.md`](../../scripts/README.md) and the README files under `exps/` explain how to run and validate the system.

## Reading convention

The series keeps three levels separate:

| Level | Question answered |
|---|---|
| **State semantics** | Which arrivals may advance a task, and what must be reconstructed after changing the physical holder? |
| **Runtime mechanisms** | How can physical realizations coexist, fail, and be replaced? |
| **Adaptation policy** | Which paths should be selected, and when should they be added, removed, migrated, or reoptimized? |

The current implementation is most complete at the state-semantics and failure-resilience levels. It probes paths during graph construction and recovery, lets selected copies compete, and rewrites the graph after failure or explicit relinquishment. Continuous fastest-path routing and general policy-driven online topology optimization are outside the current implementation.

Code references use `file::symbol`, not line numbers.
