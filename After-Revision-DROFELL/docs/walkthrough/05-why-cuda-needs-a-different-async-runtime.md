# Why CUDA Execution Needs a Different Asynchronous Runtime

## 1. The Brain decides what; the event loop decides when

The previous layer converts logical tasks into device operations in the Brain. It is tempting to give all network, task, and GPU work to asyncio, but ordinary asynchronous-I/O systems typically assume:

- operations do not truly block the thread for long;
- coroutines yield quickly;
- compute-heavy work does not occupy the event loop;
- one asynchronous scheduling model describes completion.

CUDA and PyTorch violate those assumptions at several important points.

## 2. Four mismatches

### Synchronization points really block

Calls such as `cudaEventSynchronize`, `.item()`, and `.cpu()` can pin the execution thread. If they run on the event-loop thread, networking and every other coroutine stop progressing.

### Kernel launch may encounter backpressure

Launch usually returns quickly, but a full device queue or runtime pressure can make launch itself block. Treating it as an always-non-blocking callback is unsafe.

### Torch context follows threads, not coroutines

Inference, grad/no-grad, autocast, and related contexts are thread-local. When coroutines interleave on one thread, contexts and streams can leak across tasks unless the runtime defines explicit boundaries.

### CUDA streams form another asynchronous system

A Python future completing, a kernel being launched, and GPU work finishing are three different events. Asyncio task state alone cannot describe all device completion relationships.

DROFELL does not eliminate these differences; it confines them within controlled boundaries.

## 3. First isolation: separate ordinary and device work

The runtime distinguishes:

- ordinary ready work: network activity, state updates, future callbacks, and other short coroutine work;
- blocking/device work: CUDA or potentially blocking operations carried by Containers.

The event-loop rule is approximately:

```text
ready work exists at the start of this round
  -> execute that snapshot of ready work
otherwise
  -> advance one item from the blocking scheduler
```

The loop does not endlessly drain callbacks created during the current round. It snapshots the ready count, leaving newly created callbacks for the next round so that blocking work cannot be starved forever.

Only one blocking item advances per round, after which the loop checks ready work again. Network and state callbacks therefore regain execution promptly.

## 4. Containers confine device execution

A Binding shim does not invoke CUDA from an arbitrary coroutine. It submits work through the Container/Brain path:

```text
coroutine submits Operate
  -> Brain selects an operation/batch
  -> Container queues a step
  -> event loop invokes blocking_scheduler.run_once()
  -> run_context installs stream/mode
  -> kernel launch
  -> completion resolves the future
```

A Container binds:

- a controlled execution path;
- an operation to a step;
- the step to a CUDA stream and Torch mode.

`run_context` installs the required stream and mode before launch. Context changes therefore occur at the device-execution boundary instead of relying on arbitrary coroutine interleaving.

## 5. Backend coroutine versus blocking scheduler

Both serialize CUDA-related work, but their responsibilities differ:

- Brain/backend decides **what to do**: select operations, form batches, simulate memory, and choose downgrade or swap.
- The blocking scheduler decides **when to launch**: advance one step only during a device-work slot granted by the event loop.

Both progress on the controlled loop thread, avoiding multiple coroutines locking and mutating device state and the memory model concurrently.

## 6. Two CUDA completion modes

### Blocking wait

Wait with a call such as `cudaEventSynchronize`. It does not poll, but blocks its caller and therefore must run where it cannot freeze Python scheduling.

### Poll wait

Repeatedly query with a call such as `cudaEventQuery`. It can reduce notification latency but consumes CPU; which thread or goroutine performs the polling is critical.

Waiting mode and event-loop backend are separate dimensions. “Busy wait on or off” does not fully describe every combination.

## 7. Why Python and Go backends differ

The Go backend uses goroutines and GMP scheduling for blocking C calls. When a goroutine enters blocking cgo work, the runtime can continue executing other goroutines. The Python thread releases the GIL while waiting for Go selector events; completed events return through a channel and settle futures after the GIL is reacquired.

This does not make GPU kernels automatically parallel. It prevents one blocking C call from freezing the complete Python event loop.

Python fallback behavior depends on its blocking/poll combination. In particular, polling on the main thread can hold the GIL, so different backends and waiting modes are not behaviorally identical.

## 8. Busy polling is an optional low-latency mode

Runtime configuration has two dimensions:

```text
implementation: py / chimera / golang
busy_wait:       true / false
```

With busy wait enabled, the selector uses non-blocking queries and advances poll delegates. With it disabled, idle waiting may block in the selector or move to a backend goroutine. A deployment can exchange CPU utilization for lower completion-notification latency or disable polling to reduce idle cost.

A comparison of `golang + busy` with `py + non-busy` changes several variables simultaneously and cannot attribute the entire difference to busy polling.

## 9. Do not conflate three kinds of queues

DROFELL has at least three adjacent queueing layers:

| Layer | Contents | Responsibility |
|---|---|---|
| Event loop | ready work and the blocking scheduler | when ordinary callbacks and device work receive execution |
| Brain mailboxes | prefill, generating, training, swap, and other semantic operations | what to execute and which operations may merge |
| CUDA runtime/streams | launched device work | dependencies and completion on the GPU |

Dual-class asynchronous scheduling primarily refers to the first layer. Brain mailboxes are the second, and CUDA streams are the third. Treating them as one queue hides the actual synchronization boundaries.

## 10. Seven operational invariants

When reading or modifying this path, preserve:

1. Ordinary non-blocking work does not execute managed CUDA operations directly.
2. The Brain is the unified arbiter of device operations and memory state.
3. Every device step explicitly carries its required stream and mode.
4. Device launch occurs only at blocking-scheduler advancement points.
5. Waiting for CUDA completion must not freeze all network and control coroutines.
6. Modeled-memory commits correspond to real-operation lifecycles.
7. Future completion clearly distinguishes “submitted,” “launched,” and “device complete.”

## 11. Reassemble the complete framework

Starting from a fixed chain, the causal sequence is now visible:

```text
fragile fixed physical chain
  -> layered reconfigurable multi-path graph
  -> path competition and replacement create duplicates, reorder, and late arrivals
  -> STM / canon / Promise define path-independent progress
  -> Binding separates logical roles from physical holders
  -> concurrent realizations + Shadow rebinding implement failure resilience
  -> Backbone creates layered recovery domains
  -> many tasks sharing one holder require shims / mailboxes / Brain
  -> dynamic GPU memory requires a rollback-capable Memory Digital Twin
  -> CUDA's unusual behavior requires Containers and dual-class scheduling
```

The same sequence identifies the boundary of the current system: state semantics and reactive resilience form a complete loop, while continuous path optimization and general online topology policy remain a separate layer.

## Source-reading routes

### Inter-node and recovery

- `node/infer/infer_ctrl.py`
- `node/train/train_ctrl.py`
- `node/dealer_node/`
- `node/compute_node/`
- `node/shadow_node_state/`
- `node/backbone/`

### Intra-node and memory

- `dealers/common/provider.py`
- `dealers/common/reorder.py`
- `dealers/common/brain.py`
- `dealers/memory_schedule/brain.py`
- `dealers/memory_schedule/restriction.py`
- `dealers/memory_schedule/swapper_core.py`
- `operators/core/memory/env.py`

### Asynchronous runtime

- `extend_asyncio/std_asyncio.py`
- `extend_asyncio/chimera_asyncio.py`
- `extend_asyncio/lite_asyncio.py`
- `extend_asyncio/golib/`
- `operators/container/base_container.py`

## Further reading

- [DROFELL Architecture Overview](../architecture-overview.md)
- [Intra-Node Execution and GPU-Memory Governance](../intra-node-execution-and-memory.md)
- [Runtime Platforms and Experimental Entry Points](../runtime-platform-and-experiments.md)
