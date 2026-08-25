# Offline Single-Process System Experiments

`offline` runs model segments, providers/Dealers, Bindings, shadow state, the memory twin, and scheduling as multiple coroutines in one Python process. It creates no real Orbit network, making it suitable for functional debugging and intra-node performance analysis; its throughput is not representative of multi-process or multi-host deployment.

See [`docs/offline-design-and-inference-path.md`](../../docs/offline-design-and-inference-path.md) for the complete object graph, initialization sequence, and prefill/decode paths.

| Directory | Purpose |
|---|---|
| `testbed/` | shared offline inference/training instances and pipeline-parallel test foundation |
| `basic/` | correctness, recovery, 7B/70B, and concurrent inference/training cases |
| `benchmark-basic/` | old quick-throughput scripts; some imports are currently broken |
| `benchmark/` | configurable inference/training benchmark |
| `memory_modeling/` | capture real memory operations and compare them with the memory-twin model |

Legacy scripts use `from testbed import ...`. When running `basic/`, include both the repository root and `scripts/offline` in `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PWD/scripts/offline"
```
