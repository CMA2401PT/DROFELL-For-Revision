# Unreliable-Node Protocol Evolution

This file records runner-semantics changes only and retains no historical data, measurements, or run conclusions.

## Retained design decisions

1. Inject failures only into measured benchmark sequences, not warmup or path probes.
2. Apply one Bernoulli draw to redundant duplicates of the same logical step.
3. Do not inject failures again during recovery replay.
4. `p` is the conditional probability when a Dealer receives a new measured step, not a machine time-window failure rate.
5. Multi-path topologies expose work to more failure-enabled providers; read actual exposure count from logs.
6. Combined and persistent Request-Agent deployments are comparable only when runtime switches, failure delay, and placement are identical.
7. Instrumentation JSONL I/O perturbs the hot path and is enabled only for recovery/Shadow diagnosis.
8. Validate graph capture, fallback, token ledger, and recovery events for every repeat.
9. After interruption of a parent matrix, summarize complete cells only; cross-run composition requires an explicit manifest.
10. Retain failed attempts and do not overwrite or replace them with later success.

## Topology names

`A_B_C` denotes the node/shard/path combination defined by the runner. Use the current topology table in `run.py` and generated network specification for expansion; do not infer physical placement from the string alone.

## Compatibility

Older runs may use different Request-Agent placement, recovery delay, graph configuration, token budget, or metric schema. Compare manifests before analysis. Keep incompatible output in separate local archives.
