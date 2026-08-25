# Consolidation, Batch Merging, and Dual-Queue Experiment

## Purpose

Use a 2x2x2 factorial design to enable or disable readiness consolidation (C), intra-node batch merging (M), and dual queues (D), measuring main, conditional, and interaction effects.

## Factors

| Factor | Enabled | Disabled |
|---|---|---|
| C | wait for same-next-hop work until cap or deadline, then send as a group | send each request immediately |
| M | Brain merges operations with the same Binding | `max_concurrency=1` |
| D | Shared container and Go asynchronous path | NonAsync container and Python path |

Cases are named `c{0,1}-m{0,1}-d{0,1}`. Supporting patches are under `patches/`; inspect the working tree before applying them and ensure the runner restores them afterward.

Cross-stage graph folding bypasses intermediate mailboxes. To study mailbox behavior, use `common-instrumentation.patch` to disable fused folding while retaining standalone head/decoder/tail graphs.

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export RUN_ID="${RUN_ID:-infer-consolidation}"

"$DROFELL_PY" exps/infer-consolidation/run.py all \
  --python "$DROFELL_PY" \
  --run-id "$RUN_ID" \
  --repeats 3 \
  --max-tokens 128 \
  --intercept-window-ms 10
```

Use `--cases` for a subset and `--resume` after interruption. Window-sensitivity points should use independent run IDs and be aggregated with the arguments documented by `analyze_window.py --help`.

## Output and acceptance

Parameter snapshots, runtime metrics, traces, token timings, graph statistics, and generated analysis files remain in Git-ignored directories.

Check that:

- all eight orthogonal cases and repeat counts are complete;
- standalone head/decoder/tail graphs replay;
- fused replay and unexpected fallback match the protocol;
- M=0 configures batch 1 only and M=1 configures the target range;
- packets do not exceed the consolidation cap;
- actual batch size comes from runtime metrics, not timeline-line shape;
- effect calculations do not assume linear addition of all three mechanisms.
