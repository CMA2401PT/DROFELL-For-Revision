# Unreliable-Node Experiment

## Purpose

Under application-level compute-node failure injection, measure inference throughput, recovery events, and multi-path behavior. Injection occurs in the Dealer task path and does not require namespaces or `tc`.

## Failure-probability semantics

`p` is the Bernoulli probability applied when a failure-enabled Dealer first receives a non-duplicate measured step. Injection excludes:

- path-probe tasks;
- redundant duplicate arrivals for the same `(seq_id, navi.current, step)`;
- recovery replay with `trace_invalid=True`.

`p` is therefore not the probability that a whole machine fails in a fixed time window.

## Matrix

List topologies and probabilities explicitly, for example:

```text
topologies: 1_1_1 1_2_1 1_4_1 2_2_1 2_4_1 2_1_2 2_2_2 2_4_2
probabilities: 0 1e-4 1e-3 1e-2 5e-2
```

`combined-request-agent` may run all probabilities. Include a persistent Agent in comparisons only when its failure-delay and injection semantics are identical.

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export RUN_ID="${RUN_ID:-unreliable-node}"

/usr/bin/uclampset -m 1024 -- "$DROFELL_PY" \
  exps/unreliable_node/run.py \
  --python "$DROFELL_PY" --run-id "$RUN_ID" \
  --topologies 1_1_1 1_2_1 1_4_1 2_2_1 2_4_1 2_1_2 2_2_2 2_4_2 \
  --probabilities 0 1e-4 1e-3 1e-2 5e-2 \
  --repeats 10
```

Use `run_cell.py` and a new run ID to retry one cell. Use `run_isolated.py` for isolated repeats. Do not replace an original failed sample with a later successful request.

## Switches

- `--capture-metrics`: enable recovery/Shadow instrumentation; do not use it for throughput measurement.
- `--skip-smoke`: use only after the same topology passed smoke.
- `RECOVERY_REPLAY`: recovery-replay mode; changing it requires a new run ID.

## Acceptance

- Every cell reaches the required repeat count and every trace has complete token times.
- Graphs really replay and fallback is recorded.
- `p=0` has zero injections.
- For `p>0`, verify injection, disconnect, handoff, and physical-recovery events.
- Failures target the actual active source.
- Cancelled/unmatched events are not counted as recovery completion.
- Basic single-path and multi-path cases use their own continuity criteria.
- Cross-run consolidation uses an explicit manifest and never auto-selects successful samples.

Environment metadata, logs, network specifications, per-repeat output, and summaries are local generated files and are not version-controlled.
