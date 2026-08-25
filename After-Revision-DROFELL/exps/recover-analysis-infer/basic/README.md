# Basic Inference Recovery Experiment

## Purpose

Measure single-path recovery by stage: failure trigger, disconnect handling, replacement-graph installation, logical handoff, physical model recovery, replay, and first valid output.

## Modes

- ordinary random or timed failure;
- deterministic node and deterministic step;
- chunk replay with a KV checkpoint;
- merged prefill plus boundary decode without a KV checkpoint;
- `sequential` replay as a diagnostic control only.

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export RUN_ID="${RUN_ID:-infer-recovery-basic}"

PYTHONPATH="$PWD" "$DROFELL_PY" \
  exps/recover-analysis-infer/basic/run.py \
  --python "$DROFELL_PY" \
  --run-id "$RUN_ID" \
  --repeats 5 \
  --max-tokens 256
```

Consult `run.py --help` for topology, failure-step, checkpoint, and replay parameters. Begin with a shorter token-count smoke under a separate run ID.

Reanalyze with:

```bash
PYTHONPATH="$PWD" "$DROFELL_PY" \
  exps/recover-analysis-infer/basic/analyze.py --help
```

## Acceptance

- Every failure targets the active source.
- Event time is monotonic and causal order is complete.
- Replacement-graph installation and send-loop commit are recorded separately.
- Normal and Recover have identical output tokens and confirmed prefix.
- Chunk replay step range/count is correct.
- A checkpoint case does not silently degrade to prefill.
- A no-checkpoint case retains the boundary decode.
- The analyzer rejects missing events, duplicate recovery indexes, and incomplete runs.

Logs, events, JSON, CSV, and figures remain in Git-ignored directories.
