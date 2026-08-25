# Unreliable-Node Experiment: 2x RTX 3090

This configuration does not use `tc`; sudo/`NET_ADMIN` is not required. At most two GPUs are visible.

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export RUN_ID="${RUN_ID:-unreliable-node-3090}"

/usr/bin/uclampset -m 1024 -- "$DROFELL_PY" \
  exps/unreliable_node/run.py \
  --python "$DROFELL_PY" --run-id "$RUN_ID" \
  --topologies 1_1_1 1_2_1 1_4_1 2_2_1 2_4_1 2_1_2 2_2_2 2_4_2 \
  --probabilities 0 1e-4 1e-3 1e-2 5e-2 \
  --repeats 10
```

Before running, ensure topology placement does not reference an invisible GPU. Retain original logs and attempts for failed cells. Rerun with a new run ID or the single-cell runner and never splice repeats across commits.
