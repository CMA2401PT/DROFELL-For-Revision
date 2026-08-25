# Unreliable-Node Experiment: 8-GPU Environment

This configuration uses eight visible GPUs and requires no `NET_ADMIN`.

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export RUN_ID="${RUN_ID:-unreliable-node-8gpu}"

/usr/bin/uclampset -m 1024 -- "$DROFELL_PY" \
  exps/unreliable_node/run.py \
  --python "$DROFELL_PY" --run-id "$RUN_ID" \
  --topologies 1_1_1 2_4_1 2_4_2 4_4_1 8_4_2 8_8_1 8_4_4 \
  --probabilities 0 1e-4 1e-3 1e-2 5e-2 \
  --repeats 5 --include-persistent-p0
```

Use a new run ID after changing repeat count, topology, or deployment. Run persistent deployment only for `p=0` unless failure delay and injection configuration have been made equivalent to combined deployment.
