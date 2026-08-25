# Throughput Experiment: Multi-GPU Environment without Network Administration

Use `--no-netns` when `NET_ADMIN` is unavailable. This mode is unshaped local loopback and must not report nominal bandwidth or RTT.

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export SESSION_ID="throughput-multigpu-loopback"

"$DROFELL_PY" exps/throughoutput/run_llm_serving.py start \
  --session-id "$SESSION_ID" --python "$DROFELL_PY" \
  --model-size 70b --workload inference --cards 8 \
  --deployment combined-request-agent \
  --bandwidth-mbit 1 --target-rtt-ms 0 --no-netns
```

Inference repeat:

```bash
"$DROFELL_PY" exps/throughoutput/run_benchmark.py inference \
  --session-id "$SESSION_ID" \
  --clients 3 --repeat 0 --max-tokens 130 --timeout 7200
```

Fine-tuning requires an independent fine-tuning session. Create a new session after changing card count, workload, or deployment. Stop it afterward with `run_llm_serving.py stop`.

If the entry point requires uclamp or sudo that the environment does not provide, adjust deployment support first or skip the run explicitly; do not alter the protocol silently.
