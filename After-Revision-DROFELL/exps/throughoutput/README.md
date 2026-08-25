# DROFELL Throughput Experiment

## 1. Two-stage entry points

A serving session generates the network specification and starts Orbit and Dealers. A benchmark joins as a temporary Request node and executes one repeat. They coordinate through `sessions/<session-id>/session.json`, allowing one serving session to run several client counts, lengths, and repeats without reloading the model.

`combined-request-agent` and `persistent-agent` are different deployments and must not share one session.

## 2. Environment

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export SESSION_ID="${SESSION_ID:-drofell-throughput}"
```

Controlled namespaces require root/`NET_ADMIN`. Environments without permission must pass `--no-netns` and describe the network as unshaped local loopback.

## 3. Start serving

```bash
sudo -E "$DROFELL_PY" exps/throughoutput/run_llm_serving.py start \
  --session-id "$SESSION_ID" \
  --python "$DROFELL_PY" \
  --model-size 70b \
  --workload inference \
  --cards 2 \
  --deployment combined-request-agent \
  --bandwidth-mbit 1000 \
  --target-rtt-ms 0
```

Status and stop:

```bash
sudo -E "$DROFELL_PY" exps/throughoutput/run_llm_serving.py status \
  --session-id "$SESSION_ID"

sudo -E "$DROFELL_PY" exps/throughoutput/run_llm_serving.py stop \
  --session-id "$SESSION_ID"
```

Fine-tuning uses a separate session with `--workload finetuning`.

## 4. Inference repeat

```bash
sudo -E "$DROFELL_PY" exps/throughoutput/run_benchmark.py inference \
  --session-id "$SESSION_ID" \
  --clients 1 --repeat 0 \
  --max-tokens 130 --warmup-tokens 16 --timeout 7200
```

`--max-tokens` includes framework boundary tokens. Fix its mapping and warmup when comparing output lengths. Do not run several benchmark commands concurrently.

## 5. Fine-tuning repeat

```bash
sudo -E "$DROFELL_PY" exps/throughoutput/run_benchmark.py finetuning \
  --session-id "$SESSION_ID" \
  --clients 1 --repeat 0 \
  --minibatch-size 4 --effective-batch 16 \
  --samples-per-instance 128 --sample-len 128 --timeout 7200
```

`effective-batch / minibatch-size` must be integral; the entry point derives gradient accumulation. Multiple clients represent concurrent logical tasks.

## 6. Output and acceptance

Session state, case arguments, logs, raw records, and `complete.json` or `failure.json` are local generated files and are not version-controlled.

- Start a benchmark only after serving is ready.
- Create a new session when workload, card count, deployment, or network changes.
- Retain OOM, timeout, and missing-file cases as failed attempts.
- Use `--retry` without overwriting old attempts.
- Keep model loading and warmup outside measurement.
- Do not label unshaped loopback with nominal bandwidth or RTT.
- Clean only the target session and test node afterward, without affecting other serving sessions.
