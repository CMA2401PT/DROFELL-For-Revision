# Throughput Experiment: 2x RTX 3090

This configuration uses two GPUs. Controlled networking requires sudo, network namespaces, and `NET_ADMIN`. Use the same uclamp configuration for all top-level processes.

## Inference matrix

- model: 7B or 70B, each with an independent session;
- clients: 1, 3, 8, 16;
- output-length classes: 128 and 2,048;
- repeats per cell: 0, 1, 2;
- one serving session per model/network configuration.

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export SESSION_ID="throughput-3090-70b-infer"

sudo -E "$DROFELL_PY" exps/throughoutput/run_llm_serving.py start \
  --session-id "$SESSION_ID" --python "$DROFELL_PY" \
  --model-size 70b --workload inference --cards 2 \
  --deployment combined-request-agent \
  --bandwidth-mbit 1000 --target-rtt-ms 0
```

The benchmark loop should call `run_benchmark.py inference` and use a unique case directory for every client count, length, and repeat.

## Fine-tuning matrix

- clients: 1, 3, 8, 16;
- `(minibatch, effective batch)`: `(1,1)`, `(4,16)`, `(4,64)`;
- `samples-per-instance=128`, `sample-len=128`;
- repeat 0 per cell;
- a separate fine-tuning session.

OOM caused by insufficient capacity is a failed attempt and must not be deleted or replaced. See [`README.md`](README.md) for complete command structure.
