# Online Fixed-Topology Correctness Checks

These scripts construct Orbit, Aux, Dealers/Backbones, and an Agent in source for quick validation of distributed inference/training, backup paths, self-healing, and multi-GPU 70B execution.

## Entry points

| File | Scenario |
|---|---|
| `single_point_infer.py` | 7B inference with a primary Dealer and optional backup/self-healing path |
| `single_point_train.py` | 7B training with a primary Dealer and backup |
| `single_point_infer_70B.py` | 70B two-Dealer/two-GPU inference |
| `single_point_train_70B.py` | 70B two-Dealer/two-GPU training |
| `backbone_infer.py` | 7B inference through Backbone routing and multiple Dealers/backups |
| `backbone_train.py` | 7B training through Backbone routing |
| `get_user_model.py` | helper for loading/converting P-Tuning or LoRA data |
| `gen_train_aux_data.py` | helper for constructing iterators, optimizers, and training state |

## Running

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" \
  scripts/online/base/single_point_infer.py
```

The 7B inference entry point accepts `ADAPTER=base|ptuning|lora`, default `ptuning`. Dealers default to Marlin; select another backend with `DECODER_BACKEND` and enable decode graphs with `ENABLE_CUDA_GRAPH=1`. Graph batches cover `1..MAX_CONCURRENCY`; LoRA falls back to eager, while P-Tuning can use graphs.

Inference recovery with a complete KV checkpoint defaults to `RECOVERY_REPLAY=chunk`: shadow state coalesces confirmed post-checkpoint decode input into one eager paged-decode operation. Set `RECOVERY_REPLAY=sequential` for per-task replay A/B. An online failure may include one in-flight task not yet sampled; the chunk covers only confirmed output prefix and leaves that task as a separate decode. Original token IDs remain with hidden-state carriers across nodes, allowing Agent and Backbone decoder-only shards to identify confirmed prefixes safely. Recovery without KV uses merged-prefill fallback.

Set `DROFELL_METRICS_PATH=/tmp/events.jsonl` to record `replay_task_count`, `replay_input_count`, replacement lookup, replay send, and first ordinary output. In one dual-RTX-3090 P-Tuning failure run:

| Entry point | Sequential | Chunk | Replay-send to first ordinary output |
|---|---:|---:|---:|
| `single_point_infer.py` | 15 tasks / 15 inputs | 2 tasks / 15 inputs | 218.22 ms -> 153.99 ms (-29.4%) |
| `backbone_infer.py` | 13 tasks / 13 inputs | 2 tasks / 13 inputs | 132.85 ms -> 40.97 ms (-69.2%) |

Both A/B pairs produced identical final text and released resources normally. The default `backbone_infer.py` topology escalates failure to the Agent when local Backbone replacement is unavailable, so the second timing measures hierarchical fallback through a Backbone, not local Backbone replacement. Direct assertions cover the decoder-carrier/no-tail merge branch. These single-machine samples indicate optimization direction only and are not cross-machine performance conclusions.

There is no unified CLI. Inspect `Location(process=...)`, `device_str`, model grouping, memory configuration, backup count, network model, user model, and token/batch length before each run. A 7B script may still use two GPUs; do not infer hardware requirements from the file name.

Scripts normally launch Orbit and call `run_forever()`. Agent completion may call `exit()` or terminate the process. After interruption, check port `7241` and child processes before starting another run.

Acceptance focuses on output tokens/loss, path selection, continued execution after failure, and resource release. These entry points do not have a stable JSON schema and are not suitable for cross-machine performance comparisons.
