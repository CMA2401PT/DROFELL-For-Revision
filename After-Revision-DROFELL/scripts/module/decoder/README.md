# Decoder Strategy and GPU-Memory Experiments

`test.py` measures forward, swap-out/in, and backward time for one backend and one checkpoint/replay strategy and stores a PyTorch CUDA allocator snapshot. Run only one case per process to prevent memory history, allocator state, and backend workspaces from crossing configurations.

Strategy names map to parameters as follows:

- `each` -> `each_decoder`;
- `2pass` -> `first.2pass`;
- `ckpt` -> `first.grad`;
- `grad` -> `grad`.

Run one short case from the repository root:

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
  "${DROFELL_PY:-python}" scripts/module/decoder/test.py \
  --output /tmp/drofell-decoder-each.pkl \
  --backend exllama_fuse --strategy each_decoder \
  --layers 8 --batch-size 4 --seq-len 512 --prompt-seq-len 256 \
  --warmup 2 --repeats 10 --swap --no-reuse-reconstruct-cache
```

Output contains a complete `record.pkl` and a `record.json` parameter/timing sidecar at the same path. The complete protocol and stable analysis entry point are under `exps/time_memory_trade/`.

`plot.py` provides shared plotting functions for current and historical `record.pkl` files. The memory-lifecycle x-axis is CUDA allocator allocation/free event order (snapshots do not contain per-event timestamps); stage labels show independently measured wall-clock means. `exps/time_memory_trade/analyze.py` generates these figures in batch. `analysis_all.py` remains a notebook-style script for historical fixed directories only.
