# Single-Machine qLLM Experiments

This layer assembles quantized decoders, user models, KV cache, and a modified vLLM-style scheduler/manager in one process.

- `infer/`: end-to-end inference, KV-cache kernels, multi-stream prefill, and scheduling algorithms;
- `train/`: numerical stability, speed, and GPU-memory records for P-Tuning fine-tuning.

These scripts normally execute directly after module import and do not share a CLI. Every experiment requires sharded models, trained user models for some entry points, CUDA extensions, and `extend_asyncio`.
