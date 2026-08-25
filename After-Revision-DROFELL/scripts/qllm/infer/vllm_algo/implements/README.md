# Inference Algorithm Implementations

This directory has no standalone entry point. Primary implementations are:

- `infer_manager.py`: request creation/abort, sequence-state advancement, token decoding, stop conditions, and output construction;
- `scheduler.py`: waiting/running/swapped queues, prefill/decode batching, preemption, and block operations;
- `virtual_block_space_io.py` / `virtual_block_space_holder.py`: block allocation, append, swap, fork, and free;
- `infer_engine.py`: translate scheduler metadata into local head/decoder/tail and cache-engine calls;
- `sampler.py`: temperature, top-k/top-p/min-p, presence/frequency/repetition penalties, greedy/random sampling, and logprobs;
- `policy.py`: FCFS ordering.

Performance changes require correctness and state-release checks. TPS alone will not reveal block leaks, invalid state after preemption, or stop/logprob regressions. The repository has no independent unit tests here; run small parent-testbed functional requests before long-token benchmarks.
