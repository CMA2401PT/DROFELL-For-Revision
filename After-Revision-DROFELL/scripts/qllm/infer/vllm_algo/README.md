# Modified vLLM-Style Inference Algorithm

This package supplies request, block, scheduling, sampling, and execution logic to `infer_testbed.py`. It is an internal implementation under test, not a standalone command.

```text
InferTestBed
  -> VllmInferManager (request lifecycle / text postprocessing)
  -> VllmScheduler (prefill/decode selection and preemption)
  -> VirtualBlockSpaceIO (logical to virtual-physical blocks)
  -> LocalMachineInferEngine (head/decoder/tail/cache engine)
  -> RealWorldSampler (greedy/random/penalty/logprobs)
```

- `interface/` defines data structures, abstract interfaces, and sampling parameters.
- `implements/` provides the current local implementation.

Validate the package through the parent `infer_testbed.py`, kernel benchmarks, and prefill benchmarks. There is no standalone unit test or compatibility guarantee with current upstream vLLM; the code derives from an older version and has been adapted to DROFELL data structures.
