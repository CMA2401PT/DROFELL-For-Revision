# DROFELL

DROFELL is a distributed large-language-model runtime for loosely coupled edge and Internet environments. It supports quantized-model inference and parameter-efficient fine-tuning while coordinating dynamic inter-node paths, state recovery, and task migration together with intra-node batching, asynchronous execution, and GPU-memory governance.

If you are new to the system, start with [`docs/walkthrough/README.md`](docs/walkthrough/README.md). The complete implementation documentation begins at [`docs/README.md`](docs/README.md), runnable entry points are indexed in [`scripts/README.md`](scripts/README.md), and each directory under `exps/` documents an experimental protocol.

## Core design

- **State-driven execution:** steps, versions, `Navigate`, admission controllers, and recovery state describe logical progress without permanently binding it to one physical node.
- **Inter-node resilience:** multi-path dispatch, control-side shadow state, local recovery, task replay, and self-healing inside a Backbone.
- **Intra-node execution governance:** Bindings, operation restructuring, batch merging, and a unified backend coordinate model objects and device execution.
- **GPU-memory digital twin:** candidate operations are first simulated in a rollback-capable object-level memory model and admitted through a hierarchical constraint tree.
- **Dual-class asynchronous scheduling:** ordinary asynchronous work is advanced separately from potentially blocking CUDA/container work.

## System boundaries

- A Shadow is not a reserved standalone compute node. It is recovery material maintained on the control side of a remote execution handle.
- Multi-path redundancy and post-failure state reconstruction are cooperating but distinct mechanisms.
- The default Orbit backend uses a central service for deployment control, service discovery, and connection setup. That service does not perform model computation, resource scheduling, or recovery.
- The default networking backend does not provide consensus, network-partition arbitration, general NAT traversal, or shared-bottleneck congestion control.
- Performance, recovery latency, energy use, and numerical consistency depend on the hardware, model, topology, and configuration and must be measured with the corresponding experimental entry points.

## Repository layout

| Directory | Purpose |
|---|---|
| `cuda_extends/` | Modified and extended CUDA operators |
| `dataset/` | Dataset adapters that convert samples into `ProcessedRecord` objects |
| `dealers/` | Bindings, operation queues, memory scheduling, and the real backend |
| `docs/` | Architecture, execution-path, recovery, and runtime documentation |
| `exps/` | Reproducible experimental protocols, runners, and analysis tools |
| `extend_asyncio/` | Asynchronous runtime that schedules ordinary and blocking work separately |
| `node/` | Agent, Request, Dealer, Compute, Aux, Backbone, and related roles |
| `operators/` | Model components, KV cache, samplers, quantized-operator wrappers, and complete model assembly |
| `orbit/` | Default deployment, discovery, communication, and runtime-information backend |
| `scripts/` | Model conversion, component checks, offline/online entry points, and benchmarks |
| `simulate/` | Independent simulation and probe tools |
| `split_qllm_models/` | Default lookup location for sharded foundation models |
| `utils/` | Shared utilities |

## Runtime environment

The project primarily targets Linux systems with NVIDIA GPUs and CUDA.

The following stack has been validated end to end. It is a reference configuration, not a strict minimum or the only compatible stack. Other versions may work, but extensions must be rebuilt and verified against the applicable PyTorch/CUDA ABI and GPU architecture.

| Component | Validated version |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| Local CUDA Toolkit / `nvcc` | 13.0 |
| cuDNN | 9.19.0 |
| FlashAttention | 2.8.3.post1 |

Add the repository root to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

### Build the CUDA extensions

```bash
cd cuda_extends
python setup.py build_ext --inplace

cd ../operators/helpers/cuda_async
python setup.py build_ext --inplace
```

The operators use `float16`, `bfloat16`, and recent CUDA capabilities. You may need to adjust the `-gencode` options in `setup.py` for a different GPU architecture.

### Build the asynchronous-runtime extension

The Go backend in `extend_asyncio` moves potentially blocking C/CUDA calls onto a separate execution path. Build it with:

```bash
cd extend_asyncio/golib
bash build.sh
```

The build script first compiles the Go core and then builds the Python extension. The current development environment uses Go 1.23.x; consult `go.mod` and your local toolchain when using another version.

## Prepare a sharded model

The runtime loads model shards by decoder group, so each node does not need to store a complete model. Conversion tools include:

```text
scripts/converts/model.py
operators/helpers/scripts/model_convert.py
```

A converted directory normally looks like:

```text
split_qllm_models/teared-Llama-2-7B-GPTQ/config.json
```

Provide model and dataset locations through local deployment configuration. Do not commit machine-specific absolute paths.

## Execution levels

### Component and model checks

- `scripts/base/`: basic inference and fine-tuning correctness checks.
- `scripts/module/`: QuantLinear, decoder-strategy, and memory-behavior checks.
- `scripts/qllm/`: complete quantized-model inference, KV-cache, and training checks.

### Offline

`scripts/offline/` runs the complete control and execution path in one process with multiple coroutines. It is suitable for debugging Bindings, task ordering, the Memory Digital Twin, recovery replay, and backend behavior.

### Online

`scripts/online/` uses Orbit to run multi-process or multi-host topologies. The current benchmark workflow has three stages:

1. `generate_network.py` creates a logical network specification.
2. `launch_network.py` places Dealers on concrete hosts, processes, threads, and devices.
3. `benchmark_infer.py` or `benchmark_train.py` attaches a temporary Request node and executes the workload.

Logical topology and physical placement are separate. The deployer must configure Linux namespaces, `tc qdisc`, or other network shaping outside the benchmark entry points.

See [`scripts/online/benchmark/README.md`](scripts/online/benchmark/README.md) for complete commands and parameters.

## Documentation reading order

1. [`docs/walkthrough/README.md`](docs/walkthrough/README.md): derive the framework step by step from the physical problem.
2. [`docs/architecture-overview.md`](docs/architecture-overview.md): system roles and the two control axes.
3. [`docs/concept-to-code-map.md`](docs/concept-to-code-map.md): mapping from core concepts to implementation objects.
4. [`docs/inter-node-control-and-recovery.md`](docs/inter-node-control-and-recovery.md): inter-node forwarding and recovery.
5. [`docs/intra-node-execution-and-memory.md`](docs/intra-node-execution-and-memory.md): Bindings, scheduling, and GPU-memory governance.
6. [`docs/runtime-platform-and-experiments.md`](docs/runtime-platform-and-experiments.md): build instructions, runtime platforms, and experimental entry points.

## Third-party sources

This repository contains or modifies implementations from:

- [vLLM](https://github.com/vllm-project/vllm), Apache-2.0;
- [ExLlama](https://github.com/turboderp/exllama), MIT;
- [ExLlamaV2](https://github.com/turboderp-org/exllamav2), MIT.

The main affected areas are `cuda_extends/`, `scripts/qllm/infer/vllm_algo/`, and `operators/core/sampler/`. Some ExLlama-related operators include added backward paths for training. Before using, distributing, or modifying these components, also review the copyright headers and upstream licenses in the relevant files.

## License

Project-authored DROFELL code is distributed under the Apache License 2.0 in the root [`LICENSE`](LICENSE), unless a file or directory states otherwise. Modified third-party components remain subject to their original licenses and attribution requirements. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the license files in the corresponding subdirectories.

## Known engineering limitations

- Environment setup still requires validation against the target hardware and CUDA version.
- Some early diagnostic scripts require explicit model paths or additional runtime parameters.
- A unified end-to-end test does not yet cover every supported path.
- Exact upstream revisions and per-file modification records still need further consolidation.

When reporting an issue, include the commit, Python/PyTorch/CUDA versions, GPU model, command line, and a minimal log.
