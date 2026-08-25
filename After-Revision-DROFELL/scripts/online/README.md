# Online Orbit-Distributed Experiments

This layer uses Orbit to launch Aux, Dealers, Backbones, Agents, and Request/benchmark nodes across coroutines, processes, or hosts. It is the repository's real distributed testing layer.

## Directories and shared modules

- `models.py`: build the model library from `split_qllm_models` and extra information published through Aux;
- `common_nodes.py`: shared Aux/Dealer/Backbone launch functions;
- `runtime_config.py`: Dealer decoder-backend and CUDA-graph configuration;
- `base/`: source-configured topologies for correctness and quick performance checks;
- `speed/`: early local multi-process throughput/trace experiment;
- `benchmark/`: staged benchmark and analysis workflow.

## Deployment semantics

`process`/`thread` in `Location` define local isolation, while `device_str` selects a CUDA device. They are different concepts. Several logical Dealers may share one process or GPU. A topology record should include at least logical nodes, Orbit process, host, CUDA device, model segments, replicas, memory limits, and network conditions.

Every entry point requires the repository root on `PYTHONPATH`, working Orbit/`extend_asyncio`, a free port, and prepared GPUs. Long-running setup processes need a separate terminal; stop them explicitly and verify their children have exited before another run.

Prefer `benchmark/` for reproducible runs. `base/` and `speed/` contain many hard-coded parameters and are intended for debugging.

## Decoder backend and CUDA graphs

Online and offline modes share the same `Brain`, `InferNode`, Marlin decoder, PEFT, and decode-graph implementations. An online Dealer reads these settings before loading or profiling any decoder:

| Environment variable | Default | Meaning |
|---|---:|---|
| `DECODER_BACKEND` | `marlin` | `marlin`, `exllama`, or `exllama_fuse` |
| `ENABLE_CUDA_GRAPH` | `0` | enable decode graphs |
| `CUDA_GRAPH_MAX_CONTEXT_LEN` | `256` | maximum context supported by the static graph block table |
| `MAX_CONCURRENCY` | `8` | maximum requests merged by one Brain; graphs support batch buckets `1..N` |

Graphs currently require Marlin; selecting ExLlama with `ENABLE_CUDA_GRAPH=1` fails before model loading. Decode exceeding concurrency, context, or PEFT conditions falls back to eager. Every graph for batches `1..MAX_CONCURRENCY` is captured explicitly during model loading; request paths replay only.

Graphs are captured separately at head, decoder-Binding, and tail operation boundaries. The default 7B model has four eight-layer decoder Bindings, so every batch has four decoder graphs plus head and tail graphs. Hidden state crossing Dealers still travels through online communication; there is no cross-node CUDA graph. Tail graphs cover the greedy/no-logprobs fast path only.

P-Tuning uses dynamic KV block tables and can replay graphs. LoRA inference and training support a Marlin base decoder, but LoRA decode falls back to eager with reason `lora`.

Set these variables in the Dealer/setup process. If setup and benchmark use separate terminals, changing `ENABLE_CUDA_GRAPH` only in the benchmark terminal does not alter already-loaded Dealers.

Automatic Dealer-placement profile cache keys include decoder backend. Marlin does not reuse an old ExLlama GPU-memory/TPS profile; the first backend switch regenerates it.
