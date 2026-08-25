# Runtime Platforms and Experimental Entry Points

## 1. Execution levels

| Level | Directory | Purpose |
|---|---|---|
| Base | `scripts/base/` | single-machine operator and model correctness checks |
| Module | `scripts/module/` | QuantLinear, decoder strategies, and GPU-memory behavior |
| qLLM | `scripts/qllm/` | complete quantized-model inference, KV cache, and training |
| Offline | `scripts/offline/` | single-process, multi-coroutine Bindings, Brain, recovery, and memory model |
| Online | `scripts/online/` | Orbit multi-process/multi-host deployment and benchmarks |
| Experimental protocols | `exps/` | fixed variables, runners, analyzers, and acceptance rules |

`scripts` provides general entry points. `exps` packages parameters and controls for specific questions into reproducible protocols.

## 2. Environment preparation

The repository primarily targets Linux, NVIDIA GPUs, and CUDA. Record before each run:

- Git commit and working-tree differences;
- Python, PyTorch, CUDA Toolkit, and driver versions;
- GPU model, count, and visible devices;
- CPU governor, uclamp, and NUMA configuration;
- model, dataset, and shard paths;
- network namespaces, qdisc rules, and link parameters.

`environment.yml` is an environment snapshot and is not guaranteed to reproduce unchanged on every machine.

## 3. Build components

### CUDA operators

```bash
(cd cuda_extends && python setup.py build_ext --inplace)
(cd operators/helpers/cuda_async && python setup.py build_ext --inplace)
```

### Go/Python asynchronous extension

```bash
(cd extend_asyncio/golib && bash build.sh)
```

After building, run a minimal import and basic inference check so that ABI, architecture, or dynamic-library problems are found before a long experiment.

## 4. Orbit deployment model

`Location` places a node on a host, process, thread, and coroutine. Common modes are:

- **Same-process debugging:** several nodes share one Python process; useful for breakpoints, but affected by the GIL and blocking.
- **Local multi-process:** logical nodes occupy different processes, threads, or GPUs; useful for functional and local performance checks.
- **Multi-host:** host and connection information are explicit, while the deployment network supplies bandwidth, latency, and firewall behavior.

The default Orbit service handles registration, locations, connection setup, and logs. Direct data links and the central control channel are different paths.

## 5. Three-stage online benchmark workflow

### Generate a logical network

`scripts/online/benchmark/generate_network.py` creates a network specification containing decoder-group sizes and the Dealer-to-group-ID mapping. It does not contain concrete host/process/thread/device placement.

### Launch the serving network

`scripts/online/benchmark/launch_network.py` reads the network specification and supports:

- `local`: launch Orbit, Aux, and all Dealers on one machine;
- separated launch: host nodes across multiple machines or processes according to placement arguments.

### Execute requests

- `benchmark_infer.py`: runs inference from a temporary Request node;
- `benchmark_train.py`: runs training from a temporary Request node.

The Request exits after completion, while the serving network may remain for later runs.

See [`../scripts/online/benchmark/README.md`](../scripts/online/benchmark/README.md) for complete parameters.

## 6. Failure injection

A failure runner should state explicitly:

- how the target is chosen;
- failure probability or deterministic step/time;
- whether the current active winner is targeted;
- checkpoint period;
- recovery replay mode;
- path count and redundancy;
- independent random seed for each run.

A fixed node name is not necessarily the path currently producing accepted output. For multi-path recovery, confirm the target from the source recorded with accepted output.

## 7. Network shaping

Orbit's message-level simulation is useful for debugging but does not model shared-bottleneck queues. Controlled tests should use external tools such as:

```text
ip netns
tc qdisc
netem
tbf / htb
```

The runner should record actual rules, direction, bandwidth, RTT or one-way delay, and cleanup commands. Without the required privileges, identify the run as unshaped loopback rather than a controlled network.

## 8. Asynchronous backends

`extend_asyncio` supports several selector/backend and polling combinations. Distinguish:

- ordinary ready work;
- container work in `BlockingScheduler`;
- blocking and polling delegates for C/CUDA completion;
- Python and Go backends.

Busy polling can be disabled. Measure CPU utilization, context switches, latency, and throughput separately under the same workload.

## 9. GPU-memory capture/replay

`scripts/offline/memory_modeling/` captures Brain-operation inputs, modeled-pool changes, and real CUDA metrics, then replays them offline. Graph-capture lifecycles should appear as explicit trace events.

Outputs are generated at runtime and ignored by Git by default. See:

- [`memory-twin-capture-and-replay.md`](memory-twin-capture-and-replay.md)
- [`../scripts/offline/memory_modeling/README.md`](../scripts/offline/memory_modeling/README.md)

## 10. Experimental-directory convention

Each `exps/<name>/README.md` should contain only:

1. the mechanism and variables under test;
2. prerequisites and hardware constraints;
3. directly runnable relative-path commands;
4. output-file structure;
5. acceptance and failure criteria;
6. cleanup and retry procedures.

Generated JSON, CSV, logs, figures, and statistical conclusions are not version-controlled. `.gitignore` covers common result directories and extensions.

## 11. Reproducible-run checklist

- [ ] The working tree is clean or its diff is saved.
- [ ] The commit and complete argv are recorded.
- [ ] Relative repository paths or explicit environment variables are used.
- [ ] Model and dataset inputs are fixed and verifiable.
- [ ] Seeds, topology, and failure targets are fixed.
- [ ] Warmup and measurement windows are separate.
- [ ] Smoke runs do not share a result directory with full runs.
- [ ] Success, failure, and exit status are recorded for every case.
- [ ] The analyzer rejects missing, duplicate, or cross-version data.
- [ ] Figures can be regenerated from raw output.
- [ ] Results, logs, and machine metadata remain in Git-ignored directories.
- [ ] Serving processes, Orbit, and network-shaping rules are stopped afterward.

## 12. Early entry points

Some early scripts use hard-coded model paths, removed profiles, or an old Makefile. Treat each directory README and current argparse definition as authoritative. If an entry point imports a missing module, treat it as a migration candidate rather than modifying the core runtime to satisfy it.

The root `main.py` dynamically imports `targets.<target>`, but the current tree does not contain a complete `targets/` hierarchy, so it is not a recommended primary entry point.
