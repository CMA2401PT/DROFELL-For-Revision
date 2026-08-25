# DROFELL

This is the code repository accompanying the DROFELL manuscript. It includes:

- the core system implementation;
- experiment and testing code;
- data analysis and result visualization tools;
- experimental scripts used in the manuscript;
- as well as a large number of supplementary experiments, ablation studies, and visualization tools that are not shown in the manuscript.

DROFELL is a framework for LLM inference and fine-tuning in loosely coupled distributed edge / Internet environments. 
It shares a similar end goal with [PETALS](https://github.com/bigscience-workshop/petals), but focuses on a different set of system challenges.  
DROFELL mainly targets the following core problems:

- **State-centric execution semantics**: maintaining execution continuity based on model state and state-transition constraints (rather than binding progress to fixed nodes), so that inference and fine-tuning can survive node churn, migration, and online repartitioning;

- **Resilient inter-node dynamic resource management**: preserving a valid execution frontier under failures and resource fluctuations through **multi-path redundant execution**, **shadow-assisted recovery**, and **self-healing nodes**, instead of relying on a single static path or stop-the-world failover;

- **Intra-node execution-time resource governance**: restructuring execution around fine-grained operations, **batch merging**, dual-queue scheduling, and predictive memory-safe admission, so that device time, VRAM, and migratable state objects can be coordinated at runtime without sacrificing utilization;

- **Collaborative distributed inference and fine-tuning**: supporting cross-node LLM inference, fine-tuning, experimental validation, and performance analysis in loosely coupled edge / Internet environments.

The repository has Typing support and fully relies on a modified `asyncio` implementation (see extend_asyncio) for asynchronous execution.

This repository contains not only the main implementation used in the manuscript, but also auxiliary tools for reproducing experiments, extending analysis, and supporting deeper research.

## Not Yet Public

The following content is not public yet and will be released after the manuscript is accepted:

- the full Git commit history listed in `orig_git_history.txt`.

## Referenced and Modified Upstream Repositories

Some parts of the code reference or modify other repositories. The main ones are listed below.

### vLLM

- Repository: <https://github.com/vllm-project/vllm>
- License: Apache-2.0

Notes:

- We used an earlier version of the vLLM repository, roughly from 2024.
- Under `cuda_extends`, we referenced, modified, and extended vLLM CUDA operator implementations.
- Under `scripts/qllm/infer/vllm_algo`, we referenced and modified the core logic of the vLLM scheduler.
- Under `operators/core/sampler`, we referenced and modified part of the vLLM code.

### ExLlama V1 / V2

- Repositories:
  - <https://github.com/turboderp/exllama>
  - <https://github.com/turboderp-org/exllamav2>
- License: MIT

Notes:

- We used some 2024-era version of these two repositories and integrated them together, effectively combining V1, V2, and vLLM.
- Under `cuda_extends`, we referenced, modified, and extended ExLlama V1 / V2 operator implementations.

## Repository Overview

The introduction below may still be incomplete, and we will continue improving it over time.  
If you run into any issues, feel free to open an issue.

## Directory Structure

### `cuda_extends`

Contains all CUDA operators used in the project. Most of them originate from vLLM and ExLlama V1 / V2, and we integrated them so that they can support both fine-tuning and inference.

We modified the vLLM CUDA operators to support dynamic memory allocation and release, and also changed their prefill-stage behavior to make them compatible with the rest of the codebase. Our main modifications to ExLlama were made to support backward propagation, so that fine-tuning algorithms can run on top of it. We made many additional changes for other purposes as well, but listing them all here would be impractical.

This directory also contains some hack operators implemented for auxiliary purposes.

### `data`

Specifies the locations of various data files, such as user models and datasets, so that paths can follow Python import-style resolution and avoid path lookup issues.

### `split_qllm_models`

Specifies the locations of split fundation models, again so that paths can follow Python import-style resolution and avoid path lookup issues.

### `dataset`

Dataset adapters. The final goal is to construct generators of `ProcessedRecord`. Two dataset adapters are currently provided:

- `advertise_gen` (the sample dataset used in the GLM repository);
- `person_chat` (the sample dataset used in PETALS).

### `dealers`

The main middle-end and back-end implementation. Key concepts such as STM, memory digital twins, binding, and scheduling can be found here, although the naming differs from the manuscript.

### `extend_asyncio`

A reworked and extended `asyncio` implementation. Its goal is to mimic Python `asyncio` while also supporting the dual-queue model described in the manuscript and coroutine support that eliminates blocking caused by CUDA-related calls.

It requires a core library compiled with Golang, because we rely on Golang's GMP mechanism to automatically migrate blocking C functions to other threads.

### `node`

Implementations of different roles, such as:

- compute nodes;
- shadow nodes;
- self-healing nodes;
- user nodes;
- auxiliary nodes.

### `operators`

The core Python library for LLM inference and fine-tuning, built on top of the dual-queue asynchronous model in `extend_asyncio` and the operators in `cuda_extends`.

In other words, this directory progressively assembles the CUDA operators from `cuda_extends` into LLM layers, core concepts such as samplers, modules such as KV cache management, and full models for use by the rest of the system.

### `orbit`

`orbit` is intended to provide both an experimental platform and an information collection hub. It allows a set of coroutines to run across multiple hosts, processes, threads, and coroutines, collects their runtime information, and controls whether specific coroutines run on different hosts, different processes on the same host, different threads in the same process, or the same thread.

It provides a centralized inter-node communication mechanism and can be further extended through `link` into a P2P communication mode between nodes.

Note that although it has built-in simple bandwidth and latency simulation, that functionality is suitable only for rapid prototyping, because it does not simulate contention or queueing effects. For serious testing, you should still use Linux tools such as `ip netns` and `tc qdisc` to emulate bandwidth and latency. The benchmarks reported in the manuscript were also measured using those tools.

### `scripts`

Except for `scripts/converts`, which contains helper tools for splitting models, most of the rest of this directory consists of code for testing, analysis, and result visualization at different levels of the system. These scripts are introduced in more detail in the "Running / Testing / Analysis / Visualization" section below.

## Preparation

Although this repository includes all necessary code, setup is still fairly complex. If you run into problems, feel free to open an issue.

### 1. Add Paths

To avoid Python import failures, it is recommended to add this repository to `PYTHONPATH`.

### 2. Set Up Conda

`environment.yml` reflects the current environment, but it likely contains redundant packages. In other words, you probably do not need every package listed there to run the project.

> TODO: clean up `environment.yml`

### 3. Build Operators

Because the project heavily uses half-precision types such as `float16` and `nv_bfloat16`, NVIDIA GPUs older than the 30 series may fail to compile the operators correctly, while 30-series and newer GPUs should generally work.

You will need `CUDA Toolkit 12.1` (the version we tested with) to provide tools such as `nvcc`. Then run the following under `cuda_extends`:

```bash
python setup.py build_ext --inplace
```

This will build the operators automatically.

Depending on your hardware, you may need to modify or add flags such as `-gencode` and `arch=compute_??,code=compute_??` in `setup.py`.

### 4. Build `extend_asyncio`

We rely on Golang's GMP mechanism to automatically migrate blocking C functions to other threads, which allows us to support managed CUDA asynchrony and integrate it with Python `asyncio`.

So a Golang-based core library needs to be compiled. The Golang version we use is `1.23.9`; lower versions may also work, but they may require edits to `extend_asyncio/golib/go_selector/go.mod`.

The build script is `extend_asyncio/golib/build.sh`, which performs a two-stage build:

1. first compile the core library with Golang;
2. then compile it into a Python extension with `pybind11`.

### 5. Prepare Fundation (base) Models

Since this work studies distributed deployment, requiring every node to hold a complete foundation model is clearly unreasonable. Therefore, the model needs to be split.

Using `llama2-7B` as an example, you can download it directly from:  
<https://huggingface.co/TheBloke/LLaMa-7B-GPTQ>  
Similarly, `70B` models can be downloaded in the same way.

After downloading, use one of the following scripts to split the model into multiple chunks and place them under `split_qllm_models`:

- `scripts/converts/model.py`
- `operators/helpers/scripts/model_convert.py`

If done correctly, you should end up with a structure similar to:

```text
split_qllm_models/teared-Llama-2-7B-GPTQ/config.json
```

## Running / Testing / Analysis / Visualization

Most of the related tools are under `scripts`. However, before actually using them, it is helpful to understand the different layers of this project.

The project has four main layers: `module`, `llm`, `offline`, and `online`. Each layer is built on top of the previous one:

- `module`: focuses on the behavior of an individual layer or module inside an LLM, for example the performance of quantized linear layers under different reconstruction strategies;
- `llm`: focuses on whether fine-tuning and inference remain correct after assembling those layers and modules into a full LLM model. The inference path here uses part of the vLLM logic;
- `offline`: focuses on whether concepts such as binding, shadow nodes, STM, serialization / deserialization, memory digital twins, and scheduling work correctly, while still staying in a single-node multi-coroutine setting;
- `online`: the real distributed setting, with multiple processes and multiple nodes.

Under `scripts`, we further provide the following analysis code and visualization tools.

Some of these tools have not been maintained long-term. They worked in the original version, but some may be broken or fail in the current version due to later changes elsewhere in the codebase. If you find broken parts, please let me know.

## `scripts/converts`

- `model.py`  
  Converts a model from GPTQ format into a decoder-split format.
- `linear.py`  
  Extracts all linear layers from decoder 0 of the model, including `q`, `k`, `v`, `o`, `up`, `gate`, and `down`.

## `scripts/base`

Basic correctness tests for operators:

- `infer.py`  
  Inference correctness test.
- `train.py`  
  Fine-tuning correctness test.

## `scripts/module/qlinear`

- `test.py`  
  Tests the performance of different linear implementations in both forward and backward passes under different problem sizes.
- `plot.py`  
  Visualizes the results above.

## `scripts/module/decoder`

- `test.py`  
  Tests the speed and memory behavior of different decoder strategy combinations (base decoder / user model).

Base model strategies include:

- storing the full gradient graph;
- one checkpoint per decoder;
- only checkpointing the first decoder and replaying during backpropagation. This strategy further has two variants:
  1. record the complete gradients and run backpropagation in one shot;
  2. switch to per-layer gradient recording and complete backpropagation layer by layer.

User model strategies include:

- storing the full gradient graph;
- storing only the seed and replaying during backpropagation.

The tests include:

Speed-related metrics:

- forward-pass speed;
- swap-in / swap-out speed;
- backward-pass speed.

Memory-related metrics:

- peak memory during the forward pass;
- context size left after the forward pass;
- whether memory is fully released after `swap out`;
- memory size after `swap in`;
- peak memory during the backward pass;
- whether memory is fully released after the backward pass.

- `plot.py`  
  Generates a detailed memory timeline for a single target, tracks when each memory allocation and release happens, checks whether memory usage at key time points satisfies specific constraints, and shows peak memory, key-node memory usage, and stage-wise time overhead.

- `analysis_all.py`  
  Similar to `plot.py`, but visualizes all targets at once.

## `scripts/qllm/infer`

Inference and fine-tuning support based on an algorithm similar to vLLM, but adapted to quantized linear operators.

- `kenrel_vllm.py`  
  Uses vLLM-style operators where all blocks of each layer are contiguous; intended for the 7B model.
- `kenrel_our.py`  
  Uses operators based on a pointer table; intended for the 7B model.
- `cache_kernel/compare_kernel.py`  
  Compares the inference performance difference between the two kernels above.
- `cache_kernel/analysis_one.py`  
  Analyzes one result file under different concurrency levels, including speed and memory behavior if available.
- `prefill_single.py`  
  Tests performance under different `ptuning` lengths and prompt lengths when `flags.MULTI_PREFILL_STREAM=False`, i.e. when prefills are executed serially.
- `prefill_multi.py`  
  Tests performance under different `ptuning` lengths and prompt lengths when `flags.MULTI_PREFILL_STREAM=True`, i.e. when prefills are executed in parallel on different streams.
- `prefill/analysis.py`  
  Compares the performance difference between those two prefill modes.

Nearby scripts provide the corresponding analysis and visualization utilities.

## `scripts/qllm/train`

Tests whether fine-tuning converges correctly under different settings, along with the corresponding speed and memory behavior.

- `err_eps.py`  
  Uses the standard `eps=1e-8` and `float16` fine-tuning, with the expectation of observing sudden collapse once the loss becomes flat.
- `common.py`  
  A more typical fine-tuning setup, but with `eps=1e-4`, so it does not collapse even when using `float16`.
- `bfloat16.py`  
  Uses `bfloat16` throughout fine-tuning. It is slower but stable.
- `bfloat16_start.py`  
  Starts with `bfloat16` fine-tuning, then switches to `float16`, while keeping `eps=1e-4`, so it should remain stable both initially and after convergence.
- `nan_remove.py`  
  Uses `float16` fine-tuning; if collapse occurs and weights become `nan`, those weights are reset.

Nearby scripts provide the corresponding analysis and visualization utilities.

## `scripts/offline`

From this point on, the testing code becomes much more complex and configurable, so instead of documenting every detail, we only describe the main goal of each directory.

### `scripts/offline/basic`

Tests whether inference and fine-tuning still work correctly after splitting the model and introducing binding, STM, memory twins, and the scheduler back end.

### `scripts/offline/benchmark-basic`

Runs simple speed tests for model fine-tuning / inference on top of the setup above. The analysis tool is located at:

- `scripts/offline/benchmark-basic/analysis_utils.py`

### `scripts/offline/memory_modeling`

Analyzes the memory twin model in the back end and compares modeled memory behavior against actual observations. The analysis and visualization entry point is:

- `scripts/offline/memory_modeling/simulate.py`

Note that `root_memory` in `scripts/offline/memory_modeling/simulate.py` should be changed to match device memory, or at least be consistent with the test configuration.

This part actually works by injecting an analysis framework that captures the inputs and outputs of all operations. By modifying that analysis framework, it can also be used to analyze targets beyond memory.

### `scripts/offline/benchmark`

Analyzes fine-tuning and inference performance using a `Makefile`-based test flow. The analysis utility is `analysis_utils.py`.

Note: this is **not** the source of the benchmark data reported in the manuscript. The manuscript benchmarks come from the `online` directory, not `offline`.

## `scripts/online`

Because this part involves multiple nodes, deployment is significantly more complex than the earlier sections. We first introduce three key deployment modes.

### Serious Deployment

Run the main process manually on different hosts or different namespaces.  
For namespaces, use Linux `ip` and `tc qdisc` to control bandwidth and latency between namespaces.

We use the `net_cluster` script and a `Makefile` to simplify the otherwise painful network configuration process. Both can be found in the relevant directories.

All benchmark data reported in the manuscript was collected using this mode. It is the closest to real deployment, but it is inconvenient for debugging and research iteration.

### Convenient Deployment

Still uses multiple processes, but all on a single host. For example, `scripts/online/benchmark/setup.py` can launch multiple processes at once to host nodes with different roles.

This mode cannot control network behavior the same way as the serious deployment mode, but it is still multi-process, so the results remain meaningful. It is mainly used for quick correctness testing.

### Debug Deployment

Uses a single process with multiple coroutines to host different nodes. Due to the GIL and blocking effects, the results are heavily distorted and become much slower, but everything stays within one process, so the Python debugger can directly inspect different nodes.

This mode is very convenient for prototyping and debugging.

### `scripts/online/base`

Quickly tests inference and fine-tuning correctness under different topologies and configurations in multi-node settings, including whether self-healing nodes are present.

These scripts can switch between convenient deployment and debug deployment. Specifically, through each node's `Location` configuration:

- setting the same `host`, `process`, and `thread` switches to debug deployment;
- setting different `host`, `process`, and `thread` values switches to convenient deployment and relies on `orbit`.

This directory does not contain detailed analysis tools, but it still prints rough speed and throughput information.

### `scripts/online/speed`

Runs quick throughput tests for inference. Parameters and configuration need to be adjusted manually in code. It can switch between convenient deployment and debug deployment, and it includes one analysis tool.

### `scripts/online/benchmark`

Tests inference and fine-tuning throughput and also includes analysis and visualization tools.

All nodes here can be divided into two categories:

- test nodes;
- all non-test nodes, such as compute nodes, auxiliary nodes, and other system roles.

We refer to all nodes outside the test nodes collectively as "system nodes".

Because there are many configurable options here, exposing everything as command-line arguments would be painful. So we simplify the workflow using "configuration files + environment variables".

For test nodes:

- configuration files are located under `scripts/online/benchmark/profiles`;
- they can be read by `scripts/online/benchmark/benchmark_infer.py` and `scripts/online/benchmark/benchmark_train.py`.

For system nodes:

- configuration files are located under `scripts/online/benchmark/setups`.

Each system node can either be launched directly using its corresponding configuration file plus command-line arguments (see `Makefile`) to realize serious deployment, or be launched in batch by `scripts/online/benchmark/setup.py` to realize convenient deployment.

`net_cluster` helps configure namespaces, bandwidth, and latency, but environment variables still need to be set, or the corresponding `USE_NC` option needs to be enabled in the `Makefile`.

The `Makefile` helps run different categories of tests. The analysis and visualization tools for different purposes can also be found in this directory, usually with names starting with `analysis_`.
