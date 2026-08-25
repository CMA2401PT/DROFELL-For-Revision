# Online Benchmark

The online benchmark has three independent stages:

1. `generate_network.py` creates a logical serving-network JSON specification.
2. `launch_network.py` places Dealers from that JSON on concrete hosts, processes, threads, and CUDA devices and can colocate a persistent Agent with a selected Dealer.
3. `benchmark_infer.py` or `benchmark_train.py` joins as a temporary Request Satellite and exits after the workload. `benchmark_infer.py --measurement decode` provides fixed-logical-round decode measurement.

A benchmark connects to an existing serving network through `--orbit`; it does not launch that network. Benchmarks no longer import `profiles/` or `setups/` dynamically, and the old Makefile, profiles, and setups have been removed.

## 1. Logical network

`--decoder-group-sizes` specifies exact groups, not candidates. For example:

```text
--decoder-group-sizes 16 16
```

For a 32-layer model this defines group 0 as `[0, 16)` and group 1 as `[16, 32)`.

Repeat `--dealer NAME=GROUP_ID[,GROUP_ID...]` to map Dealers to groups:

```text
--dealer dealer-a=0 --dealer dealer-b=1
```

Every group needs at least one Dealer. Overlap represents logical replicas: `dealer-a=0,1` and `dealer-b=1` give group 1 two providers. The head follows group 0 automatically and the tail follows the final group.

Generate a 7B two-group specification with CUDA graphs:

```bash
python -m scripts.online.benchmark.generate_network \
  --name llama2-7b-two-groups \
  --model teared-Llama-2-7B-GPTQ \
  --decoder-group-sizes 16 16 \
  --dealer dealer-a=0 \
  --dealer dealer-b=1 \
  --decoder-backend marlin \
  --max-seq-len 256 \
  --max-concurrency 8 \
  --cuda-graph --graph-budget-mib 64 \
  --memory-policy static \
  --major-reserved-mib 4096 \
  --output scripts/online/benchmark/networks/llama2-7b-two-groups.json
```

The generated example is committed as [`networks/llama2-7b-two-groups.json`](networks/llama2-7b-two-groups.json). The JSON includes a content hash; startup validates schema, layer count, complete coverage, and hash to detect silent edits.

The network JSON describes model partitioning, Dealer coverage, inference implementation, and memory policy. It contains no host, process, thread, or CUDA-device placement.

## 2. Place and launch the serving network

### One host

Place two Dealers in two processes on two GPUs:

```bash
python -m scripts.online.benchmark.launch_network local \
  --network scripts/online/benchmark/networks/llama2-7b-two-groups.json \
  --port 7241 \
  --advertise-host 127.0.0.1 \
  --container shared \
  --place dealer-a 0 0 cuda:0 \
  --place dealer-b 1 0 cuda:1 \
  --agent decode-agent dealer-a
```

`local` requires complete placement of every Dealer in the JSON. `--agent NAME DEALER` colocates the Agent with that Dealer in one process/thread, so Orbit delivers original objects without serialization. The Agent is not part of the logical model specification and does not enter network JSON. `--container shared` is the default production path; `--container nonasync` uses `NonAsyncContainer` for controlled synchronous-CUDA experiments.

### Multiple hosts

On the coordinator, launch only Orbit and Aux:

```bash
python -m scripts.online.benchmark.launch_network orbit \
  --network scripts/online/benchmark/networks/llama2-7b-two-groups.json \
  --port 7241 \
  --advertise-host 10.0.0.10
```

Host A:

```bash
python -m scripts.online.benchmark.launch_network satellite \
  --network scripts/online/benchmark/networks/llama2-7b-two-groups.json \
  --orbit 10.0.0.10:7241 \
  --host-id gpu-host-a \
  --advertise-host 10.0.0.11 \
  --place dealer-a 0 0 cuda:0 \
  --agent decode-agent dealer-a
```

Host B:

```bash
python -m scripts.online.benchmark.launch_network satellite \
  --network scripts/online/benchmark/networks/llama2-7b-two-groups.json \
  --orbit 10.0.0.10:7241 \
  --host-id gpu-host-b \
  --advertise-host 10.0.0.12 \
  --place dealer-b 0 0 cuda:0
```

`PROCESS` and `THREAD` are numbered within each host's LaunchPad and have no implicit relationship to group IDs, Dealer names, or CUDA devices. Satellite commands may join incrementally but must use the same network JSON.

CUDA graphs are captured explicitly by `Brain.capture_decode_graph()` during Dealer initialization. A benchmark cannot enable or recapture them later.

## 3. Temporary Request node

General inference:

```bash
python -m scripts.online.benchmark.benchmark_infer \
  --orbit 127.0.0.1:7241 \
  --model teared-Llama-2-7B-GPTQ \
  --seqs 8 \
  --max-tokens 256 \
  --warmup-tokens 32 \
  --num-paths 1 \
  --output results/online/infer-seqs8
```

Decode-only:

```bash
python -m scripts.online.benchmark.benchmark_infer \
  --measurement decode \
  --orbit 127.0.0.1:7241 \
  --agent decode-agent \
  --model teared-Llama-2-7B-GPTQ \
  --expected-dealers 2 \
  --num-seqs 1 8 \
  --runs 3 \
  --warmup-requests 2 \
  --warmup-decode-tokens 8 \
  --decode-tokens 200 \
  --max-model-len 256 \
  --output results/online/decode-two-dealers
```

The Request's first `infer_step()` creates a remote session and submits prefill. The Agent then feeds each Dealer output directly into the next round while streaming tokens to the Request. Later Request `infer_step()` calls consume the next result only. Benchmark call code is identical for both placements.

If `--agent` is omitted, `RequestNode` constructs an Agent inside its current `BaseNode`, equivalent to the former root-control benchmark. Request, decode, and failure modes support external Agents. Adapter inference also permits an external Agent; when PEFT conditions do not support graph replay, eager fallback is expected and should be confirmed from graph statistics in output.

Decode timing uses the Request clock only. It starts after prefill and decode warmup and stops when the Request consumes the last measured token. The Agent exposes no benchmark-specific RPC or internal timing result.

See [`docs/online-agent-request-design.md`](../../../docs/online-agent-request-design.md) for node and transport design.

P-Tuning training:

```bash
python -m scripts.online.benchmark.benchmark_train \
  --orbit 127.0.0.1:7241 \
  --agent decode-agent \
  --model teared-Llama-2-7B-GPTQ \
  --instances 1 \
  --samples-per-instance 256 \
  --minibatch-size 4 \
  --grad-accum-steps 4 \
  --sample-len 128 \
  --adapter ptuning \
  --ptuning-prefix-len 32 \
  --learning-rate 1e-2 \
  --ckpt-period 16 \
  --detail-trace \
  --output results/online/train-ptuning
```

Training uses the same `RequestNode` deployment model. With `--agent`, the Request sends serializable PEFT, optimizer, scheduler, initialization policy, and minibatches. The real `TrainControl`, checkpoints, and recovery state remain at the Agent; PEFT/optimizer tensors remain at Dealers. Independent RPC tickets match loss/trace for concurrent minibatches. Omitting `--agent` constructs the Agent internally without changing the call interface.

Each entry point writes effective arguments to `args.json` and records `network_name` and `network_spec_hash` from Aux. Consult `--help` for the current CLI.

## 4. Failure injection

Add `--failure-injection` when generating the logical network to expose Dealer `infer_drop_reset` RPC. The flag only determines whether long-running Dealers support injection; the benchmark sets probabilities before the test. Ordinary networks omit this logic.

The unified inference entry point also runs a failure-probability matrix:

```bash
python -m scripts.online.benchmark.benchmark_infer \
  --measurement failure \
  --orbit 127.0.0.1:7241 \
  --model teared-Llama-2-7B-GPTQ \
  --seqs 8 --max-tokens 256 --warmup-tokens 32 \
  --drop-probabilities 0 0.001 0.01 \
  --repeats 3 --inspection-postpone 0.3 \
  --logprobs 0 \
  --output results/online/infer-failure
```

An external persistent Agent can execute the failure protocol, but benchmark `--inspection-postpone` mutates `node.flags.InspectPostpone` only in the benchmark process and cannot affect a persistent Agent in another process. Results with non-zero postpone therefore cannot compare combined and persistent Agents under equivalent recovery semantics; no persistent-Agent `p != 0` performance claim is made.

Each group writes `drop-P/repeat-N/`. Before matrix completion or exception propagation, the entry point attempts to reset Dealer drop probability to zero.

`net_cluster` is a separate Linux namespace/veth/tc tool and is neither part of this configuration layer nor invoked automatically by a benchmark.
