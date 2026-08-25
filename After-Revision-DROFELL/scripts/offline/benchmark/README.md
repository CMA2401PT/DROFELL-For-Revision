# Offline Benchmark

This directory measures inference and training through the offline `Brain`, providers, and `PipelineParallelTestbed`, without an Orbit network. Every runtime parameter is explicit on the command line; entry points neither import profiles dynamically nor support `--preset` or `--config`.

## Entry points

```bash
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" \
  scripts/offline/benchmark/inference/benchmark.py --help
PYTHONPATH="$PWD" "${DROFELL_PY:-python}" \
  scripts/offline/benchmark/train/benchmark.py --help
```

Run both Python entry points from the repository root. Unknown arguments fail immediately.

## Topology semantics

`--decoder-splits` lists contiguous decoder-layer counts. Their sum must equal the model's decoder-layer count. The number of splits must be divisible by the number of devices; Bindings are assigned to devices in order and evenly.

Examples:

- 7B, one GPU: `--devices cuda:0 --decoder-splits 32`
- 7B, two GPUs: `--devices cuda:0 cuda:1 --decoder-splits 16 16`
- 70B, two GPUs: `--devices cuda:0 cuda:1 --decoder-splits 40 40`
- one GPU, four graph/Binding segments: `--devices cuda:0 --decoder-splits 8 8 8 8`

Each GPU has one `Brain` and one `SharedBaseContainer`. These containers share the offline blocking scheduler; they are not separate Orbit processes.

See [`inference/README.md`](inference/README.md) and [`train/README.md`](train/README.md) for complete examples. Each run writes effective arguments to `args.json` in its output directory.

Set `--max-seq-len` explicitly, especially with graphs, to avoid inheriting a model's very large context limit. `static` memory policy uses `--major-reserved-mib`; `percentage` uses `--major-percentage`. Other reservation/minimum/maximum parameters apply to both.

## Recovery schedule

Inference and training share one format:

```bash
--recover 10 20 30:split 40
```

This represents `[(10, False), (20, False), (30, True), (40, False)]`. `STEP` triggers recovery at that step; `STEP:split` additionally simulates a complete node split. Omitting `--recover` disables recovery. Steps must be unique and strictly increasing, and at most one may use `:split`. Inference steps begin at 1; training allows step 0, before the first minibatch submission.

Steps are per inference/training instance, not global benchmark counters. Concurrent instances execute the same schedule independently. Training steps follow each instance's minibatch submission order rather than asynchronous completion order. Any training recovery requires a positive `--backup-period`.
