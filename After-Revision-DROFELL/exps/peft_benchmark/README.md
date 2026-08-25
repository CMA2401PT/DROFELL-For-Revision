# Llama-2-7B PEFT Throughput Experiment

## Purpose

Run an online P-Tuning and LoRA training matrix to check combined Request-Agent deployment, client count, effective batch, adapter parameters, and decoder backend.

## Parameter semantics

- model: `teared-Llama-2-7B-GPTQ`;
- topology: place contiguous decoder groups on target GPUs through the runner's network specification;
- combine Request and Agent by omitting `--agent`;
- effective batch = `minibatch-size × grad-accum-steps`;
- distribute total LoRA rank across Q/V projections rather than assigning the total to each projection;
- treat `exllama_fuse`, Marlin, and other backends as explicit cases with separate output.

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export RUN_ID="${RUN_ID:-peft-benchmark}"

PYTHONPATH="$PWD" "$DROFELL_PY" exps/peft_benchmark/run.py \
  --python "$DROFELL_PY" \
  --run-id "$RUN_ID" \
  --samples-per-client 256
```

Use `run.py --help` for backend, case, and recovery options. Run a small case subset as smoke, then execute the complete matrix under a new run ID. Do not repeat runs and select the best value.

## Output and acceptance

The runner generates network specifications, argument snapshots, per-case training records, logs, and automatic summaries in Git-ignored directories.

Check that:

- Agent semantics in `args.json` match deployment;
- client, minibatch, and gradient-accumulation mappings are correct;
- every case completes expected samples and optimizer updates;
- network-spec snapshots match GPU placement;
- top-level Python processes use the same CPU scheduling configuration;
- residual Orbit/Dealer processes and GPU memory are cleared before running;
- outputs from different logical partitions or backends are not combined into one case.
