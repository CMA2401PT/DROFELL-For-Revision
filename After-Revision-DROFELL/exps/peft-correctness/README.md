# PEFT Correctness and Convergence Experiment

## Purpose

Under identical data order and optimizer configuration, compare P-Tuning and LoRA across reference, offline, online-static, and online-recovery execution.

The protocol checks that training continues and that loss/state do not become invalid. It does not assume bitwise identity of each minibatch loss, gradient, or final parameter.

## Fixed conditions

- foundation model: `teared-Llama-2-7B-GPTQ`;
- adapters: P-Tuning and LoRA;
- optimizer: AdamW;
- fixed sample selection, order, and seed;
- identical topology for online-static and online-recovery;
- explicit failure points and checkpoint period for recovery.

## Running

```bash
export DROFELL_PY="${DROFELL_PY:-python}"
export RUN_ID="${RUN_ID:-peft-correctness}"

PYTHONPATH="$PWD" "$DROFELL_PY" exps/peft-correctness/run.py \
  --python "$DROFELL_PY" \
  --run-id "$RUN_ID" \
  --adapter ptuning --adapter lora \
  --duration-s 1800 \
  --submission-interval-ms 500 \
  --minibatch-size 1 --grad-accum-steps 1 \
  --learning-rate 1e-4 --optimizer-eps 1e-4 \
  --max-query-tokens -1 --max-response-tokens -1 \
  --max-seq-len 1024 \
  --ptuning-prefix-len 128 --lora-len-limit 128 \
  --checkpoint-period 10 \
  --failure-times-s 450 900 1350 \
  --seed 20260815
```

Use a shorter `--duration-s` and `--dry-run` for environment checks; give short runs independent run IDs.

## Output and acceptance

Logs, manifests, sample indexes, per-minibatch records, state summaries, and figures are local generated files and are not version-controlled.

Check that:

- all four conditions use identical sample IDs and order;
- adapter initialization or checkpoint source is recorded explicitly;
- minibatch/update counts are complete without duplicate IDs;
- loss, gradients, and parameters contain no NaN/Inf;
- failures trigger at configured points;
- version, optimizer, and checkpoint progress continue after recovery;
- similar loss is not interpreted as equal state when reference and DROFELL implementations differ.
