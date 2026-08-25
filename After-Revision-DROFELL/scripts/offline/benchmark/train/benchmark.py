"""General offline training benchmark configured only through CLI options.

See ``README.md`` for complete 7B/70B and single/multi-GPU examples. Unknown
arguments are rejected; the effective values are written to ``args.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import List, Tuple

import extend_asyncio as asyncio
from asyncio import Task
from dataset.common import ProcessedRecord
from dealers.analysis.frame import AnalysisFrame
from dealers.analysis.recorder import TimeRecorder
from dealers.common.brain import (
    OptimizerConfig,
    UserModelDataInitConfig,
)
from node.compute_node import TrainHeadInput
from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig
from scripts.base.train_state_maintainer import TrainStatusMaintainer, StepSize
from scripts.offline.benchmark.common import (
    add_execution_arguments,
    add_memory_arguments,
    add_recovery_argument,
    make_brains,
    make_groups,
    make_model_config,
    positive_int,
    validate_execution_arguments,
    validate_recovery_arguments,
    write_effective_args,
)
from scripts.offline.testbed import PipelineParallelTestbed
from scripts.offline.testbed.model_config import ModelConfig
from scripts.offline.testbed.train_instance import TrainTestConfig
from utils.minibatch_feeder import AsyncMinibatchFeeder


async def wait_first_ready(
    tasks: Tuple[Task[None], ...],
) -> tuple[set[Task[None]], set[Task[None]]]:
    """Avoid lite_asyncio.wait_first hanging when a task is already done."""

    done = {task for task in tasks if task.done()}
    if done:
        return done, set(tasks) - done
    completed, pending = await asyncio.wait_first(tasks)
    return set(completed), set(pending)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline node-form training benchmark without Orbit networking"
    )
    add_execution_arguments(parser)
    add_memory_arguments(parser)
    parser.add_argument("--output", required=True)
    parser.add_argument("--instances", type=positive_int, required=True)
    parser.add_argument("--samples-per-instance", type=positive_int, required=True)
    parser.add_argument("--minibatch-size", type=positive_int, required=True)
    parser.add_argument(
        "--grad-accum-steps",
        type=positive_int,
        required=True,
        help="Number of minibatches in one optimizer update.",
    )
    parser.add_argument("--sample-len", type=positive_int, required=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--backup-period", type=int, default=0)
    add_recovery_argument(parser)
    parser.add_argument(
        "--adapter",
        choices=("ptuning", "lora"),
        required=True,
    )
    parser.add_argument(
        "--adapter-prompt-template",
        default="[instruction]{}\n[generation]",
    )
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--ptuning-prefix-len", type=positive_int, default=128)
    parser.add_argument("--lora-r-q", type=int, default=-1)
    parser.add_argument("--lora-r-k", type=int, default=0)
    parser.add_argument("--lora-r-v", type=int, default=-1)
    parser.add_argument("--lora-len-limit", type=positive_int, default=128)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--optimizer-eps", type=float, default=1e-4)
    args = parser.parse_args()
    validate_execution_arguments(parser, args)
    validate_recovery_arguments(parser, args.recover, allow_step_zero=True)
    if args.backup_period < 0:
        parser.error("--backup-period must be non-negative")
    if args.recover and args.backup_period == 0:
        parser.error("--recover requires a positive --backup-period")
    if args.samples_per_instance < 2 * args.minibatch_size:
        parser.error(
            "--samples-per-instance must provide at least two minibatches"
        )
    if not 0.0 <= args.adapter_dropout <= 1.0:
        parser.error("--adapter-dropout must be in [0, 1]")
    if args.learning_rate <= 0 or args.optimizer_eps <= 0:
        parser.error("optimizer learning rate and epsilon must be positive")
    return args


def make_adapter(args: argparse.Namespace):
    if args.adapter == "ptuning":
        return PtuningConfig(
            build_prompt_str=args.adapter_prompt_template,
            dropout_p=args.adapter_dropout,
            prefix_len=args.ptuning_prefix_len,
        )
    return LoraConfig(
        r_q=args.lora_r_q,
        r_k=args.lora_r_k,
        r_v=args.lora_r_v,
        build_prompt_str=args.adapter_prompt_template,
        dropout_p=args.adapter_dropout,
        len_limit=args.lora_len_limit,
    )


async def amain(args: argparse.Namespace, model_config: ModelConfig) -> None:
    output_dir = Path(args.output)
    brains = make_brains(
        args,
        model_config,
        max_concurrency=args.instances,
    )
    time_recorders = [TimeRecorder(sync_cuda=False) for _ in brains]
    testbed = PipelineParallelTestbed(
        [
            AnalysisFrame.wrap_brain(brain, recorder)
            for brain, recorder in zip(brains, time_recorders, strict=True)
        ],
        model_config,
    )
    write_effective_args(
        output_dir,
        args,
        extra={
            "effective_max_seq_len": model_config.max_seq_len,
            "model_decoder_layers": model_config.layers,
            "effective_samples_per_optimizer_update": (
                args.minibatch_size * args.grad_accum_steps
            ),
        },
    )

    adapter_config = make_adapter(args)
    optimizer_config = OptimizerConfig(
        optimizer_name=args.optimizer,
        lr=args.learning_rate,
        eps=args.optimizer_eps,
    )
    test_config = TrainTestConfig(
        backup_period=args.backup_period,
        recover=args.recover,
    )
    global_status = TrainStatusMaintainer(
        step_size=StepSize(
            steps_in_each_update=args.grad_accum_steps,
            step_batch_size=args.minibatch_size,
        ),
        total_samples=args.samples_per_instance * args.instances,
        events=None,
        prefix="total",
    )

    async def run_instance(prefix: str):
        total_samples = 0

        def gen_sample():
            nonlocal total_samples
            if total_samples >= args.samples_per_instance:
                return None
            records: List[ProcessedRecord] = []
            for _ in range(args.minibatch_size):
                records.append(
                    ProcessedRecord(
                        input_ids=[3 for _ in range(args.sample_len)],
                        output_ids=[3 for _ in range(args.sample_len)],
                    )
                )
                total_samples += 1
                if total_samples >= args.samples_per_instance:
                    break
            return records

        feeder = AsyncMinibatchFeeder(
            gen_sample,
            args.grad_accum_steps,
            args.grad_accum_steps * 2,
        )
        status = TrainStatusMaintainer(
            step_size=StepSize(
                steps_in_each_update=args.grad_accum_steps,
                step_batch_size=args.minibatch_size,
            ),
            total_samples=args.samples_per_instance,
            events=None,
            prefix=prefix,
        )
        train_instance = await testbed.create_train_instance(
            groups=make_groups(
                model_config,
                num_devices=len(args.devices),
                user_model_prefix=prefix,
            ),
            user_model_config=adapter_config,
            optimizer_config=optimizer_config,
            scheduler_config=None,
            data_init_method=UserModelDataInitConfig(
                init_method="reinit",
                ckpt_bytes=None,
            ),
            test_config=test_config,
        )

        async def do_one_minibatch(
            sample: Tuple[List[str], int, str, List[ProcessedRecord]],
        ) -> None:
            batch_ids, request_version, minibatch_id, records = sample
            inputs = TrainHeadInput.from_records(records, is_bfloat16=args.bf16)
            started = time.time()
            loss = await train_instance.do_mini_batch(
                request_version,
                full_batch_ids=batch_ids,
                minibatch_id=minibatch_id,
                sample=inputs,
            )
            total_tokens = sum(len(record.input_ids) for record in records)
            print(status.update_loss(loss, total_tokens, started, minibatch_id))
            print(global_status.update_loss(loss, total_tokens, started, minibatch_id))

        initial_tasks: list[Task[None]] = []
        for _ in range(2):
            sample = await feeder.get_next(wait=False)
            if sample is None:
                raise RuntimeError("training feeder ended during initial submission")
            initial_tasks.append(asyncio.create_task(do_one_minibatch(sample)))
        running_tasks: Tuple[Task[None], ...] = tuple(initial_tasks)

        while True:
            sample = await feeder.get_next(wait=False)
            if sample is None:
                break
            running_tasks = (
                *running_tasks,
                asyncio.create_task(do_one_minibatch(sample)),
            )
            _, pending = await wait_first_ready(running_tasks)
            running_tasks = tuple(pending)
        while running_tasks:
            _, pending = await wait_first_ready(running_tasks)
            running_tasks = tuple(pending)
        return status.dump_info()

    results = await asyncio.gather(
        *(run_instance(f"train-{index}") for index in range(args.instances))
    )
    results.append(global_status.dump_info())
    (output_dir / "results.json").write_text(
        json.dumps(results) + "\n",
        encoding="utf-8",
    )
    (output_dir / "brain.json").write_text(
        json.dumps(
            {
                str(index): recorder.dump_info()
                for index, recorder in enumerate(time_recorders)
            }
        )
        + "\n",
        encoding="utf-8",
    )
    print("to review the result, run:")
    print(f"python scripts/offline/benchmark/train/analysis_utils.py {output_dir}")


def main() -> None:
    args = parse_args()
    model_config = make_model_config(args, enable_cuda_graph=False)
    if args.sample_len > model_config.max_seq_len:
        raise ValueError("--sample-len cannot exceed the effective max sequence length")
    asyncio.config_impl(args.async_backend, args.busy_wait)
    asyncio.run(amain(args, model_config))


if __name__ == "__main__":
    main()
