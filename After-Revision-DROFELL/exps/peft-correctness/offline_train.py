#!/usr/bin/env python3
"""Finite, paced DROFELL offline (sharded, no network) PEFT run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import List

import torch

import cache
import data
import extend_asyncio as asyncio
import split_qllm_models
from dataset.advertise_gen import make_dataset_adv_gen
from dataset.common import ProcessedRecord
from dealers.common.base_types import balance_divide
from dealers.common.brain import OptimizerConfig, UserModelDataInitConfig
from dealers.memory_schedule.brain import Brain
from node.compute_node.train_meta import TrainHeadInput
from operators.container import get_thread_context
from operators.container.base_container import BaseContainer
from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig
from scripts.base.train_state_maintainer import StepSize, TrainStatusMaintainer
from scripts.offline.testbed import ModelConfig, PipelineParallelTestbed
from scripts.offline.testbed.train_instance import TrainTestConfig
from utils.minibatch_feeder import MinibatchFeeder


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_indices(path: Path) -> list[int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or any(type(item) is not int for item in value):
        raise ValueError("sample-index file must be a non-empty JSON integer list")
    return value


async def train(args: argparse.Namespace) -> None:
    set_determinism(args.seed)
    indices = load_indices(args.sample_indices)
    expected = args.minibatches * args.minibatch_size
    if len(indices) != expected:
        raise ValueError(f"expected {expected} sample indices, got {len(indices)}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    device = torch.device(args.device)
    stream = torch.cuda.Stream(device=device)
    container = BaseContainer(stream)
    cfg = ModelConfig(
        decoder_backend="exllama_fuse",
        max_seq_len=args.max_seq_len,
    )
    brain = Brain(
        models_path=split_qllm_models.lib_path,
        device=device,
        container=container,
        max_seq_len=cfg.max_seq_len,
        max_concurrency=8,
        decoder_implement=cfg.decoder_backend,
    )
    testbed = PipelineParallelTestbed([brain], cfg)

    if args.adapter == "ptuning":
        adapter_config = PtuningConfig(
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.0,
            prefix_len=args.ptuning_prefix_len,
        )
    else:
        adapter_config = LoraConfig(
            r_q=-1,
            r_k=0,
            r_v=-1,
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.0,
            len_limit=args.lora_len_limit,
        )

    dataset = make_dataset_adv_gen(
        json_file=data.advertise_gen_train_path,
        build_prompt_str=adapter_config.build_prompt_str,
        tokenizer_path=cfg.model_info.tokenizer_path,
        eos_token_id=cfg.model_info.config.tokenizer_special_tokens.eos_token_id,
        max_query_tokens=args.max_query_tokens,
        max_response_tokens=args.max_response_tokens,
        force_query_tokens=0,
        force_response_tokens=0,
        cache_dir=cache.dataset_cache_path,
    )
    records = dataset.records
    if min(indices) < 0 or max(indices) >= len(records):
        raise ValueError("sample index is outside the processed dataset")
    selected = [records[index] for index in indices]
    cursor = 0

    def next_records() -> List[ProcessedRecord] | None:
        nonlocal cursor
        if cursor >= len(selected):
            return None
        result = selected[cursor:cursor + args.minibatch_size]
        cursor += len(result)
        return result

    feeder = MinibatchFeeder(next_records, args.grad_accum_steps)
    groups = balance_divide(
        base_models=testbed.base_model_names,
        with_head=True,
        with_tail=True,
        decoders_groups=args.groups,
        user_model_prefix="user_model",
    )
    instance = await testbed.create_train_instance(
        groups=groups,
        user_model_config=adapter_config,
        optimizer_config=OptimizerConfig(
            optimizer_name="AdamW",
            lr=args.learning_rate,
            eps=args.optimizer_eps,
        ),
        scheduler_config=None,
        data_init_method=UserModelDataInitConfig(init_method="reinit"),
        test_config=TrainTestConfig(backup_period=args.checkpoint_period),
    )
    status = TrainStatusMaintainer(
        step_size=StepSize(args.grad_accum_steps, args.minibatch_size),
        total_samples=len(selected),
    )
    started = time.monotonic()
    for minibatch_index in range(args.minibatches):
        item = feeder.get_next()
        if item is None:
            raise RuntimeError("sample feeder ended before the configured workload")
        batch_ids, request_version, minibatch_id, batch_records = item
        start_time = time.time()
        loss = await instance.do_mini_batch(
            request_version,
            full_batch_ids=batch_ids,
            minibatch_id=minibatch_id,
            sample=TrainHeadInput.from_records(batch_records, is_bfloat16=False),
        )
        tokens = sum(len(record.input_ids) for record in batch_records)
        print(status.update_loss(loss, tokens, start_time, minibatch_id))
        deadline = started + (minibatch_index + 1) * args.submission_interval_ms / 1000.0
        remaining = deadline - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)

    await asyncio.sleep(0.5)
    checkpoints = {}
    for _, shadow in instance.nodes_and_data:
        for model, checkpoint in shadow.ready_ckpts.items():
            checkpoints[model] = {"version": checkpoint.version, "state": checkpoint.state}
    torch.save(checkpoints, output / "final-checkpoints.pt")
    (output / "record.json").write_text(
        json.dumps(status.dump_info(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "args.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "offline",
                "adapter": args.adapter,
                "config": vars(args) | {
                    "output": str(output),
                    "sample_indices": str(args.sample_indices),
                },
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    instance.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("ptuning", "lora"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-indices", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--minibatches", type=int, default=3600)
    parser.add_argument("--minibatch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--submission-interval-ms", type=float, default=500.0)
    parser.add_argument("--max-query-tokens", type=int, default=-1)
    parser.add_argument("--max-response-tokens", type=int, default=-1)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--checkpoint-period", type=int, default=10)
    parser.add_argument("--ptuning-prefix-len", type=int, default=128)
    parser.add_argument("--lora-len-limit", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--optimizer-eps", type=float, default=1e-4)
    parser.add_argument("--async-backend", choices=("py", "chimera", "golang"), default="golang")
    parser.add_argument("--busy-wait", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.minibatches, args.minibatch_size, args.grad_accum_steps,
           args.groups, args.max_seq_len) < 1:
        parser.error("batch, group, and workload sizes must be positive")
    if args.minibatches % args.grad_accum_steps != 0:
        parser.error("minibatches must be divisible by grad-accum-steps")
    if args.max_query_tokens != -1 or args.max_response_tokens != -1:
        parser.error("this experiment requires untruncated query/response tokens (-1/-1)")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    asyncio.config_impl(parsed.async_backend, parsed.busy_wait)
    asyncio.run(train(parsed))
