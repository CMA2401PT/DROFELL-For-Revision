#!/usr/bin/env python3
"""Capture sequential or concurrent minibatch training memory behavior."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

import cache
import data
import extend_asyncio as asyncio
import split_qllm_models
from dataset.advertise_gen import make_dataset_adv_gen
from dataset.common import ProcessedRecord
from dealers.analysis.frame import AnalysisFrame
from dealers.analysis.recorder import ActualIORecorder
from dealers.common.base_types import balance_divide
from dealers.common.brain import OptimizerConfig, UserModelDataInitConfig
from dealers.memory_schedule.brain import Brain
from node.compute_node.train_meta import TrainHeadInput
from operators.container import get_thread_context
from operators.container.base_container import BaseContainer
from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig
from scripts.offline.memory_modeling.capture_common import (
    add_capture_runtime_arguments,
    capture_metadata,
    dump_capture,
    make_memory_config,
    make_model_config,
    validate_capture_runtime_arguments,
)
from scripts.offline.testbed import PipelineParallelTestbed
from scripts.offline.testbed.train_instance import TrainTestConfig
from utils.minibatch_feeder import MinibatchFeeder


def make_user_model(method: str):
    if method == "ptuning":
        return (
            PtuningConfig(
                build_prompt_str="[instruction]{}\n[generation]",
                dropout_p=0.1,
                prefix_len=128,
            ),
            OptimizerConfig(
                optimizer_name="AdamW", lr=1e-2, eps=1e-4
            ),
        )
    return (
        LoraConfig(
            r_q=-1,
            r_k=0,
            r_v=-1,
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.0,
            len_limit=128,
        ),
        OptimizerConfig(optimizer_name="AdamW", lr=1e-3, eps=1e-4),
    )


async def amain(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    stream = torch.cuda.Stream(device=device)
    container = BaseContainer(stream)

    cfg = make_model_config(args)
    memory_config = make_memory_config(args)
    brain = Brain(
        models_path=split_qllm_models.lib_path,
        device=device,
        container=container,
        max_seq_len=cfg.max_seq_len,
        max_concurrency=args.max_concurrency,
        decoder_implement=cfg.decoder_backend,
        individual_train_thread=args.individual_train_thread,
        memory_config=memory_config,
    )
    recorder = ActualIORecorder(brain)
    analysis_brain = AnalysisFrame.wrap_brain(brain, recorder)

    user_model_config, optimizer_config = make_user_model(args.method)
    train_dataset = make_dataset_adv_gen(
        json_file=data.advertise_gen_train_path,
        build_prompt_str=user_model_config.build_prompt_str,
        tokenizer_path=cfg.model_info.tokenizer_path,
        eos_token_id=(
            cfg.model_info.config.tokenizer_special_tokens.eos_token_id
        ),
        max_query_tokens=-1,
        max_response_tokens=-1,
        cache_dir=cache.dataset_cache_path,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=lambda samples: samples,
        batch_size=args.minibatch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
    )
    get_thread_context().switch_cpu()
    train_iter = iter(train_loader)

    def next_records() -> List[ProcessedRecord] | None:
        get_thread_context().switch_cpu()
        try:
            return next(train_iter)
        except StopIteration:
            return None

    feeder = MinibatchFeeder(next_records, args.batch_size)
    testbed = PipelineParallelTestbed([analysis_brain], cfg)
    decoder_model_groups = len(testbed.base_model_names) - 2
    if decoder_model_groups % args.pipeline_decoder_groups != 0:
        raise ValueError(
            f"{decoder_model_groups} decoder model groups cannot be divided "
            f"into {args.pipeline_decoder_groups} pipeline groups"
        )
    groups = balance_divide(
        base_models=testbed.base_model_names,
        with_head=True,
        with_tail=True,
        decoders_groups=args.pipeline_decoder_groups,
        user_model_prefix="user_model",
    )
    train_instance = await testbed.create_train_instance(
        groups=groups,
        user_model_config=user_model_config,
        optimizer_config=optimizer_config,
        scheduler_config=None,
        data_init_method=UserModelDataInitConfig(init_method="reinit"),
        test_config=TrainTestConfig(backup_period=0),
    )

    async def do_one_batch(sample) -> None:
        batch_ids, request_version, minibatch_id, records = sample
        train_input = TrainHeadInput.from_records(
            records, is_bfloat16=args.bfloat16
        )
        await train_instance.do_mini_batch(
            request_version,
            full_batch_ids=batch_ids,
            minibatch_id=minibatch_id,
            sample=train_input,
        )

    pending = []
    submitted = 0
    while submitted < args.steps:
        sample = feeder.get_next()
        if sample is None:
            break
        pending.append(asyncio.create_task(do_one_batch(sample)))
        submitted += 1
        if len(pending) >= args.max_in_flight:
            _, still_pending = await asyncio.wait_first(tuple(pending))
            pending = list(still_pending)
    while pending:
        _, still_pending = await asyncio.wait_first(tuple(pending))
        pending = list(still_pending)

    train_instance.remove()
    await asyncio.sleep(0.1)
    for provider in testbed.providers:
        provider.remove()
    testbed.providers.clear()
    del testbed
    torch.cuda.synchronize(device)

    dump_capture(
        args.output,
        recorder,
        capture_metadata(
            cfg=cfg,
            memory_config=memory_config,
            max_concurrency=args.max_concurrency,
            decoder_group_size=args.decoder_group_size,
            random_seed=args.seed,
            scenario={
                "kind": "train",
                "method": args.method,
                "steps": submitted,
                "minibatch_size": args.minibatch_size,
                "batch_size": args.batch_size,
                "max_in_flight": args.max_in_flight,
                "pipeline_decoder_groups": args.pipeline_decoder_groups,
                "bfloat16": args.bfloat16,
                "individual_train_thread": args.individual_train_thread,
            },
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--method", choices=["ptuning", "lora"], default="ptuning"
    )
    parser.add_argument("--minibatch-size", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="number of minibatches in one optimizer-update batch",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=1,
        help="maximum concurrently submitted minibatches",
    )
    parser.add_argument(
        "--pipeline-decoder-groups", type=int, default=2
    )
    parser.add_argument(
        "--bfloat16", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--individual-train-thread",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    add_capture_runtime_arguments(
        parser, default_max_concurrency=8, default_preallocate=False
    )
    args = parser.parse_args()
    validate_capture_runtime_arguments(parser, args)
    for name in (
        "steps",
        "minibatch_size",
        "batch_size",
        "max_in_flight",
        "pipeline_decoder_groups",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    asyncio.config_impl("golang", True)
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
