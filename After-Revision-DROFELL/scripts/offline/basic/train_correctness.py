"""Deterministic, finite-step DROFELL training correctness runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

import cache
import data
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
from utils.size_metrics import retained_bytes
import extend_asyncio as asyncio
import split_qllm_models


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def tensor_summary(value: Any) -> Dict[str, Any]:
    tensors: List[torch.Tensor] = []

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensors.append(item.detach().cpu())
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    elements = sum(tensor.numel() for tensor in tensors)
    squared = sum(float(torch.sum(tensor.float() ** 2)) for tensor in tensors)
    maximum = max((float(torch.max(torch.abs(tensor.float())))
                  for tensor in tensors if tensor.numel()), default=0.0)
    return {
        "tensor_count": len(tensors),
        "elements": elements,
        "l2": squared ** 0.5,
        "max_abs": maximum,
        "retained_bytes": retained_bytes(value),
    }


async def train_main(args: argparse.Namespace) -> None:
    set_determinism(args.seed)
    device = torch.device(args.device)
    stream = torch.cuda.Stream(device=device)
    container = BaseContainer(stream)
    cfg = ModelConfig(
        decoder_backend=args.decoder_backend,
        max_layers_to_load=args.layers,
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

    if args.method == "ptuning":
        user_model_config = PtuningConfig(
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=args.dropout,
            prefix_len=args.trainable,
        )
        optimizer_config = OptimizerConfig(
            optimizer_name="AdamW", lr=1e-2, eps=1e-4)
    else:
        user_model_config = LoraConfig(
            r_q=-1, r_k=0, r_v=-1,
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=args.dropout,
            len_limit=args.trainable,
        )
        optimizer_config = OptimizerConfig(
            optimizer_name="AdamW", lr=1e-3, eps=1e-4)

    train_dataset = make_dataset_adv_gen(
        json_file=data.advertise_gen_train_path,
        build_prompt_str=user_model_config.build_prompt_str,
        tokenizer_path=cfg.model_info.tokenizer_path,
        eos_token_id=cfg.model_info.config.tokenizer_special_tokens.eos_token_id,
        max_query_tokens=-1,
        max_response_tokens=-1,
        cache_dir=cache.dataset_cache_path,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=lambda samples: samples,
        batch_size=args.minibatch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        generator=generator,
    )
    get_thread_context().switch_cpu()
    train_iter = iter(train_loader)

    def gen_sample() -> List[ProcessedRecord]:
        get_thread_context().switch_cpu()
        return next(train_iter)

    from utils.minibatch_feeder import MinibatchFeeder
    feeder = MinibatchFeeder(gen_sample, args.accumulation_steps)
    recover = [] if args.mode == "no_fault" else [
        (args.recover_step, args.split_recovery)]
    test_config = TrainTestConfig(
        backup_period=args.checkpoint_period, recover=recover)
    groups = balance_divide(
        base_models=testbed.base_model_names,
        with_head=True,
        with_tail=True,
        decoders_groups=args.groups,
        user_model_prefix="user_model",
    )
    train_instance = await testbed.create_train_instance(
        groups=groups,
        user_model_config=user_model_config,
        optimizer_config=optimizer_config,
        scheduler_config=None,
        data_init_method=UserModelDataInitConfig(init_method="reinit"),
        test_config=test_config,
    )
    status = TrainStatusMaintainer(
        step_size=StepSize(args.accumulation_steps, args.minibatch_size),
        total_samples=len(train_dataset),
    )
    step_duration_ms: List[float] = []
    training_started_ns = time.perf_counter_ns()
    for _ in range(args.steps):
        next_inp = feeder.get_next()
        if next_inp is None:
            break
        batch_ids, req_version, minibatch_id, records = next_inp
        sample = TrainHeadInput.from_records(
            records, is_bfloat16=args.bfloat16)
        step_started_ns = time.perf_counter_ns()
        loss = await train_instance.do_mini_batch(
            req_version,
            full_batch_ids=batch_ids,
            minibatch_id=minibatch_id,
            sample=sample,
        )
        step_duration_ms.append(
            (time.perf_counter_ns() - step_started_ns) / 1e6)
        status.update_loss(loss, batch_id=minibatch_id)
    training_duration_seconds = (
        time.perf_counter_ns() - training_started_ns) / 1e9

    await asyncio.sleep(0.5)
    checkpoints: Dict[str, Any] = {}
    snapshots: List[Dict[str, int]] = []
    seen_data: set[int] = set()
    for _, shadow_data in train_instance.nodes_and_data:
        if id(shadow_data) in seen_data:
            continue
        seen_data.add(id(shadow_data))
        snapshots.append(shadow_data.metrics_snapshot())
        for model, checkpoint in shadow_data.ready_ckpts.items():
            checkpoints[model] = {
                "version": checkpoint.version, "state": checkpoint.state}

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoints, output / "checkpoints.pt")
    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "method": args.method,
        "seed": args.seed,
        "steps": args.steps,
        "layers": cfg.layers,
        "loss": status.dump_info()["history"],
        "training_duration_seconds": training_duration_seconds,
        "samples_per_second": (
            args.steps * args.minibatch_size / training_duration_seconds
            if training_duration_seconds > 0 else None
        ),
        "step_duration_ms": step_duration_ms,
        "checkpoint_summary": tensor_summary(checkpoints),
        "shadow_snapshots": snapshots,
        "config": vars(args),
    }
    (output / "result.json").write_text(json.dumps(summary,
                                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    train_instance.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["no_fault", "recovery"], required=True)
    parser.add_argument(
        "--method", choices=["ptuning", "lora"], default="ptuning")
    parser.add_argument(
        "--decoder-backend", choices=["marlin", "exllama", "exllama_fuse"], default="marlin")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--checkpoint-period", type=int, default=1)
    parser.add_argument("--recover-step", type=int, default=4)
    parser.add_argument("--split-recovery", action="store_true")
    parser.add_argument("--trainable", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bfloat16", action="store_true")
    parser.add_argument("--async-backend",
                        choices=["py", "chimera", "golang"], default="golang")
    parser.add_argument("--busy-wait", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    asyncio.config_impl(parsed.async_backend, parsed.busy_wait)
    asyncio.run(train_main(parsed))
