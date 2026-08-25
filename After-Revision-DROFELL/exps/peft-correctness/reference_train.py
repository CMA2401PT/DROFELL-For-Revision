#!/usr/bin/env python3
"""Finite, paced monolithic PEFT reference run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import extend_asyncio as asyncio
from operators.container.container import NonAsyncContainer
from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig
from scripts.base.train import TrainConfig, run


def load_indices(path: Path) -> list[int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or any(type(item) is not int for item in value):
        raise ValueError("sample-index file must be a non-empty JSON integer list")
    return value


async def train(args: argparse.Namespace) -> None:
    indices = load_indices(args.sample_indices)
    expected = args.minibatches * args.minibatch_size
    if len(indices) != expected:
        raise ValueError(f"expected {expected} sample indices, got {len(indices)}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

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

    config = TrainConfig(
        user_model_config=adapter_config,
        decoder_backend="exllama_fuse",
        batch_size=args.minibatch_size,
        steps_in_update=args.grad_accum_steps,
        sample_indices=indices,
        submission_interval_ms=args.submission_interval_ms,
        max_query_tokens=args.max_query_tokens,
        max_response_tokens=args.max_response_tokens,
        force_query_tokens=0,
        force_response_tokens=0,
        lr=args.learning_rate,
        final_lr=-1,
        eps=args.optimizer_eps,
        bfloat16_end=-1,
        max_steps=args.minibatches,
        seed=args.seed,
        event_dir=str(output),
        ckpt_steps=0,
        best_ckpt_name="",
        record_steps=0,
        memory_steps=0,
        final_ckpt_name="final.pt",
    )
    provenance = {
        "schema_version": 1,
        "mode": "reference",
        "adapter": args.adapter,
        "config": vars(args) | {"output": str(output), "sample_indices": str(args.sample_indices)},
    }
    (output / "args.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    device = torch.device(args.device)
    stream = torch.cuda.Stream(device=device)
    container = NonAsyncContainer(stream)
    await run(config, container, device=container.device)


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
    parser.add_argument("--ptuning-prefix-len", type=int, default=128)
    parser.add_argument("--lora-len-limit", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--optimizer-eps", type=float, default=1e-4)
    parser.add_argument("--async-backend", choices=("py", "chimera", "golang"), default="golang")
    parser.add_argument("--busy-wait", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.minibatches, args.minibatch_size, args.grad_accum_steps) < 1:
        parser.error("batch and workload sizes must be positive")
    if args.minibatches % args.grad_accum_steps != 0:
        parser.error("minibatches must be divisible by grad-accum-steps")
    if args.max_query_tokens != -1 or args.max_response_tokens != -1:
        parser.error("this experiment requires untruncated query/response tokens (-1/-1)")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    asyncio.config_impl(parsed.async_backend, parsed.busy_wait)
    asyncio.run(train(parsed))
