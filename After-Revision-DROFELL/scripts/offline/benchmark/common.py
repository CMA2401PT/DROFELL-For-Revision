"""Shared CLI construction helpers for offline benchmarks.

This module builds the existing Brain/PipelineParallelTestbed execution layer
from explicit command-line values.  It is intentionally not a configuration
file loader and does not contain model presets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import split_qllm_models
from dealers.common.base_types import balance_divide, group_model
from dealers.memory_schedule.brain import Brain
from dealers.memory_schedule.restriction import (
    MemoryConfig,
    MemoryConfigPercentage,
    MemoryConfigStatic,
)
from operators.container.base_container import SharedBaseContainer
from operators.container.container import NonAsyncContainer
from scripts.offline.testbed import ModelConfig

from ..testbed.pipeline_parallel_testbed import PipelineParallelTestbed


MIB = 2**20
RecoveryEvent = tuple[int, bool]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_recovery_event(value: str) -> RecoveryEvent:
    """Parse STEP or STEP:split into a recovery schedule entry."""

    step_text, separator, mode = value.partition(":")
    try:
        step = non_negative_int(step_text)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError(
            f"invalid recovery event {value!r}: STEP must be a non-negative integer"
        ) from error
    if not separator:
        return step, False
    if mode != "split":
        raise argparse.ArgumentTypeError(
            f"invalid recovery event {value!r}: expected STEP or STEP:split"
        )
    return step, True


def add_recovery_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recover",
        nargs="+",
        type=parse_recovery_event,
        default=[],
        metavar="STEP[:split]",
        help=(
            "Recovery schedule, for example: --recover 10 20 30:split 40. "
            "Omit this option to disable recovery."
        ),
    )


def validate_recovery_arguments(
    parser: argparse.ArgumentParser,
    recover: list[RecoveryEvent],
    *,
    allow_step_zero: bool,
) -> None:
    steps = [step for step, _ in recover]
    if not allow_step_zero and any(step == 0 for step in steps):
        parser.error("inference --recover steps must be positive")
    if any(current <= previous for previous, current in zip(steps, steps[1:])):
        parser.error("--recover steps must be unique and strictly increasing")
    if sum(split for _, split in recover) > 1:
        parser.error("--recover may contain at most one STEP:split event")


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True,
                        help="Model name in ModelsLib.")
    parser.add_argument(
        "--devices",
        nargs="+",
        required=True,
        metavar="CUDA_DEVICE",
        help="Ordered pipeline devices, for example: cuda:0 cuda:1.",
    )
    parser.add_argument(
        "--decoder-splits",
        nargs="+",
        type=positive_int,
        required=True,
        metavar="LAYERS",
        help=(
            "Ordered decoder binding sizes. Their sum must equal the model's "
            "decoder count and their count must be divisible by device count."
        ),
    )
    parser.add_argument(
        "--decoder-backend",
        choices=("marlin", "exllama", "exllama_fuse"),
        default="marlin",
    )
    parser.add_argument(
        "--max-seq-len",
        type=positive_int,
        required=True,
        help="Override the model maximum sequence length.",
    )
    parser.add_argument(
        "--async-backend",
        choices=("py", "chimera", "golang"),
        default="golang",
    )
    parser.add_argument(
        "--busy-wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable busy-waiting in the selected extend_asyncio backend.",
    )
    parser.add_argument(
        "--container",
        choices=("shared", "nonasync"),
        default="shared",
        help=(
            "Brain execution container. nonasync removes blocking-scheduler "
            "dispatch overhead for isolated offline measurements."
        ),
    )


def add_memory_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--memory-policy",
        choices=("static", "percentage"),
        default="static",
    )
    parser.add_argument("--major-reserved-mib",
                        type=non_negative_int, default=8192)
    parser.add_argument("--major-percentage", type=float, default=0.8)
    parser.add_argument("--unmanaged-reserved-mib",
                        type=non_negative_int, default=32)
    parser.add_argument("--minor-min-mib", type=non_negative_int, default=0)
    parser.add_argument("--minor-max-mib", type=non_negative_int, default=4096)
    parser.add_argument("--operate-reserved-mib",
                        type=non_negative_int, default=1024)
    parser.add_argument("--graph-budget-mib",
                        type=non_negative_int, default=0)


def validate_execution_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if len(set(args.devices)) != len(args.devices):
        parser.error("--devices values must be unique")
    devices = [torch.device(value) for value in args.devices]
    if any(device.type != "cuda" or device.index is None for device in devices):
        parser.error(
            "offline benchmark requires indexed devices such as cuda:0")
    device_count = torch.cuda.device_count()
    if any(device.index >= device_count for device in devices):
        parser.error(
            f"a requested CUDA device is unavailable; visible device count={device_count}"
        )
    if len(args.decoder_splits) % len(devices) != 0:
        parser.error(
            "the number of --decoder-splits entries must be divisible by "
            "the number of --devices"
        )
    if not 0.0 < args.major_percentage <= 1.0:
        parser.error("--major-percentage must be in (0, 1]")
    if args.minor_min_mib > args.minor_max_mib:
        parser.error("--minor-min-mib cannot exceed --minor-max-mib")
    if getattr(args, "cuda_graph", False) and args.graph_budget_mib <= 0:
        parser.error("--cuda-graph requires a positive --graph-budget-mib")


def make_model_config(
    args: argparse.Namespace,
    *,
    enable_cuda_graph: bool,
) -> ModelConfig:
    splits = tuple(args.decoder_splits)
    config = ModelConfig(
        model_name=args.model,
        decoder_backend=args.decoder_backend,
        enable_cuda_graph=enable_cuda_graph,
        max_seq_len=args.max_seq_len,
        group_fn=lambda _layers: list(splits),
    )
    if sum(splits) != config.layers:
        raise ValueError(
            "--decoder-splits must cover every decoder exactly: "
            f"model_layers={config.layers}, splits={list(splits)}, "
            f"sum={sum(splits)}"
        )
    return config


def make_memory_config(args: argparse.Namespace) -> MemoryConfig:
    common = {
        "major_percentage": args.major_percentage,
        "unmanaged_reserved": args.unmanaged_reserved_mib * MIB,
        "minor_minimum": args.minor_min_mib * MIB,
        "minor_maximum": args.minor_max_mib * MIB,
        "operate_reserved": args.operate_reserved_mib * MIB,
        "graph_budget": args.graph_budget_mib * MIB,
    }
    if args.memory_policy == "static":
        return MemoryConfigStatic(
            major_reserved=args.major_reserved_mib * MIB,
            **common,
        )
    return MemoryConfigPercentage(**common)


def make_brains(
    args: argparse.Namespace,
    model_config: ModelConfig,
    *,
    max_concurrency: int,
) -> list[Brain]:
    brains = []
    for value in args.devices:
        device = torch.device(value)
        torch.cuda.set_device(device)
        stream = torch.cuda.Stream(device=device)
        container = (
            NonAsyncContainer(stream)
            if args.container == "nonasync"
            else SharedBaseContainer(stream)
        )
        brains.append(
            Brain(
                models_path=split_qllm_models.lib_path,
                device=device,
                container=container,
                memory_config=make_memory_config(args),
                max_seq_len=model_config.max_seq_len,
                max_concurrency=max_concurrency,
                decoder_implement=model_config.decoder_backend,
            )
        )
    return brains


def make_testbed(
    args: argparse.Namespace,
    model_config: ModelConfig,
    *,
    max_concurrency: int,
) -> PipelineParallelTestbed:
    return PipelineParallelTestbed(
        brains=make_brains(args, model_config,
                           max_concurrency=max_concurrency),
        cfg=model_config,
    )


def make_groups(
    model_config: ModelConfig,
    *,
    num_devices: int,
    user_model_prefix: str,
) -> list[list[tuple[str, str]]]:
    return balance_divide(
        base_models=group_model(
            base_model_name=model_config.model_name,
            decoders=model_config.layers,
            group_fn=model_config.group_fn,
        ),
        with_head=True,
        with_tail=True,
        decoders_groups=num_devices,
        user_model_prefix=user_model_prefix,
    )


def write_effective_args(
    output_dir: Path,
    args: argparse.Namespace,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    values = dict(vars(args))
    values["schema_version"] = 1
    if extra:
        values.update(extra)
    (output_dir / "args.json").write_text(
        json.dumps(values, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
