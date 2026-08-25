"""Shared CLI and trace-envelope helpers for memory capture scripts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import pickle
from typing import Any

import torch

from dealers.analysis.recorder import ActualIORecorder
from dealers.memory_schedule.restriction import MemoryConfigStatic
from scripts.offline.testbed import ModelConfig


DECODER_BACKENDS = (
    "triton",
    "exllama_v2",
    "exllama_v1",
    "cuda_old",
    "cuda",
    "marlin",
    "exllama",
    "exllama_fuse",
)


def add_capture_runtime_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_max_concurrency: int,
    default_preallocate: bool,
) -> None:
    parser.add_argument("--model-name", default="teared-Llama-2-7B-GPTQ")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="PyTorch CPU/CUDA seed used before model and dataset setup",
    )
    parser.add_argument(
        "--decoder-backend", choices=DECODER_BACKENDS, default="marlin"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=-1,
        help="model default when omitted or -1",
    )
    parser.add_argument(
        "--max-layers-to-load",
        type=int,
        default=-1,
        help="load only the first N decoder layers; -1 loads all layers",
    )
    parser.add_argument(
        "--decoder-group-size",
        type=int,
        default=8,
        help="number of decoder layers in each model group",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=default_max_concurrency
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "capture decode graphs for batch sizes 1..max-concurrency while "
            "models are loaded"
        ),
    )
    parser.add_argument(
        "--preallocate",
        action=argparse.BooleanOptionalAction,
        default=default_preallocate,
        help="preallocate PEFT optimizer/gradient memory in the Brain policy",
    )
    parser.add_argument(
        "--preswap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable scalable KV-cache preswap",
    )
    parser.add_argument(
        "--root-memory-gib",
        type=float,
        help="override the Brain memory-policy capacity; defaults to GPU total",
    )
    parser.add_argument(
        "--graph-budget-mib",
        type=int,
        default=0,
        help="hard CUDA graph memory budget; required when graphs are enabled",
    )


def validate_capture_runtime_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    for name in ("decoder_group_size", "max_concurrency"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_seq_len == 0 or args.max_seq_len < -1:
        parser.error("--max-seq-len must be positive or -1")
    if args.max_layers_to_load == 0 or args.max_layers_to_load < -1:
        parser.error("--max-layers-to-load must be positive or -1")
    if args.root_memory_gib is not None and args.root_memory_gib <= 0:
        parser.error("--root-memory-gib must be positive")
    if args.graph_budget_mib < 0:
        parser.error("--graph-budget-mib must be non-negative")
    if args.cuda_graph and args.graph_budget_mib <= 0:
        parser.error("--cuda-graph requires a positive --graph-budget-mib")


def make_model_config(args: argparse.Namespace) -> ModelConfig:
    group_size = args.decoder_group_size

    def group_fn(layers: int) -> list[int]:
        return [
            min(group_size, layers - start)
            for start in range(0, layers, group_size)
        ]

    return ModelConfig(
        model_name=args.model_name,
        decoder_backend=args.decoder_backend,
        enable_cuda_graph=args.cuda_graph,
        max_seq_len=args.max_seq_len,
        max_layers_to_load=args.max_layers_to_load,
        group_fn=group_fn,
    )


def make_memory_config(args: argparse.Namespace) -> MemoryConfigStatic:
    root_memory = -1
    if args.root_memory_gib is not None:
        root_memory = round(args.root_memory_gib * 2**30)
    return MemoryConfigStatic(
        root_memory=root_memory,
        preallocate=args.preallocate,
        preswap=args.preswap,
        graph_budget=args.graph_budget_mib * 2**20,
    )


def capture_metadata(
    *,
    cfg: ModelConfig,
    memory_config: MemoryConfigStatic,
    max_concurrency: int,
    decoder_group_size: int,
    random_seed: int,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    device = torch.device("cuda", torch.cuda.current_device())
    return {
        "model_name": cfg.model_name,
        "max_seq_len": cfg.max_seq_len,
        "max_layers_to_load": cfg.max_layers_to_load,
        "decoder_backend": cfg.decoder_backend,
        "decoder_group_size": decoder_group_size,
        "random_seed": random_seed,
        "max_concurrency": max_concurrency,
        "enable_cuda_graph": cfg.enable_cuda_graph,
        "graph_batch_sizes": (
            list(range(1, max_concurrency + 1))
            if cfg.enable_cuda_graph else []
        ),
        "memory_config": asdict(memory_config),
        "device": {
            "index": device.index,
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        },
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "scenario": scenario,
    }


def dump_capture(
    output: Path,
    recorder: ActualIORecorder,
    metadata: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(recorder.capture_payload(metadata), handle)
