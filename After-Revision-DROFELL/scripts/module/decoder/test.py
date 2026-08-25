"""Measure one decoder checkpoint/swap strategy in an isolated process."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
import pickle
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Literal

import torch
from torch import optim

import extend_asyncio as asyncio
import split_qllm_models
from operators import flags
from operators.assemble.decoders.real import (
    DecodersImplement,
    make_decoders_from_files,
)
from operators.assemble.fwd_ctx import backward, forward
from operators.container.container import NonAsyncContainer
from operators.core.ptuning.real import Ptuning, PtuningConfig
from cuda_extends.marlin import MarlinTuningParams
from operators.core.swappable.real import CanSwapObject
from operators.helpers.model_lib import ModelsLib


REPO_ROOT = Path(__file__).resolve().parents[3]
Strategy = Literal["each_decoder", "first.2pass", "first.grad", "grad"]
STRATEGIES: tuple[Strategy, ...] = (
    "each_decoder",
    "first.2pass",
    "first.grad",
    "grad",
)
BACKENDS = ("exllama_fuse", "exllama", "marlin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="teared-Llama-2-7B-GPTQ")
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--prompt-seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--swap", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--reuse-reconstruct-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    for name in ("layers", "batch_size", "seq_len", "prompt_seq_len", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def timing_stats(values_ns: list[int]) -> dict[str, Any]:
    values_ms = [value / 1e6 for value in values_ns]
    return {
        "samples_ns": values_ns,
        "mean_ms": statistics.fmean(values_ms),
        "median_ms": statistics.median(values_ms),
        "stdev_ms": statistics.stdev(values_ms) if len(values_ms) > 1 else 0.0,
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
    }


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    models = ModelsLib(split_qllm_models.lib_path)
    model_info = models.models_storage[args.model]

    prefix_config = PtuningConfig(
        build_prompt_str="[instruction]{}\n[generation]",
        prefix_len=args.prompt_seq_len,
    )
    prefix_config.num_key_value_heads = model_info.config.num_key_value_heads
    prefix_config.head_dim = model_info.config.head_dim
    prefix_config.num_layers = args.layers

    flags.RECONSTRUCTE_WEIGTH_FLOAT16 = False
    flags.RECONSTRUCTE_WEIGTH_BFLOAT16 = False
    flags.REUSE_RECONSTRUCT_CACHE_WHEN_GRAD = args.reuse_reconstruct_cache
    flags.USER_MODEL_ENABLE_CHECKPOTIN = True
    flags.DECODERS_TRAIN_METHOD = args.strategy

    with torch.device(device):
        decoders_group = make_decoders_from_files(
            implement=args.backend,
            config=model_info.config.quant_decoder_config,
            state_files=[
                model_info.decoder_path(layer_index)
                for layer_index in range(args.layers)
            ],
        )
        prefix_encoder = Ptuning(prefix_config)

    optimizer = optim.AdamW(prefix_encoder.parameters(), lr=1e-2, eps=1e-4)
    container = NonAsyncContainer(torch.cuda.current_stream(device))

    async def do_forward() -> CanSwapObject:
        def create_input() -> tuple[torch.Tensor, torch.Tensor]:
            hidden_states = torch.empty(
                size=(args.batch_size * args.seq_len, model_info.config.hidden_size),
                device=device,
                dtype=torch.float16,
            )
            position_ids = torch.arange(
                0, args.seq_len, dtype=torch.int32, device=device
            ).repeat(args.batch_size)
            return hidden_states, position_ids

        hidden_states, position_ids = await container.step(
            "create_input", create_input
        )
        input_lens = [args.seq_len for _ in range(args.batch_size)]
        _, forward_ctx = await forward(
            container=container,
            hidden_states=hidden_states,
            position_ids=position_ids,
            input_lens=input_lens,
            user_model=prefix_encoder,
            decoders=decoders_group,
        )
        return forward_ctx

    async def do_backward(forward_ctx: CanSwapObject) -> None:
        hidden_states_grad = await container.step(
            "create_fake_grad",
            torch.empty,
            size=(args.batch_size * args.seq_len, model_info.config.hidden_size),
            device=device,
            dtype=torch.float16,
        )
        await backward(
            container=container,
            grad=hidden_states_grad,
            ctx=forward_ctx,
            user_model=prefix_encoder,
            decoders=decoders_group,
        )
        optimizer.zero_grad()

    async def swap_out_in(forward_ctx: CanSwapObject) -> None:
        def operation() -> None:
            forward_ctx.swap_out(None)
            forward_ctx.swap_in(device=device, stream=None)
            torch.cuda.synchronize(device)

        await container.step("do_swap_out_in", operation)

    print("warmup start", flush=True)
    for iteration in range(args.warmup):
        forward_ctx = await do_forward()
        if args.swap:
            await swap_out_in(forward_ctx)
        await do_backward(forward_ctx)
        print(f"warmup {iteration + 1}/{args.warmup}", flush=True)
    if args.warmup:
        # Match the old warmup() helper's scope boundary. Otherwise assigning
        # the first timed context also destroys the final warmup context and
        # incorrectly charges that cleanup to the first forward sample.
        del forward_ctx
        gc.collect()
        torch.cuda.synchronize(device)

    print("timing start", flush=True)
    forward_time: list[int] = []
    swap_time: list[int] = []
    backward_time: list[int] = []
    for iteration in range(args.repeats):
        forward_start = time.time_ns()
        forward_ctx = await do_forward()
        torch.cuda.synchronize(device)
        forward_complete = time.time_ns()

        if args.swap:
            # Preserve the historical paper implementation's interval: swap
            # begins at the synchronized forward completion timestamp.
            swap_start = forward_complete
            await swap_out_in(forward_ctx)
            swap_complete = time.time_ns()
            swap_time.append(swap_complete - swap_start)

        backward_start = time.time_ns()
        await do_backward(forward_ctx)
        torch.cuda.synchronize(device)
        backward_complete = time.time_ns()
        forward_time.append(forward_complete - forward_start)
        backward_time.append(backward_complete - backward_start)
        print(f"timing {iteration + 1}/{args.repeats}", flush=True)

    print("memory capture start", flush=True)
    # The historical implementation returned from its timing helper before
    # starting memory history, which dropped the last timing context. Preserve
    # that scope boundary explicitly in this parameterized single-case runner.
    del forward_ctx
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.memory._record_memory_history()
    forward_ctx = await do_forward()
    torch.cuda.synchronize(device)
    snapshot_after_forward = torch.cuda.memory._snapshot()
    snapshot_after_swap_out = None
    snapshot_after_swap_in = None
    if args.swap:
        def capture_swap() -> tuple[dict[str, Any], dict[str, Any]]:
            forward_ctx.swap_out(None)
            torch.cuda.synchronize(device)
            after_out = torch.cuda.memory._snapshot()
            forward_ctx.swap_in(device=device, stream=None)
            torch.cuda.synchronize(device)
            after_in = torch.cuda.memory._snapshot()
            return after_out, after_in

        snapshot_after_swap_out, snapshot_after_swap_in = await container.step(
            "capture_swap", capture_swap
        )
    await do_backward(forward_ctx)
    torch.cuda.synchronize(device)
    snapshot_after_backward = torch.cuda.memory._snapshot()
    print("memory capture complete", flush=True)

    props = torch.cuda.get_device_properties(device)
    metadata = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "parameters": {
            "model": args.model,
            "backend": args.backend,
            "strategy": args.strategy,
            "layers": args.layers,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "flattened_tokens": args.batch_size * args.seq_len,
            "prompt_seq_len": args.prompt_seq_len,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "swap": args.swap,
            "dtype": "float16",
            "user_model_checkpoint": "seed",
            "reuse_reconstruct_cache_when_grad": args.reuse_reconstruct_cache,
            "marlin_tuning": (
                asdict(MarlinTuningParams())
                if args.backend == "marlin"
                else None
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": props.name,
            "gpu_total_memory_bytes": props.total_memory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "timing": {
            "forward": timing_stats(forward_time),
            "swap": timing_stats(swap_time) if swap_time else None,
            "backward": timing_stats(backward_time),
        },
    }
    return {
        "metadata": metadata,
        "forward_memory": snapshot_after_forward,
        "swap_out_memory": snapshot_after_swap_out,
        "swap_in_memory": snapshot_after_swap_in,
        "backward_memory": snapshot_after_backward,
        "forward_time": forward_time,
        "swap_time": swap_time,
        "backward_time": backward_time,
    }


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.config_impl("golang", True)
    record = asyncio.run(run_benchmark(args))
    with args.output.open("wb") as stream:
        pickle.dump(record, stream)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(record["metadata"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
