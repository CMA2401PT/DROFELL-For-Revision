#!/usr/bin/env python3
"""Fingerprint a short monolithic P-Tuning run around every AdamW update."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import extend_asyncio as asyncio
from operators.container.container import NonAsyncContainer
from operators import flags as operator_flags
from operators.core.ptuning.common import PtuningConfig
from operators.core.ptuning.real import Ptuning
import scripts.base.train as base_train
from scripts.base.train import TrainConfig


def tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous().to("cpu")
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    numeric = value.float()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "numel": value.numel(),
        "mean": float(numeric.mean()) if value.numel() else 0.0,
        "max_abs": float(numeric.abs().max()) if value.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(numeric)) if value.numel() else 0.0,
    }


def tree_fingerprint(value: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    tensors: dict[str, Any] = {}

    def visit(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            fp = tensor_fingerprint(item)
            tensors[path] = fp
            digest.update(path.encode())
            digest.update(fp["sha256"].encode())
            digest.update(fp["dtype"].encode())
            digest.update(json.dumps(fp["shape"]).encode())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: str(key)):
                visit(item[key], f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        else:
            digest.update(f"{path}={item!r}".encode())

    visit(value, "root")
    return {"sha256": digest.hexdigest(), "tensors": tensors}


class Recorder:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.events: list[dict[str, Any]] = []
        self.update_index = 0

    def add(self, event: str, **fields: Any) -> None:
        self.events.append({"event": event, **fields})
        self.flush()

    def flush(self) -> None:
        (self.output / "fingerprints.json").write_text(
            json.dumps(self.events, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def sample_indices(seed: int, count: int) -> list[int]:
    # Match scripts/online/benchmark/benchmark_train.py exactly.
    # AdvertiseGen currently has more records than this probe needs; the upper
    # bound is checked by scripts.base.train when the dataset is loaded.
    indices = list(range(114599))
    random.Random(seed).shuffle(indices)
    return indices[:count]


async def probe(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    indices = sample_indices(args.seed, args.minibatches)
    (output / "sample_indices.json").write_text(
        json.dumps(indices, indent=2) + "\n", encoding="utf-8"
    )
    recorder = Recorder(output)
    original_reinit = Ptuning.reinit
    original_train_forward = Ptuning.train_forward
    original_train_backward = Ptuning.train_backward
    original_step = torch.optim.AdamW.step
    original_math_attention = operator_flags.USE_PYTORCH_MATH_ATTENTION
    original_forward = base_train.forward
    original_backward = base_train.backward
    stage_indices = {
        "adapter_forward_output": 0,
        "decoder_forward_output": 0,
        "decoder_backward_input": 0,
        "adapter_backward_input": 0,
        "adapter_gradient_after_backward": 0,
    }

    def stage(event: str, value: Any) -> None:
        index = stage_indices[event]
        stage_indices[event] += 1
        recorder.add(
            event,
            minibatch_index=index,
            fingerprint=tree_fingerprint(value),
        )

    def recorded_reinit(model: Ptuning, **kwargs: Any) -> None:
        original_reinit(model, **kwargs)
        torch.cuda.synchronize(torch.device(args.device))
        recorder.add(
            "adapter_initialized",
            adapter=tree_fingerprint(model.state_dict()),
        )

    def recorded_train_forward(
        model: Ptuning, dtype: torch.dtype, reproduce_seed: int | None,
    ):
        result = original_train_forward(model, dtype, reproduce_seed)
        stage("adapter_forward_output", result[1])
        return result

    def recorded_train_backward(
        model: Ptuning, ctx: Any, grad: list[torch.Tensor],
    ) -> None:
        stage("adapter_backward_input", grad)
        original_train_backward(model, ctx, grad)
        stage(
            "adapter_gradient_after_backward",
            [parameter.grad for parameter in model.parameters()],
        )

    async def recorded_forward(*call_args: Any, **call_kwargs: Any):
        result = await original_forward(*call_args, **call_kwargs)
        stage("decoder_forward_output", result[0])
        return result

    async def recorded_backward(
        container: Any, hidden_gradient: torch.Tensor, *call_args: Any,
        **call_kwargs: Any,
    ):
        stage("decoder_backward_input", hidden_gradient)
        return await original_backward(
            container, hidden_gradient, *call_args, **call_kwargs
        )

    def recorded_step(optimizer: torch.optim.AdamW, closure=None):
        torch.cuda.synchronize(torch.device(args.device))
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        before = {
            "parameters": tree_fingerprint(parameters),
            "gradients": tree_fingerprint([
                parameter.grad for parameter in parameters
            ]),
            "optimizer": tree_fingerprint(optimizer.state_dict()),
        }
        result = original_step(optimizer, closure)
        torch.cuda.synchronize(torch.device(args.device))
        after = {
            "parameters": tree_fingerprint(parameters),
            "optimizer": tree_fingerprint(optimizer.state_dict()),
        }
        recorder.add(
            "optimizer_step",
            update_index=recorder.update_index,
            before=before,
            after=after,
        )
        recorder.update_index += 1
        return result

    Ptuning.reinit = recorded_reinit
    Ptuning.train_forward = recorded_train_forward
    Ptuning.train_backward = recorded_train_backward
    torch.optim.AdamW.step = recorded_step
    base_train.forward = recorded_forward
    base_train.backward = recorded_backward
    operator_flags.USE_PYTORCH_MATH_ATTENTION = (
        args.attention_backend == "torch-math"
    )
    try:
        config = TrainConfig(
            user_model_config=PtuningConfig(
                build_prompt_str="[instruction]{}\n[generation]",
                dropout_p=0.0,
                prefix_len=args.prefix_len,
            ),
            decoder_backend=args.decoder_backend,
            batch_size=1,
            steps_in_update=args.grad_accum_steps,
            sample_indices=indices,
            submission_interval_ms=0.0,
            force_query_tokens=args.query_tokens,
            force_response_tokens=args.response_tokens,
            lr=args.learning_rate,
            eps=args.optimizer_eps,
            max_steps=args.minibatches,
            seed=args.seed,
            event_dir=str(output),
            ckpt_steps=0,
            best_ckpt_name="",
            record_steps=0,
            memory_steps=0,
            # The probe compares fingerprints and losses; a final checkpoint
            # adds hundreds of MiB across repeats without contributing evidence.
            final_ckpt_name="",
        )
        device = torch.device(args.device)
        stream = torch.cuda.Stream(device=device)
        container = NonAsyncContainer(stream)
        await base_train.run(config, container, device=container.device)
    finally:
        Ptuning.reinit = original_reinit
        Ptuning.train_forward = original_train_forward
        Ptuning.train_backward = original_train_backward
        torch.optim.AdamW.step = original_step
        base_train.forward = original_forward
        base_train.backward = original_backward
        operator_flags.USE_PYTORCH_MATH_ATTENTION = original_math_attention

    record = json.loads((output / "record.json").read_text(encoding="utf-8"))
    if len(record["history"]) != args.minibatches:
        raise RuntimeError("training probe did not complete every minibatch")
    if recorder.update_index != args.minibatches // args.grad_accum_steps:
        raise RuntimeError("training probe recorded the wrong update count")
    if any(value != args.minibatches for value in stage_indices.values()):
        raise RuntimeError(f"training probe recorded incomplete stages: {stage_indices}")
    (output / "probe_args.json").write_text(
        json.dumps(vars(args) | {"output": str(output)}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--minibatches", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--optimizer-eps", type=float, default=1e-4)
    parser.add_argument("--query-tokens", type=int, default=64)
    parser.add_argument("--response-tokens", type=int, default=64)
    parser.add_argument("--prefix-len", type=int, default=32)
    parser.add_argument(
        "--decoder-backend", choices=("exllama_fuse", "marlin"),
        default="exllama_fuse",
    )
    parser.add_argument(
        "--attention-backend", choices=("flash", "torch-math"),
        default="flash",
    )
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--async-backend", choices=("py", "chimera", "golang"), default="golang")
    parser.add_argument("--busy-wait", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.minibatches < 1 or args.minibatches % args.grad_accum_steps:
        parser.error("minibatches must be a positive multiple of grad accumulation")
    if min(args.query_tokens, args.response_tokens, args.prefix_len) < 1:
        parser.error("token and prefix lengths must be positive")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    asyncio.config_impl(parsed.async_backend, busy_wait=parsed.busy_wait)
    asyncio.run(probe(parsed))
