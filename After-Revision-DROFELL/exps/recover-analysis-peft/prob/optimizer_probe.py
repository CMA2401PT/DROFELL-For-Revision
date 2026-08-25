#!/usr/bin/env python3
"""Run AdamW on identical synthetic parameters and gradients."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().to("cpu")
    return hashlib.sha256(
        value.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--optimizer-eps", type=float, default=1e-4)
    parser.add_argument("--elements", type=int, default=2**20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    initial = torch.randn(args.elements, generator=generator, dtype=torch.float32)
    gradients = [
        torch.randn(args.elements, generator=generator, dtype=torch.float32)
        for _ in range(args.updates)
    ]
    device = torch.device(args.device)
    parameter = torch.nn.Parameter(initial.to(device))
    optimizer = torch.optim.AdamW(
        [parameter], lr=args.learning_rate, eps=args.optimizer_eps
    )
    events = [{"event": "initial", "parameter": digest(parameter)}]
    for update, gradient in enumerate(gradients):
        parameter.grad = gradient.to(device)
        events.append({
            "event": "before_step",
            "update_index": update,
            "parameter": digest(parameter),
            "gradient": digest(parameter.grad),
        })
        optimizer.step()
        optimizer.zero_grad()
        state = optimizer.state[parameter]
        events.append({
            "event": "after_step",
            "update_index": update,
            "parameter": digest(parameter),
            "exp_avg": digest(state["exp_avg"]),
            "exp_avg_sq": digest(state["exp_avg_sq"]),
            "step": float(state["step"].item()),
        })
    (args.output / "fingerprints.json").write_text(
        json.dumps(events, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "probe_args.json").write_text(
        json.dumps(vars(args) | {"output": str(args.output)}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
