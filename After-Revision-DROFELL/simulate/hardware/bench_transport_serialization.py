from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dealers.common.base_types import DeviceTensorCarrier, ModelDescribe
from node.compute_node.infer_meta import InferTask, Navigate
from node.dealer_node.protocols import GroupedInferTask, MultiPathInferTask
from orbit.link.faster_serialize.pickle import FasterPickler, FasterUnpickler


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def payload(batch: int, hidden_size: int) -> GroupedInferTask:
    model = "teared-Llama-2-7B-GPTQ"
    user_models = [
        str(ModelDescribe.new(model, start=0, end=8)),
        str(ModelDescribe.new(model, start=8, end=16)),
    ]
    tasks = []
    for index in range(batch):
        carrier = DeviceTensorCarrier(
            torch.empty((hidden_size,), dtype=torch.float16, device="cpu"),
            None,
        )
        task = InferTask(
            seq_id=f"bench-{index}",
            step=64,
            navi=Navigate(user_models=user_models),
            is_prefill=False,
            input=carrier,
            output=None,
        )
        tasks.append(MultiPathInferTask(
            task=task,
            resource_loc={0: ["dealer-0"], 1: ["dealer-1"]},
            trace=["dealer-0"],
        ))
    return GroupedInferTask(tasks=tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for batch in args.batches:
        value = payload(batch, args.hidden_size)
        serialize_ns: list[float] = []
        deserialize_ns: list[float] = []
        frame_sizes: list[int] | None = None
        for index in range(args.warmup + args.iterations):
            started = time.perf_counter_ns()
            frames, no_gc = FasterPickler.dumps(value)
            serialize_elapsed = time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            recovered, unused, consumed = FasterUnpickler.loads(frames, no_gc)
            deserialize_elapsed = time.perf_counter_ns() - started
            if unused or consumed != len(frames):
                raise RuntimeError("FasterUnpickler did not consume all frames")
            if not isinstance(recovered, GroupedInferTask):
                raise TypeError("round trip changed payload type")
            if index >= args.warmup:
                serialize_ns.append(float(serialize_elapsed))
                deserialize_ns.append(float(deserialize_elapsed))
            frame_sizes = [len(frame) for frame in frames]
        assert frame_sizes is not None
        rows.append({
            "batch": batch,
            "hidden_size": args.hidden_size,
            "logical_tensor_bytes": batch * args.hidden_size * 2,
            "frame_sizes": frame_sizes,
            "wire_bytes_without_length_prefix": sum(frame_sizes),
            "serialize_cpu_ns": summary(serialize_ns),
            "deserialize_cpu_ns": summary(deserialize_ns),
        })
        print(
            f"batch={batch}: serialize p50={rows[-1]['serialize_cpu_ns']['p50'] / 1e3:.1f}us "
            f"deserialize p50={rows[-1]['deserialize_cpu_ns']['p50'] / 1e3:.1f}us"
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "config": {
            "batches": args.batches,
            "hidden_size": args.hidden_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "dtype": "torch.float16",
            "cpu_memory": "pageable",
        },
        "torch_version": torch.__version__,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
