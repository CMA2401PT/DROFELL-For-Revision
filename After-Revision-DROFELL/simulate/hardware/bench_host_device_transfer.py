#!/usr/bin/env python3
"""Measure isolated and concurrent H2D/D2H for real hidden-state tensors.

The pageable path matches DROFELL's current TCP receive and `.to(...,
non_blocking=True)` behavior. Optional pinned results are reported separately,
not substituted for the production path. CUDA Events measure stream time while
`enqueue_cpu_ns` records Python/allocator submission cost.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch


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


def one_direction(
    *,
    device_index: int,
    batch: int,
    hidden_size: int,
    direction: str,
    pinned: bool,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    stream = torch.cuda.Stream(device=device)
    cpu = torch.empty(
        (batch, hidden_size), dtype=torch.float16,
        device="cpu", pin_memory=pinned,
    )
    gpu = torch.empty((batch, hidden_size), dtype=torch.float16, device=device)
    event_ms: list[float] = []
    enqueue_ns: list[float] = []
    result_cpu_pinned: bool | None = None
    total = warmup + iterations
    for index in range(total):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            cpu_started = time.perf_counter_ns()
            result = (
                gpu.to(device="cpu", non_blocking=True)
                if direction == "d2h"
                else cpu.to(device=device, non_blocking=True)
            )
            cpu_elapsed = time.perf_counter_ns() - cpu_started
            end.record(stream)
        end.synchronize()
        if direction == "d2h":
            result_cpu_pinned = bool(result.is_pinned())
        if index >= warmup:
            event_ms.append(start.elapsed_time(end))
            enqueue_ns.append(float(cpu_elapsed))
        del result
    size = batch * hidden_size * 2
    return {
        "device": device_index,
        "direction": direction,
        "batch": batch,
        "hidden_size": hidden_size,
        "bytes": size,
        "source_memory": "pinned" if pinned else "pageable",
        "cuda_event_ms": summary(event_ms),
        "enqueue_cpu_ns": summary(enqueue_ns),
        "effective_gbps_at_p50": (
            size * 8 / (float(summary(event_ms)["p50"]) * 1e6)
        ),
        "result_cpu_pinned": result_cpu_pinned,
    }


def concurrent_direction(
    *,
    devices: list[int],
    batch: int,
    hidden_size: int,
    direction: str,
    pinned: bool,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    barrier = threading.Barrier(len(devices))

    def worker(device_index: int) -> dict[str, Any]:
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
        stream = torch.cuda.Stream(device=device)
        cpu = torch.empty(
            (batch, hidden_size), dtype=torch.float16,
            device="cpu", pin_memory=pinned,
        )
        gpu = torch.empty(
            (batch, hidden_size), dtype=torch.float16, device=device
        )
        event_ms: list[float] = []
        enqueue_ns: list[float] = []
        wall_ms: list[float] = []
        for index in range(warmup + iterations):
            barrier.wait()
            wall_started = time.perf_counter_ns()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record(stream)
                cpu_started = time.perf_counter_ns()
                result = (
                    gpu.to(device="cpu", non_blocking=True)
                    if direction == "d2h"
                    else cpu.to(device=device, non_blocking=True)
                )
                cpu_elapsed = time.perf_counter_ns() - cpu_started
                end.record(stream)
            end.synchronize()
            wall_elapsed = (time.perf_counter_ns() - wall_started) / 1e6
            if index >= warmup:
                event_ms.append(start.elapsed_time(end))
                enqueue_ns.append(float(cpu_elapsed))
                wall_ms.append(wall_elapsed)
            del result
        return {
            "device": device_index,
            "cuda_event_ms": summary(event_ms),
            "enqueue_cpu_ns": summary(enqueue_ns),
            "wall_ms": summary(wall_ms),
        }

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        per_device = list(pool.map(worker, devices))
    size = batch * hidden_size * 2
    slowest_p50_ms = max(
        float(row["wall_ms"]["p50"]) for row in per_device
    )
    return {
        "devices": devices,
        "direction": direction,
        "batch": batch,
        "hidden_size": hidden_size,
        "bytes_per_device": size,
        "source_memory": "pinned" if pinned else "pageable",
        "per_device": per_device,
        "aggregate_effective_gbps_p50": (
            size * len(devices) * 8 / (slowest_p50_ms * 1e6)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--batches", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--include-pinned", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = {
        "devices": args.devices,
        "batches": args.batches,
        "hidden_size": args.hidden_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "memory_modes": ["pageable"] + (["pinned"] if args.include_pinned else []),
    }
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    missing = [index for index in args.devices if index >= torch.cuda.device_count()]
    if missing:
        raise SystemExit(f"CUDA devices not present: {missing}")

    isolated = []
    concurrent = []
    for pinned in (False, True) if args.include_pinned else (False,):
        for batch in args.batches:
            for direction in ("h2d", "d2h"):
                for device in args.devices:
                    isolated.append(one_direction(
                        device_index=device,
                        batch=batch,
                        hidden_size=args.hidden_size,
                        direction=direction,
                        pinned=pinned,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    ))
                concurrent.append(concurrent_direction(
                    devices=args.devices,
                    batch=batch,
                    hidden_size=args.hidden_size,
                    direction=direction,
                    pinned=pinned,
                    warmup=args.warmup,
                    iterations=args.iterations,
                ))
    report = {
        "schema_version": 1,
        "config": config,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_names": {
            str(index): torch.cuda.get_device_name(index)
            for index in args.devices
        },
        "isolated": isolated,
        "concurrent": concurrent,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
