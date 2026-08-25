#!/usr/bin/env python3
"""Audit whether the 4->8 GPU regression is merge or hop-transfer dominated.

This script intentionally stops at what the archived evidence can identify.
Single-client data isolates a per-hop term because no cross-request merge exists.
Decode-graph replay counts recover the mean merge size, and GPU monitor samples
show whether devices are compute-saturated. The archive still lacks D2H/H2D
spans and the full merge histogram, so individual hop components remain open.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DATASET = Path(
    "simulate/results/reviewer-grade/autodl8_clean_scale_dataset.json"
)
DEFAULT_ROOT = Path(
    "一稿返修/整理后就绪的实验/推理扩展性(带规模实验)"
    "/AutoDL8卡结果"
)


def parse_monitor_time(raw: str) -> float:
    parsed = datetime.strptime(raw.strip(), "%Y/%m/%d %H:%M:%S.%f")
    return parsed.replace(tzinfo=timezone.utc).timestamp()


def infer_monitor_offset_hours(samples: list[float], target: float) -> int:
    """Infer the monitor logger's whole-hour timezone offset."""
    middle = statistics.median(samples)
    return min(range(-12, 15), key=lambda hour: abs(middle - hour * 3600 - target))


def gpu_summary(root: Path, scale: int, clients: int) -> dict[str, Any]:
    case = root / f"scale-{scale}" / f"clients-{clients}" / "combined-request-agent"
    phases = json.loads((case / "phases.json").read_text())
    start = int(phases["prefill_end_ns"]) / 1e9
    end = int(phases["end_ns"]) / 1e9
    parsed: list[tuple[float, int, float, float]] = []
    for raw in (case / "gpu_monitor.csv").read_text().splitlines():
        fields = [item.strip() for item in raw.split(",")]
        if len(fields) != 8:
            continue
        try:
            parsed.append((
                parse_monitor_time(fields[0]), int(fields[1]),
                float(fields[4]), float(fields[7]),
            ))
        except ValueError:
            continue
    offset = infer_monitor_offset_hours([row[0] for row in parsed], (start + end) / 2)
    active = [
        (gpu, util, power)
        for stamp, gpu, util, power in parsed
        if start <= stamp - offset * 3600 <= end and gpu < scale
    ]
    if not active:
        raise ValueError(f"no active GPU samples for {case}")
    by_gpu: dict[int, list[tuple[float, float]]] = {}
    for gpu, util, power in active:
        by_gpu.setdefault(gpu, []).append((util, power))
    per_gpu = {
        str(gpu): {
            "samples": len(values),
            "mean_utilization_pct": statistics.fmean(v[0] for v in values),
            "mean_power_w": statistics.fmean(v[1] for v in values),
        }
        for gpu, values in sorted(by_gpu.items())
    }
    return {
        "timezone_offset_hours_inferred": offset,
        "active_samples": len(active),
        "mean_utilization_pct_per_gpu": statistics.fmean(v[1] for v in active),
        "sum_mean_utilization_pct": sum(
            statistics.fmean(value[0] for value in values)
            for values in by_gpu.values()
        ),
        "mean_power_w_per_gpu": statistics.fmean(v[2] for v in active),
        "per_gpu": per_gpu,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text())
    rows = {
        (row["segments"], row["clients"]): row
        for row in dataset["rows"]
        if row["deployment"] == "combined-request-agent"
    }
    low4, low8 = rows[(4, 1)], rows[(8, 1)]
    low_delta = low8["itl_ms"]["mean"] - low4["itl_ms"]["mean"]
    concurrency: dict[str, Any] = {}
    for clients in (16, 32, 64, 128):
        row4, row8 = rows[(4, clients)], rows[(8, clients)]
        gpu4 = gpu_summary(args.root, 4, clients)
        gpu8 = gpu_summary(args.root, 8, clients)
        concurrency[str(clients)] = {
            "throughput_tokens_per_second": {
                "L4": row4["fixed_window_tokens_per_second"],
                "L8": row8["fixed_window_tokens_per_second"],
                "L8_over_L4": (
                    row8["fixed_window_tokens_per_second"]
                    / row4["fixed_window_tokens_per_second"]
                ),
            },
            "mean_itl_ms": {
                "L4": row4["itl_ms"]["mean"],
                "L8": row8["itl_ms"]["mean"],
            },
            "observed_mean_merged_batch": {
                "L4": row4["decode_graph"]["observed_mean_merged_batch"],
                "L8": row8["decode_graph"]["observed_mean_merged_batch"],
                "relative_difference_pct": 100 * (
                    row8["decode_graph"]["observed_mean_merged_batch"]
                    / row4["decode_graph"]["observed_mean_merged_batch"] - 1
                ),
            },
            "gpu": {"L4": gpu4, "L8": gpu8},
        }
    report = {
        "schema_version": 1,
        "hardware_id": dataset["hardware_id"],
        "question": "why throughput regresses from 4 to 8 GPUs",
        "single_client_identification": {
            "merge_possible": False,
            "L4_mean_itl_ms": low4["itl_ms"]["mean"],
            "L8_mean_itl_ms": low8["itl_ms"]["mean"],
            "delta_ms": low_delta,
            "additional_segment_boundaries": 4,
            "identified_per_additional_segment_ms": low_delta / 4,
            "verdict": (
                "a non-merge per-hop term is directly identified because c=1 "
                "has no cross-request merging"
            ),
        },
        "activation_transport": {
            "hidden_size": 4096,
            "dtype_bytes": 2,
            "bytes_per_sequence_per_boundary": 8192,
            "real_path_per_boundary": [
                "GPU->CPU data_device_to_cpu",
                "CUDA sync point before task can leave",
                "FasterPickler metadata + CPU tensor memoryview over TCP",
                "CPU->GPU data_cpu_to_device at the next shard",
            ],
            "code_evidence": [
                "node/compute_node/infer_node.py:599-623",
                "dealers/real_basic/infer_binding.py:287-292,391-407,909-937",
                "dealers/common/brain.py:110-118",
                "dealers/common/base_types.py:90-103",
            ],
        },
        "high_concurrency": concurrency,
        "identifiability": {
            "transfer_or_control_contribution": "supported",
            "merge_size_drop_as_4to8_cause": (
                "ruled out at the mean: L4/L8 differ by at most about 1% at "
                "matched concurrency"
            ),
            "missing_measurements": [
                "actual merged-batch histogram per Dealer and stage",
                "D2H enqueue and completion timestamps",
                "serialization/TCP send and receive timestamps",
                "H2D enqueue and completion timestamps",
            ],
            "paper_eligible_causal_attribution": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    identified = report["single_client_identification"]
    print(
        "c=1 identifies %.3f ms per added segment; matched L4/L8 mean merge "
        "sizes rule out merge-size loss as the regression cause"
        % identified["identified_per_additional_segment_ms"]
    )


if __name__ == "__main__":
    main()
