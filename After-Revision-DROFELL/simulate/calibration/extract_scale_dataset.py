#!/usr/bin/env python3
"""Build an auditable clean-scale calibration/holdout dataset.

Unlike the legacy p16/p18 probes, this extractor reads each client's raw output
timestamps. It never reconstructs ITL as ``clients / aggregate_throughput`` and
never borrows variance from another experiment. The split is predeclared here:

* calibration: combined-request-agent, clients {1, 4, 16, 64}
* primary holdout: combined-request-agent, clients {2, 8, 32, 128}
* external holdout: every persistent-agent cell

There is currently one independent hardware run per cell. Block-bootstrap
intervals quantify within-run temporal uncertainty, not run-to-run variance;
the output states that limitation and remains paper-ineligible until repeated
hardware runs exist or the paper explicitly limits the validation claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_ROOT = Path(
    "一稿返修/整理后就绪的实验/推理扩展性(带规模实验)"
    "/AutoDL8卡结果"
)
CALIBRATION_CLIENTS = {1, 4, 16, 64}
PRIMARY_HOLDOUT_CLIENTS = {2, 8, 32, 128}
FIXED_WINDOW_OUTPUT_TOKENS = 256


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty data")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def distribution(values: list[float]) -> dict[str, float | int]:
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


def block_bootstrap_ci(
    values: list[float],
    *,
    statistic: str,
    block_size: int = 128,
    samples: int = 500,
    seed: int = 20260822,
) -> list[float]:
    """Moving-block bootstrap CI for one long correlated token timeline."""
    if not values:
        return [float("nan"), float("nan")]
    size = min(block_size, len(values))
    blocks = [values[i:i + size] for i in range(0, len(values), size)]
    # Resample block statistics rather than repeatedly materialising the full
    # 35k-interval timeline. For the mean this is exact apart from a possibly
    # shorter final block; for p50 it is the standard median-of-blocks robust
    # approximation and is labelled as within-run uncertainty only.
    block_values = [
        statistics.fmean(block)
        if statistic == "mean"
        else statistics.median(block)
        for block in blocks
    ]
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(block_values) for _ in block_values]
        estimates.append(
            statistics.fmean(sampled)
            if statistic == "mean"
            else statistics.median(sampled)
        )
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for(deployment: str, clients: int) -> str:
    if deployment == "persistent-agent":
        return "external_holdout"
    if clients in CALIBRATION_CLIENTS:
        return "calibration"
    if clients in PRIMARY_HOLDOUT_CLIENTS:
        return "primary_holdout"
    raise ValueError(f"client count {clients} is not assigned to a split")


def cell(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root)
    scale = int(relative.parts[0].removeprefix("scale-"))
    clients = int(relative.parts[1].removeprefix("clients-"))
    deployment = relative.parts[2]
    phases_path = path / "phases.json"
    case_path = path / "case.json"
    result_path = path / "result.json"
    graph_path = path / "decode_graph.json"
    phases = json.loads(phases_path.read_text())
    case = json.loads(case_path.read_text())
    graph = json.loads(graph_path.read_text())
    warmup = int(phases["parameters"]["warmup_tokens"])
    raw = json.loads(result_path.read_text())
    per_client: dict[int, list[int]] = {}
    for client_id, timestamp_ns in raw:
        per_client.setdefault(int(client_id), []).append(int(timestamp_ns))

    interval_groups: list[list[float]] = []
    full_interval_groups: list[list[float]] = []
    anchors: list[int] = []
    ends: list[int] = []
    for client_id in sorted(per_client):
        stamps = sorted(per_client[client_id])
        if len(stamps) < FIXED_WINDOW_OUTPUT_TOKENS:
            raise ValueError(
                f"{relative}: client {client_id} has {len(stamps)} outputs, "
                f"not enough for fixed window={FIXED_WINDOW_OUTPUT_TOKENS}"
            )
        anchor = warmup - 1
        fixed_stamps = stamps[:FIXED_WINDOW_OUTPUT_TOKENS]
        interval_groups.append([
            (right - left) / 1e6
            for left, right in zip(
                fixed_stamps[anchor:], fixed_stamps[anchor + 1:]
            )
        ])
        full_interval_groups.append([
            (right - left) / 1e6
            for left, right in zip(stamps[anchor:], stamps[anchor + 1:])
        ])
        anchors.append(fixed_stamps[anchor])
        ends.append(fixed_stamps[-1])
    intervals = [value for group in interval_groups for value in group]
    full_intervals = [
        value for group in full_interval_groups for value in group
    ]
    stats = distribution(intervals)
    measured_intervals = sum(len(group) for group in interval_groups)
    fixed_window_seconds = (max(ends) - min(anchors)) / 1e9
    graph_replays_snapshot = sum(
        int(stats["replays"]) for stats in graph.values()
    )
    # The network process persists while clients are executed in descending
    # order, so this counter is cumulative. `main()` differences adjacent
    # snapshots before deriving the batch size. The measured run contributes
    # `actual_total_output_token_budget` decode steps; warm-up contributes
    # warmup-1 graph decodes per client because its first output is prefill.
    graph_work_items = (
        int(case["actual_total_output_token_budget"])
        + clients * (warmup - 1)
    ) * scale
    return {
        "cell_id": f"L{scale}-c{clients}-{deployment}",
        "segments": scale,
        "clients": clients,
        "deployment": deployment,
        "split": split_for(deployment, clients),
        "output_tokens_per_client": int(case["output_tokens_per_client"]),
        "warmup_tokens": warmup,
        "fixed_window_output_tokens": FIXED_WINDOW_OUTPUT_TOKENS,
        "independent_runs": 1,
        "itl_ms": stats,
        "within_run_block_bootstrap_95ci": {
            "mean": block_bootstrap_ci(intervals, statistic="mean"),
            "p50": block_bootstrap_ci(intervals, statistic="p50"),
            "block_size_intervals": min(128, len(intervals)),
            "samples": 500,
        },
        "fixed_window_tokens_per_second": (
            measured_intervals / fixed_window_seconds
        ),
        "decode_graph": {
            "cumulative_replays_snapshot": graph_replays_snapshot,
            "inferred_work_items_including_warmup": graph_work_items,
            "case_replays": None,
            "observed_mean_merged_batch": None,
            "derivation": (
                "(measured decode items + clients*(warmup-1)) * segments / "
                "delta(sum(per-Dealer cumulative graph replays)); first "
                "warm-up output is prefill and excluded"
            ),
        },
        "full_run": {
            "itl_ms": distribution(full_intervals),
            "overall_tokens_per_second": (
                int(case["actual_total_output_token_budget"])
                / (
                    (int(phases["end_ns"]) - int(phases["prefill_end_ns"]))
                    / 1e9
                )
            ),
            "warning": (
                "not comparable across client counts because output tokens "
                "per client vary with the fixed total-token budget"
            ),
        },
        "source": {
            "directory": str(relative),
            "result_sha256": sha256(result_path),
            "phases_sha256": sha256(phases_path),
            "case_sha256": sha256(case_path),
            "decode_graph_sha256": sha256(graph_path),
        },
    }


def iter_cells(
    root: Path, segments: Optional[set[int]] = None
) -> Iterable[Path]:
    paths = sorted(
        path.parent
        for path in root.glob("scale-*/clients-*/*/result.json")
        if path.parent.name in {"combined-request-agent", "persistent-agent"}
    )
    if segments is None:
        return paths
    return [
        path for path in paths
        if int(path.relative_to(root).parts[0].removeprefix("scale-"))
        in segments
    ]


def derive_case_replays(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (int(row["segments"]), str(row["deployment"])), []
        ).append(row)
    for group in grouped.values():
        previous = 0
        # exps/infer_scale/run.py executes descending concurrency while one
        # network process remains alive.
        for row in sorted(group, key=lambda item: int(item["clients"]), reverse=True):
            snapshot = int(
                row["decode_graph"]["cumulative_replays_snapshot"]
            )
            case_replays = snapshot - previous
            if case_replays <= 0:
                raise ValueError(
                    f"non-positive graph replay delta at {row['cell_id']}"
                )
            previous = snapshot
            row["decode_graph"]["case_replays"] = case_replays
            row["decode_graph"]["observed_mean_merged_batch"] = (
                int(row["decode_graph"]["inferred_work_items_including_warmup"])
                / case_replays
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--segments", nargs="+", type=int,
        help="include only these measured segment/GPU counts",
    )
    args = parser.parse_args()
    requested_segments = (
        set(args.segments) if args.segments is not None else None
    )
    rows = [
        cell(args.root, path)
        for path in iter_cells(args.root, requested_segments)
    ]
    if not rows:
        raise SystemExit("no matching scale cells found")
    included_segments = sorted({int(row["segments"]) for row in rows})
    if requested_segments is not None:
        missing = sorted(requested_segments - set(included_segments))
        if missing:
            raise SystemExit(f"requested segments not found: {missing}")
    derive_case_replays(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    report = {
        "schema_version": 1,
        "dataset_id": (
            "autodl8-clean-scale-fixed-context-v2"
            if requested_segments is None
            else "autodl-clean-scale-L"
            + "-L".join(str(value) for value in included_segments)
            + "-fixed-context-v3"
        ),
        "hardware_id": "autodl-8gpu-host-2026-08",
        "source_root": str(args.root),
        "included_segments": included_segments,
        "split_policy": {
            "calibration": (
                "combined-request-agent clients in {1,4,16,64}, included L"
            ),
            "primary_holdout": (
                "combined-request-agent clients in {2,8,32,128}, included L"
            ),
            "external_holdout": "all persistent-agent cells",
        },
        "split_counts": counts,
        "independent_runs_per_cell": 1,
        "fixed_window_output_tokens": FIXED_WINDOW_OUTPUT_TOKENS,
        "paper_eligible": False,
        "limitation": (
            "bootstrap intervals quantify temporal variation inside one long "
            "run; they are not run-to-run confidence intervals"
        ),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}: {len(rows)} cells, splits={counts}")
    print("paper_eligible=false: only one independent hardware run per cell")


if __name__ == "__main__":
    main()
