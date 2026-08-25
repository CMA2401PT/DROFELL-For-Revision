#!/usr/bin/env python3
"""Summarize and plot the infer-scale experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any

from scripts.online.benchmark.analysis_infer_scale import (
    discover_records,
    paper_speed_points,
    plot_deployment,
)


CHECKPOINTS = (512, 1000, 2000)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def integer_leaf_total(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return sum(integer_leaf_total(item) for item in value.values())
    if isinstance(value, list):
        return sum(integer_leaf_total(item) for item in value)
    return 0


def counter_total(value: Any, key: str) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, dict):
        return sum(
            integer_leaf_total(item) if name == key
            else counter_total(item, key)
            for name, item in value.items()
        )
    if isinstance(value, list):
        return sum(counter_total(item, key) for item in value)
    return 0


def monitor_summary(path: Path, scale: int) -> dict[str, Any]:
    peak_used: dict[int, float] = {}
    minimum_free: dict[int, float] = {}
    minimum_clock: dict[int, float] = {}
    samples = 0
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 8:
            continue
        try:
            index = int(parts[1])
            if index >= scale:
                continue
            used = float(parts[2])
            free = float(parts[3])
            clock = float(parts[6])
        except ValueError:
            continue
        peak_used[index] = max(peak_used.get(index, 0.0), used)
        minimum_free[index] = min(minimum_free.get(index, float("inf")), free)
        minimum_clock[index] = min(
            minimum_clock.get(index, float("inf")), clock
        )
        samples += 1
    return {
        "samples": samples,
        "peak_memory_used_mib": peak_used,
        "minimum_memory_free_mib": minimum_free,
        "minimum_sm_clock_mhz": minimum_clock,
    }


def nearest_speed(points: list[tuple[int, float]], target: int) -> float | None:
    if not points or target < points[0][0] or target > points[-1][0]:
        return None
    _, value = min(points, key=lambda item: abs(item[0] - target))
    return value


def speed_ratio(
    numerator: float | None,
    denominator: float | None,
) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def scale_comparison_fields(
    selected: dict[int, float | None],
) -> dict[str, float | None]:
    return {
        "scale_1_tokens_per_second": selected.get(1),
        "scale_2_tokens_per_second": selected.get(2),
        "scale_4_tokens_per_second": selected.get(4),
        "scale_8_tokens_per_second": selected.get(8),
        "scale_2_over_scale_1": speed_ratio(
            selected.get(2), selected.get(1)
        ),
        "scale_4_over_scale_1": speed_ratio(
            selected.get(4), selected.get(1)
        ),
        "scale_8_over_scale_1": speed_ratio(
            selected.get(8), selected.get(1)
        ),
        "scale_4_over_scale_2": speed_ratio(
            selected.get(4), selected.get(2)
        ),
        "scale_8_over_scale_4": speed_ratio(
            selected.get(8), selected.get(4)
        ),
    }


def has_scale_ratio(comparison: dict[str, float | None]) -> bool:
    return any(
        value is not None
        for name, value in comparison.items()
        if "_over_scale_" in name
    )


def summarize(run_dir: Path) -> dict[str, Any]:
    records = discover_records(run_dir)
    rows: list[dict[str, Any]] = []
    for (scale, clients, deployment), record in sorted(records.items()):
        case_dir = run_dir / f"scale-{scale}" / f"clients-{clients}" / deployment
        case = read_json(case_dir / "case.json")
        phases = read_json(case_dir / "phases.json")
        graph = read_json(case_dir / "decode_graph.json")
        expected_per_client = case["output_tokens_per_client"]
        expected_records = clients * expected_per_client
        counts = Counter(int(item[0]) for item in record)
        if len(record) != expected_records or counts != Counter({
            client: expected_per_client for client in range(clients)
        }):
            raise ValueError(f"incomplete token record: {case_dir}")
        if any(
            int(left[1]) > int(right[1])
            for left, right in zip(record, record[1:])
        ):
            raise ValueError(f"non-monotonic token timestamps: {case_dir}")
        if phases["parameters"]["disable_gc"]:
            raise ValueError(f"GC disabled in measured case: {case_dir}")
        if phases["parameters"]["logprobs"] != 0:
            raise ValueError(f"logprobs enabled in measured case: {case_dir}")
        for dealer, stats in graph.items():
            if not stats.get("enabled"):
                raise ValueError(f"graph disabled for {dealer}: {case_dir}")
            if stats.get("configured_batch_sizes") != list(range(1, 9)):
                raise ValueError(
                    f"unexpected graph batches for {dealer}: {case_dir}"
                )
        graph_fallbacks = counter_total(graph, "fallbacks")
        fused_replays = counter_total(graph, "fused_replays")
        if graph_fallbacks or fused_replays <= 0:
            raise ValueError(f"invalid graph execution: {case_dir}")
        # New runs distinguish the nominal fixed budget from the actual work
        # after integer division and the per-client cap.  Archived cases only
        # have total_output_token_budget, which already denotes actual work.
        actual_total_budget = case.get(
            "actual_total_output_token_budget",
            case["total_output_token_budget"],
        )
        nominal_total_budget = case.get(
            "nominal_total_output_token_budget",
            case["total_output_token_budget"],
        )
        if actual_total_budget != expected_records:
            raise ValueError(f"actual token budget mismatch: {case_dir}")
        cap = case.get("tokens_per_client_cap")
        if nominal_total_budget is not None and cap is not None:
            derived_per_client = min(nominal_total_budget // clients, cap)
            expected_capped = nominal_total_budget // clients > cap
            if expected_per_client != derived_per_client:
                raise ValueError(f"per-client token budget mismatch: {case_dir}")
            if case.get("capped", False) != expected_capped:
                raise ValueError(f"capped flag mismatch: {case_dir}")
        points = paper_speed_points(record)
        duration_seconds = (int(record[-1][1]) - int(record[0][1])) / 1e9
        overall = (len(record) - 1) / duration_seconds
        row: dict[str, Any] = {
            "scale": scale,
            "clients": clients,
            "deployment": deployment,
            "output_tokens_per_client": case["output_tokens_per_client"],
            "nominal_total_output_token_budget": nominal_total_budget,
            "actual_total_output_token_budget": actual_total_budget,
            "tokens_per_client_cap": case.get("tokens_per_client_cap"),
            "capped": case.get("capped", False),
            "recorded_tokens": len(record),
            "measurement_seconds": duration_seconds,
            "wall_seconds": case["wall_seconds"],
            "overall_tokens_per_second": overall,
            "smoothed_points": len(points),
            "graph_replays_snapshot": counter_total(graph, "replays"),
            "fused_replays_snapshot": fused_replays,
            "graph_fallbacks_snapshot": graph_fallbacks,
            "gc_enabled": not phases["parameters"]["disable_gc"],
            "logprobs": phases["parameters"]["logprobs"],
            "gpu": monitor_summary(case_dir / "gpu_monitor.csv", scale),
        }
        for checkpoint in CHECKPOINTS:
            row[f"current_tokens_per_second_at_{checkpoint}"] = nearest_speed(
                points, checkpoint
            )
        rows.append(row)
    comparisons: list[dict[str, Any]] = []
    for clients in sorted({row["clients"] for row in rows}):
        for deployment in sorted({row["deployment"] for row in rows}):
            for checkpoint in CHECKPOINTS:
                selected = {
                    row["scale"]: row[
                        f"current_tokens_per_second_at_{checkpoint}"
                    ]
                    for row in rows
                    if row["clients"] == clients
                    and row["deployment"] == deployment
                }
                comparison = scale_comparison_fields(selected)
                if not has_scale_ratio(comparison):
                    continue
                comparisons.append({
                    "clients": clients,
                    "deployment": deployment,
                    "token_position": checkpoint,
                    **comparison,
                })
    overall_comparisons: list[dict[str, Any]] = []
    for clients in sorted({row["clients"] for row in rows}):
        for deployment in sorted({row["deployment"] for row in rows}):
            selected = {
                row["scale"]: row["overall_tokens_per_second"]
                for row in rows
                if row["clients"] == clients
                and row["deployment"] == deployment
            }
            comparison = scale_comparison_fields(selected)
            if not has_scale_ratio(comparison):
                continue
            overall_comparisons.append({
                "clients": clients,
                "deployment": deployment,
                **comparison,
            })
    figures = run_dir / "figures"
    for deployment in sorted({item[2] for item in records}):
        plot_deployment(
            records,
            deployment=deployment,
            output=figures / f"infer-scale-{deployment}.png",
        )
    summary = {
        "run_dir": str(run_dir),
        "throughput_definition": (
            "all completions between adjacent client-0 token completions, "
            "followed by a centered three-point arithmetic mean"
        ),
        "cases": rows,
        "scale_comparisons": comparisons,
        "overall_scale_comparisons": overall_comparisons,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        "scale", "clients", "deployment", "output_tokens_per_client",
        "nominal_total_output_token_budget",
        "actual_total_output_token_budget", "tokens_per_client_cap", "capped",
        "recorded_tokens", "measurement_seconds", "overall_tokens_per_second",
        "current_tokens_per_second_at_512",
        "current_tokens_per_second_at_1000",
        "current_tokens_per_second_at_2000",
        "fused_replays_snapshot", "graph_fallbacks_snapshot",
    ]
    with (run_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fields})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.run_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
