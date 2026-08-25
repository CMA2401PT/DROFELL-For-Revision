#!/usr/bin/env python3
"""Summarize only complete probability cells from interrupted formal runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from run import (
    EXP_ROOT,
    PAPER_PROBABILITIES,
    TOPOLOGIES,
    ExperimentError,
    summarize_run,
    write_json,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["keypoint_coverage"] = json.dumps(row["keypoint_coverage"])
            writer.writerow(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--probabilities", type=float, nargs="+",
        default=(0.0, 1e-4, 1e-3, 1e-2),
    )
    args = parser.parse_args()
    if any(value not in PAPER_PROBABILITIES for value in args.probabilities):
        parser.error("--probabilities must be a subset of the paper grid")
    if not args.batch_id or "/" in args.batch_id or args.batch_id in {".", ".."}:
        parser.error("--batch-id must be one path component")
    output = EXP_ROOT / "results" / args.batch_id
    if output.exists():
        raise ExperimentError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)

    all_runs: list[dict[str, Any]] = []
    all_aggregates: list[dict[str, Any]] = []
    sources = []
    by_name = {topology.name: topology for topology in TOPOLOGIES}
    for run_id in args.run_ids:
        result_root = EXP_ROOT / "results" / run_id
        environment_path = EXP_ROOT / "environment" / f"{run_id}.json"
        if not result_root.is_dir() or not environment_path.is_file():
            raise ExperimentError(f"missing result/environment for {run_id}")
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        topology_names = [
            str(item["name"])
            for item in environment["parameters"]["topologies"]
            if (result_root / "combined-request-agent" / str(item["name"])).is_dir()
        ]
        topologies = tuple(
            replace(by_name[name], probabilities=tuple(args.probabilities))
            for name in topology_names
        )
        filtered_network_runs = []
        allowed = {format(value, ".8g") for value in args.probabilities}
        for network_run in environment.get("network_runs", []):
            item = dict(network_run)
            item["probabilities"] = {
                key: value
                for key, value in network_run.get("probabilities", {}).items()
                if key in allowed
            }
            filtered_network_runs.append(item)
        summary = summarize_run(
            result_root,
            filtered_network_runs,
            repeats=args.repeats,
            topologies=topologies,
            include_persistent_p0=False,
            capture_metrics=False,
        )
        all_runs.extend(summary["runs"])
        all_aggregates.extend(summary["aggregates"])
        sources.append({
            "run_id": run_id,
            "environment": str(environment_path.relative_to(EXP_ROOT)),
            "topologies": topology_names,
            "selected_probabilities": list(args.probabilities),
        })

    expected = len({row["topology"] for row in all_runs}) * len(
        set(args.probabilities)
    ) * args.repeats
    if len(all_runs) != expected:
        raise ExperimentError(
            f"combined row count {len(all_runs)} does not match expected {expected}"
        )
    summary = {
        "schema_version": 1,
        "batch_id": args.batch_id,
        "protocol": (
            "Only explicitly selected, complete probability cells are "
            "summarized from interrupted fixed-run-id experiments. Partial "
            "higher-probability directories are excluded, not deleted."
        ),
        "sources": sources,
        "repeats": args.repeats,
        "probabilities": list(args.probabilities),
        "runs": all_runs,
        "aggregates": all_aggregates,
    }
    write_json(output / "summary.json", summary)
    write_csv(output / "summary.csv", all_runs)
    print(json.dumps({
        "output": str(output),
        "run_rows": len(all_runs),
        "aggregate_rows": len(all_aggregates),
    }, indent=2))


if __name__ == "__main__":
    main()
