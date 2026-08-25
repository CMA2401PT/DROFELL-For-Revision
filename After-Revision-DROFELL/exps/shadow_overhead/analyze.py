#!/usr/bin/env python3
"""Audit and summarize shadow-state snapshots from formal recovery runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = Path(__file__).resolve().parent
RECOVERY_ROOT = REPO_ROOT / "exps" / "recover-analysis-infer" / "results"
DEFAULT_SOURCE_RUNS = (
    "nonblocking-causal-ordinary-f8-r3-v5-20260814",
    "multipath-pause-free-f4-r3-v1-20260814",
    "multipath-clean-reprefill-f1-r3-v2-20260815",
)
METRICS = (
    "metadata_bytes",
    "prompt_bytes",
    "generating_bytes",
    "pending_task_bytes",
    "kv_bytes",
    "total_retained_bytes",
    "update_cpu_ns",
    "task_input_bytes_observed",
    "kv_bytes_received",
    "replay_task_count",
    "checkpoint_block_count",
)


class AnalysisError(RuntimeError):
    pass


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "sample_sd": None,
                "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def collect(source_runs: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for run_id in source_runs:
        run_root = RECOVERY_ROOT / run_id
        if not run_root.is_dir():
            raise AnalysisError(f"source run does not exist: {run_root}")
        files = sorted(run_root.glob("**/runtime_metrics.jsonl"))
        if not files:
            raise AnalysisError(f"source run has no runtime metrics: {run_root}")
        event_count = 0
        source_files = []
        for path in files:
            relative = path.relative_to(run_root)
            parts = relative.parts
            case = parts[0] if parts else "unknown"
            repeat = next(
                (part.removeprefix("repeat-") for part in parts
                 if part.startswith("repeat-")),
                "unknown",
            )
            file_events = 0
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise AnalysisError(f"invalid JSONL: {path}:{line_number}") from error
                if event.get("event") != "shadow_state_snapshot":
                    continue
                metrics = event.get("metrics")
                if not isinstance(metrics, dict):
                    raise AnalysisError(f"snapshot missing metrics: {path}:{line_number}")
                missing = [name for name in METRICS if name not in metrics]
                if missing:
                    raise AnalysisError(
                        f"snapshot missing {missing}: {path}:{line_number}"
                    )
                scope = str(event.get("recovery_scope", "agent"))
                row = {
                    "source_run": run_id,
                    "case": case,
                    "repeat": repeat,
                    "scope": scope,
                    "model": "teared-Llama-2-7B-GPTQ",
                    # All selected source runs use four 8-layer groups, while
                    # the failed ordinary handle owns the full 32-layer span.
                    "decoder_group_layers": 8,
                    "failed_handle_layers": 32,
                    "retrieve_mode": "mix" if "mix" in case else "reprefill",
                    "recovery_id": event.get("recovery_id"),
                    "compute_node": event.get("compute_node"),
                    "runtime_metrics_path": str(path.relative_to(REPO_ROOT)),
                    "line_number": line_number,
                }
                row.update({name: int(metrics[name]) for name in METRICS})
                rows.append(row)
                event_count += 1
                file_events += 1
            source_files.append({
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256(path),
                "snapshot_events": file_events,
            })
        if event_count == 0:
            raise AnalysisError(f"source run has no shadow snapshots: {run_root}")
        sources.append({
            "run_id": run_id,
            "snapshot_events": event_count,
            "runtime_metric_files": source_files,
        })
    return rows, sources


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["scope"], row["retrieve_mode"], row["replay_task_count"],
            row["checkpoint_block_count"], row["failed_handle_layers"],
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, items in sorted(groups.items(), key=lambda item: item[0]):
        scope, mode, replay, blocks, layers = key
        output.append({
            "scope": scope,
            "retrieve_mode": mode,
            "replay_task_count": replay,
            "checkpoint_block_count": blocks,
            "failed_handle_layers": layers,
            "snapshot_count": len(items),
            **{
                name: distribution([int(item[name]) for item in items])
                for name in METRICS
                if name not in {"replay_task_count", "checkpoint_block_count"}
            },
        })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--source-runs", nargs="+", default=DEFAULT_SOURCE_RUNS)
    args = parser.parse_args()
    output = EXP_ROOT / "results" / args.run_id
    if output.exists():
        raise AnalysisError(f"refusing to overwrite existing output: {output}")
    rows, sources = collect(tuple(args.source_runs))
    groups = aggregate(rows)
    write_csv(output / "snapshots.csv", rows)
    flat_groups = []
    for group in groups:
        flat = {
            key: value for key, value in group.items()
            if not isinstance(value, dict)
        }
        for metric, stats in group.items():
            if not isinstance(stats, dict):
                continue
            for stat, value in stats.items():
                flat[f"{metric}_{stat}"] = value
        flat_groups.append(flat)
    write_csv(output / "summary.csv", flat_groups)
    write_json(output / "summary.json", {
        "schema_version": 1,
        "run_id": args.run_id,
        "source_runs": sources,
        "snapshot_count": len(rows),
        "model": "teared-Llama-2-7B-GPTQ",
        "decoder_group_layers": 8,
        "failed_handle_layers": 32,
        "aggregates": groups,
        "scope_limit": {
            "covered": (
                "Existing formal online runs; 7B only; four 8-layer decoder "
                "groups; failed ordinary handle spans 32 layers; observed "
                "recovery windows and retrieve/reprefill modes."
            ),
            "not_covered": (
                "No model-size or failed-handle shard-size variation. These "
                "results are evidence for the recovery-window axis only and "
                "must not be presented as the requested full 3-D scan."
            ),
        },
    })
    print(json.dumps({
        "output": str(output),
        "snapshots": len(rows),
        "aggregates": len(groups),
    }, indent=2))


if __name__ == "__main__":
    main()
