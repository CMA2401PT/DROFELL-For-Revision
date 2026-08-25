#!/usr/bin/env python3
"""Aggregate merge-limit and clean Golang busy-wait revision scans."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


EXP_ROOT = Path(__file__).resolve().parent


class AnalysisError(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dist(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        return {"count": 0, "mean": None, "median": None,
                "sample_sd": None, "min": None, "max": None}
    return {
        "count": len(data), "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "sample_sd": statistics.stdev(data) if len(data) > 1 else None,
        "min": min(data), "max": max(data),
    }


def load_case(run_id: str, case_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = EXP_ROOT / "results" / run_id
    case_path = root / case_id / "case.json"
    summary_path = root / "summary.json"
    if not case_path.is_file() or not summary_path.is_file():
        raise AnalysisError(f"missing case/summary for {run_id}:{case_id}")
    return root, read_json(case_path), read_json(summary_path)


def analyze_merge(run_ids: list[str], case_id: str, output: Path) -> None:
    rows = []
    details = []
    seen_limits = set()
    for run_id in run_ids:
        root, case, summary = load_case(run_id, case_id)
        params = case["parameters"]
        limit = int(params["merge_limit"])
        if limit in seen_limits:
            raise AnalysisError(f"duplicate merge limit: {limit}")
        seen_limits.add(limit)
        if case.get("async_backend") != "golang" or not case.get("busy_wait"):
            raise AnalysisError(f"not a golang busy-wait case: {run_id}")
        if int(params["clients"]) != 16 or case.get("consolidation"):
            raise AnalysisError(f"merge scan must use c0 and 16 clients: {run_id}")
        case_summary = summary["cases"][case_id]
        throughputs = [
            float(item["observed_tok_s"]) for item in case_summary["repeats"]
        ]
        counts: Counter[int] = Counter()
        with (root / "batch_summary.csv").open(encoding="utf-8", newline="") as file:
            for item in csv.DictReader(file):
                if item["case"] != case_id or item["operation"] != "decoder_generating":
                    continue
                counts.update({
                    int(size): int(count)
                    for size, count in json.loads(item["size_counts"]).items()
                })
        if not counts:
            raise AnalysisError(f"no decoder batch_merge events: {run_id}")
        if max(counts) > limit:
            raise AnalysisError(f"actual merge size exceeds limit: {run_id}")
        total = sum(counts.values())
        mean_batch = sum(size * count for size, count in counts.items()) / total
        row = {
            "merge_limit": limit,
            "repeat_count": len(throughputs),
            "throughput_mean_tok_s": statistics.fmean(throughputs),
            "throughput_median_tok_s": statistics.median(throughputs),
            "throughput_sample_sd_tok_s": (
                statistics.stdev(throughputs) if len(throughputs) > 1 else None
            ),
            "decoder_merge_event_count": total,
            "decoder_actual_mean_size": mean_batch,
            "decoder_fraction_at_limit": counts[limit] / total,
            "decoder_actual_size_counts": json.dumps(
                dict(sorted(counts.items())), sort_keys=True
            ),
            "run_id": run_id,
        }
        rows.append(row)
        details.append({"run_id": run_id, "case": case, "summary": case_summary})
    if seen_limits != {1, 2, 4, 8, 16}:
        raise AnalysisError(f"merge scan limits incomplete: {sorted(seen_limits)}")
    rows.sort(key=lambda item: int(item["merge_limit"]))
    write_csv(output / "merge_limit_summary.csv", rows)
    write_json(output / "merge_limit_summary.json", {
        "schema_version": 1,
        "protocol": (
            "c0-m1-d1, 16 clients, Golang busy-wait, three repeats. Only "
            "Brain max_concurrency/CUDA graph batch set changes. Upstream "
            "consolidation is disabled so its cap=8 cannot clip limit=16."
        ),
        "rows": rows,
        "sources": details,
    })
    print(json.dumps({"merge_rows": len(rows), "output": str(output)}, indent=2))


def analyze_busy(run_ids: list[str], case_id: str, output: Path) -> None:
    load_rows = []
    idle_by_mode: dict[bool, list[dict[str, Any]]] = {True: [], False: []}
    seen = set()
    sources = []
    for run_id in run_ids:
        root, case, summary = load_case(run_id, case_id)
        params = case["parameters"]
        busy = bool(case["busy_wait"])
        clients = int(params["clients"])
        key = (busy, clients)
        if key in seen:
            raise AnalysisError(f"duplicate busy/load cell: {key}")
        seen.add(key)
        if case.get("async_backend") != "golang":
            raise AnalysisError(f"backend changed in busy scan: {run_id}")
        if int(params["merge_limit"]) != 8:
            raise AnalysisError(f"merge limit changed in busy scan: {run_id}")
        idle_path = root / case_id / "idle_process_metrics.json"
        if not idle_path.is_file():
            raise AnalysisError(f"missing idle metrics: {run_id}")
        idle_by_mode[busy].append(read_json(idle_path))
        case_summary = summary["cases"][case_id]
        throughput = []
        cpu_cores = []
        switch_rates = []
        voluntary_rates = []
        involuntary_rates = []
        for repeat in case_summary["repeats"]:
            index = int(repeat["repeat"])
            metrics_path = root / case_id / f"repeat-{index}" / "process_metrics.json"
            metrics = read_json(metrics_path)
            wall = float(metrics["wall_seconds"])
            throughput.append(float(repeat["observed_tok_s"]))
            cpu_cores.append(float(metrics["average_cpu_cores"]))
            voluntary = int(metrics["voluntary_context_switches"])
            involuntary = int(metrics["involuntary_context_switches"])
            voluntary_rates.append(voluntary / wall)
            involuntary_rates.append(involuntary / wall)
            switch_rates.append((voluntary + involuntary) / wall)
        load_rows.append({
            "busy_wait": busy,
            "load": "ordinary" if clients == 1 else "high",
            "clients": clients,
            "repeat_count": len(throughput),
            "throughput_mean_tok_s": statistics.fmean(throughput),
            "throughput_sample_sd_tok_s": (
                statistics.stdev(throughput) if len(throughput) > 1 else None
            ),
            "cpu_cores_mean": statistics.fmean(cpu_cores),
            "cpu_cores_sample_sd": (
                statistics.stdev(cpu_cores) if len(cpu_cores) > 1 else None
            ),
            "scheduler_switches_per_s_mean": statistics.fmean(switch_rates),
            "voluntary_switches_per_s_mean": statistics.fmean(voluntary_rates),
            "involuntary_switches_per_s_mean": statistics.fmean(involuntary_rates),
            "run_id": run_id,
        })
        sources.append({"run_id": run_id, "case": case})
    if seen != {(True, 1), (False, 1), (True, 16), (False, 16)}:
        raise AnalysisError(f"busy scan cells incomplete: {sorted(seen)}")
    idle_rows = []
    for busy, metrics_list in idle_by_mode.items():
        core_values = [float(item["average_cpu_cores"]) for item in metrics_list]
        switch_values = [
            (int(item["voluntary_context_switches"]) +
             int(item["involuntary_context_switches"])) /
            float(item["wall_seconds"])
            for item in metrics_list
        ]
        idle_rows.append({
            "busy_wait": busy, "load": "idle", "clients": 0,
            "repeat_count": len(metrics_list),
            "throughput_mean_tok_s": None,
            "throughput_sample_sd_tok_s": None,
            "cpu_cores_mean": statistics.fmean(core_values),
            "cpu_cores_sample_sd": (
                statistics.stdev(core_values) if len(core_values) > 1 else None
            ),
            "scheduler_switches_per_s_mean": statistics.fmean(switch_values),
            "voluntary_switches_per_s_mean": None,
            "involuntary_switches_per_s_mean": None,
            "run_id": ";".join(
                run_id for run_id in run_ids
                if bool(load_case(run_id, case_id)[1]["busy_wait"]) == busy
            ),
        })
    rows = sorted(idle_rows + load_rows,
                  key=lambda item: (str(item["load"]), not bool(item["busy_wait"])))
    write_csv(output / "busy_wait_summary.csv", rows)
    write_json(output / "busy_wait_summary.json", {
        "schema_version": 1,
        "protocol": (
            "Single-variable comparison: both conditions use the Golang "
            "backend, identical c1-m1-d1 patches/topology/merge limit, and "
            "differ only in --busy-wait. Loaded-idle is measured for 30 s; "
            "ordinary/high use 1/16 clients and three repeats."
        ),
        "measurement_limit": (
            "Linux scheduler context switches are an OS proxy and are not an "
            "instrumented event-loop wakeup count. CPU RAPL energy is not "
            "included unless readable evidence is added separately."
        ),
        "rows": rows,
        "sources": sources,
    })
    print(json.dumps({"busy_rows": len(rows), "output": str(output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--run-ids", nargs="+", required=True)
    merge.add_argument("--case", default="c0-m1-d1")
    merge.add_argument("--output", type=Path, required=True)
    busy = subparsers.add_parser("busy")
    busy.add_argument("--run-ids", nargs="+", required=True)
    busy.add_argument("--case", default="c1-m1-d1")
    busy.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise AnalysisError(f"refusing to overwrite: {args.output}")
    if args.mode == "merge":
        analyze_merge(args.run_ids, args.case, args.output)
    else:
        analyze_busy(args.run_ids, args.case, args.output)


if __name__ == "__main__":
    main()
