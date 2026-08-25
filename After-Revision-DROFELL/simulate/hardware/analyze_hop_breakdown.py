#!/usr/bin/env python3
"""Validate and summarize hop-breakdown hardware runs across repeats."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_EVENTS = {
    "decoder_batch_selected",
    "device_to_host_enqueue",
    "host_to_device_enqueue",
    "operate_sync_wait",
    "transport_serialize",
    "transport_socket_write",
    "transport_deserialize",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None,
                "p99": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def mean_ci95(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    # Student-t critical values for the small repeat counts used here.
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
                5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
                9: 2.262}.get(len(values) - 1, 1.96)
    center = statistics.fmean(values)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [center - half, center + half]


def token_metrics(result_path: Path, warmup: int = 32) -> dict[str, Any]:
    raw = json.loads(result_path.read_text())
    per_client: dict[int, list[int]] = {}
    for client, timestamp in raw:
        per_client.setdefault(int(client), []).append(int(timestamp))
    intervals: list[float] = []
    anchors: list[int] = []
    ends: list[int] = []
    for stamps in per_client.values():
        stamps.sort()
        anchor = min(warmup - 1, len(stamps) - 2)
        selected = stamps[anchor:]
        intervals.extend(
            (right - left) / 1e6
            for left, right in zip(selected, selected[1:])
        )
        anchors.append(selected[0])
        ends.append(selected[-1])
    measured = sum(max(0, len(stamps) - warmup)
                   for stamps in per_client.values())
    seconds = (max(ends) - min(anchors)) / 1e9
    return {
        "clients": len(per_client),
        "recorded_outputs": len(raw),
        "measured_intervals": measured,
        "steady_tokens_per_second": measured / seconds,
        "itl_ms": distribution(intervals),
        "measurement_start_ns": min(anchors),
        "measurement_end_ns": max(ends),
    }


def parse_monitor_time(raw: str) -> float:
    value = datetime.strptime(raw.strip(), "%Y/%m/%d %H:%M:%S.%f")
    return value.replace(tzinfo=timezone.utc).timestamp()


def gpu_metrics(path: Path, start_ns: int, end_ns: int, scale: int) -> dict[str, Any]:
    rows: list[tuple[float, int, float, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 8:
            continue
        try:
            rows.append((parse_monitor_time(fields[0]), int(fields[1]),
                         float(fields[4]), float(fields[7])))
        except ValueError:
            continue
    target = (start_ns + end_ns) / 2e9
    offset = min(
        range(-12, 15),
        key=lambda hour: abs(
            statistics.median(row[0] for row in rows) - hour * 3600 - target
        ),
    )
    active = [
        row for row in rows
        if start_ns / 1e9 <= row[0] - offset * 3600 <= end_ns / 1e9
        and row[1] < scale
    ]
    return {
        "samples": len(active),
        "timezone_offset_hours_inferred": offset,
        "utilization_pct": distribution([row[2] for row in active]),
        "power_w": distribution([row[3] for row in active]),
    }


def event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_name: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for event in events:
        by_name[event["event"]].append(event)
    batches = [
        float(event["actual_size"])
        for event in by_name["decoder_batch_selected"]
        if event.get("operation") == "decoder_generating"
    ]
    return {
        "event_counts": {name: len(values) for name, values in by_name.items()},
        "decoder_batch_size": distribution(batches),
        "d2h_enqueue_cpu_us": distribution([
            float(event["enqueue_cpu_ns"]) / 1e3
            for event in by_name["device_to_host_enqueue"]
        ]),
        "h2d_enqueue_cpu_us": distribution([
            float(event["enqueue_cpu_ns"]) / 1e3
            for event in by_name["host_to_device_enqueue"]
        ]),
        "sync_wait_ms": distribution([
            float(event["duration_cpu_ns"]) / 1e6
            for event in by_name["operate_sync_wait"]
        ]),
        "serialize_cpu_us": distribution([
            float(event["duration_cpu_ns"]) / 1e3
            for event in by_name["transport_serialize"]
            if event.get("api") == "grouped_infer_task"
        ]),
        "socket_write_cpu_us": distribution([
            float(event["duration_cpu_ns"]) / 1e3
            for event in by_name["transport_socket_write"]
            if event.get("api") == "grouped_infer_task"
        ]),
        "deserialize_cpu_us": distribution([
            float(event["duration_cpu_ns"]) / 1e3
            for event in by_name["transport_deserialize"]
            if event.get("api") == "grouped_infer_task"
        ]),
        "sender_to_deserialize_ms": distribution([
            float(event["sender_to_deserialize_ns"]) / 1e6
            for event in by_name["transport_deserialize"]
            if event.get("api") == "grouped_infer_task"
        ]),
        "grouped_wire_bytes": distribution([
            float(event["wire_bytes"])
            for event in by_name["transport_serialize"]
            if event.get("api") == "grouped_infer_task"
        ]),
    }


def run_row(
    record: dict[str, Any], package_root: Path | None = None
) -> dict[str, Any]:
    if package_root is None:
        case_dir = Path(record["case_dir"])
        root = Path(record["result_root"])
        environment_path = (
            root.parent.parent / "environment" / f"{record['run_id']}.json"
        )
    else:
        root = package_root / "raw/results" / str(record["run_id"])
        case_dir = (
            root / f"scale-{record['scale']}" / f"clients-{record['clients']}"
            / str(record["deployment"])
        )
        environment_path = (
            package_root / "raw/environment" / f"{record['run_id']}.json"
        )
    is_control = bool(record.get("instrumentation_control", False))
    required = [
        case_dir / "result.json", case_dir / "phases.json",
        case_dir / "case.json", case_dir / "gpu_monitor.csv",
        case_dir / "args.json", case_dir / "tokens.json",
        case_dir / "decode_graph.json", environment_path,
    ]
    if not is_control:
        required.append(root / "runtime_metrics.jsonl")
    network_specs = sorted((root / "network_specs").glob("*.json"))
    missing = [str(path) for path in required if not path.exists()]
    if not network_specs:
        missing.append(str(root / "network_specs/*.json"))
    if missing:
        return {**record, "complete": False, "missing": missing}
    phases = json.loads((case_dir / "phases.json").read_text())
    environment = json.loads(environment_path.read_text())
    tokens = token_metrics(case_dir / "result.json")
    start_ns = int(tokens["measurement_start_ns"])
    end_ns = int(tokens["measurement_end_ns"])
    events = [] if is_control else [
        json.loads(line)
        for line in (root / "runtime_metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    active_events = [
        event for event in events
        if start_ns <= int(event.get("time_ns", 0)) <= end_ns
    ]
    event_types = {event["event"] for event in active_events}
    hashed = required + network_specs
    return {
        **record,
        "complete": True,
        "missing_event_types": (
            [] if is_control else sorted(REQUIRED_EVENTS - event_types)
        ),
        "source_identity": {
            "git_commit": environment.get("git_commit"),
            "git_status": environment.get("git_status"),
            "artifact_sha256": {
                str(path): sha256(path) for path in hashed
            },
        },
        "tokens": tokens,
        "gpu": gpu_metrics(
            case_dir / "gpu_monitor.csv", start_ns, end_ns,
            int(record["scale"]),
        ),
        "events": None if is_control else event_summary(active_events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--required-repeats", type=int, default=5)
    parser.add_argument(
        "--package-root", type=Path,
        help="relocate journal paths into a packaged hop-breakdown directory",
    )
    args = parser.parse_args()
    journal = json.loads(args.journal.read_text())
    rows = [
        run_row(record, package_root=args.package_root)
        for record in journal["runs"]
    ]
    grouped: dict[tuple[int, int, str, bool], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["scale"]), int(row["clients"]),
            str(row["deployment"]),
            bool(row.get("instrumentation_control", False)),
        )
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (scale, clients, deployment, is_control), group in sorted(grouped.items()):
        valid = [
            row for row in group
            if row.get("complete") and not row.get("missing_event_types")
        ]
        throughput = [row["tokens"]["steady_tokens_per_second"]
                      for row in valid]
        itl = [row["tokens"]["itl_ms"]["mean"] for row in valid]
        summaries.append({
            "scale": scale, "clients": clients, "deployment": deployment,
            "instrumentation_control": is_control,
            "valid_repeats": len(valid),
            "throughput_mean": statistics.fmean(throughput) if throughput else None,
            "throughput_95ci": mean_ci95(throughput),
            "itl_mean_ms": statistics.fmean(itl) if itl else None,
            "itl_95ci": mean_ci95(itl),
        })
    complete = all(row.get("complete") for row in rows)
    events_complete = complete and all(
        row.get("instrumentation_control", False)
        or not row.get("missing_event_types") for row in rows
    )
    repeats_complete = all(
        row["valid_repeats"] >= args.required_repeats
        for row in summaries if not row["instrumentation_control"]
    )
    control_required = int(
        journal.get("instrumentation", {}).get("control_repeats", 0)
    )
    control_rows = [
        row for row in summaries if row["instrumentation_control"]
    ]
    controls_complete = (
        control_required == 0
        or (
            len(control_rows) == 1
            and control_rows[0]["valid_repeats"] >= control_required
        )
    )
    instrumented_reference = next((
        row for row in summaries
        if not row["instrumentation_control"]
        and row["scale"] == 4 and row["clients"] == 64
        and row["deployment"] == "combined-request-agent"
    ), None)
    control_reference = control_rows[0] if control_rows else None
    observer_effect_pct = None
    observer_effect_pass = control_required == 0
    if instrumented_reference and control_reference \
            and instrumented_reference["throughput_mean"] \
            and control_reference["throughput_mean"]:
        observer_effect_pct = 100 * (
            instrumented_reference["throughput_mean"]
            / control_reference["throughput_mean"] - 1
        )
        observer_effect_pass = abs(observer_effect_pct) <= 5
    passed = (
        complete and events_complete and repeats_complete
        and controls_complete and observer_effect_pass
    )
    report = {
        "schema_version": 1,
        "evidence_id": "hardware-hop-breakdown-v1",
        "passed": passed,
        "paper_eligible": False,
        "checks": {
            "all_artifacts_present": complete,
            "all_instrumentation_events_present": events_complete,
            "required_independent_repeats": repeats_complete,
            "instrumentation_controls_complete": controls_complete,
            "instrumentation_overhead_le_5pct": observer_effect_pass,
        },
        "instrumentation_observer_effect_pct": observer_effect_pct,
        "reason": (
            "measurement matrix complete; calibration gates still required"
            if passed else "hardware measurement matrix incomplete"
        ),
        "runs": rows,
        "cell_summaries": summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"measurement_gate={'PASS' if passed else 'FAIL'}")
    print(f"wrote {args.out}")
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
