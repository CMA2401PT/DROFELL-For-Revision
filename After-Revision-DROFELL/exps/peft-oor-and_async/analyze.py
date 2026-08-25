#!/usr/bin/env python3
"""Analyze throughput, request concurrency, and same-device CUDA overlap."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt

from scripts.online.benchmark.analysis_train import plot_time_line


EXP_ROOT = Path(__file__).resolve().parent
EVENT_STYLE = {
    "async": "goldenrod",
    "forward": "green",
    "backward": "red",
    "net": "black",
    "update": "blue",
    "swap": "brown",
    "dump": "pink",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class TaskInterval:
    task_id: str
    client: str
    lane: str
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class DeviceOp:
    task_id: str
    client: str
    lane: str
    device: str
    operation: str
    ordinal: int
    exec_ns: int
    done_ns: int
    async_ns: int


def task_records(traces: dict[str, list[list[list[Any]]]]) -> Iterable[
    tuple[str, str, list[list[Any]]]
]:
    for lane, records in sorted(traces.items()):
        for index, record in enumerate(records):
            yield lane, f"{lane}#{index}", record


def extract_task_intervals(
    traces: dict[str, list[list[list[Any]]]],
) -> list[TaskInterval]:
    intervals = []
    for lane, task_id, record in task_records(traces):
        if not record:
            continue
        intervals.append(TaskInterval(
            task_id=task_id,
            client=lane.split("-", 1)[0],
            lane=lane,
            start_ns=int(record[0][1]),
            end_ns=int(record[-1][1]),
        ))
    return intervals


def request_overlap(intervals: list[TaskInterval]) -> dict[str, Any]:
    examples = []
    total_ns = 0
    for index, first in enumerate(intervals):
        for second in intervals[index + 1:]:
            if first.client != second.client or first.task_id == second.task_id:
                continue
            start = max(first.start_ns, second.start_ns)
            end = min(first.end_ns, second.end_ns)
            if end <= start:
                continue
            overlap = end - start
            total_ns += overlap
            examples.append({
                "client": first.client,
                "first_task": first.task_id,
                "second_task": second.task_id,
                "start_ns": start,
                "end_ns": end,
                "overlap_ms": overlap / 1e6,
            })
    examples.sort(key=lambda item: item["overlap_ms"], reverse=True)
    return {
        "pair_count": len(examples),
        "total_pairwise_overlap_ms": total_ns / 1e6,
        "max_overlap_ms": examples[0]["overlap_ms"] if examples else 0.0,
        "examples": examples[:20],
    }


def _device_for(operation: str, ordinal: int, count: int) -> str:
    if operation == "head":
        return "device-1"
    if operation == "tail":
        return "device-2"
    if operation == "forward":
        # Four 8-layer groups: first two on dealer/device 1, last two on 2.
        return "device-1" if ordinal < count / 2 else "device-2"
    if operation == "backward":
        # Backward traverses the same groups in reverse.
        return "device-2" if ordinal < count / 2 else "device-1"
    raise ValueError(operation)


def extract_device_ops(
    traces: dict[str, list[list[list[Any]]]],
) -> list[DeviceOp]:
    result = []
    for lane, task_id, record in task_records(traces):
        events = [(str(name), int(stamp)) for name, stamp in record]
        counts = {
            operation: sum(name == f"exec_{operation}" for name, _ in events)
            for operation in ("head", "forward", "tail", "backward")
        }
        ordinals = {operation: 0 for operation in counts}
        for index in range(len(events) - 2):
            exec_name, exec_ns = events[index]
            if not exec_name.startswith("exec_"):
                continue
            operation = exec_name.removeprefix("exec_")
            if operation not in counts:
                continue
            done_name, done_ns = events[index + 1]
            async_name, async_ns = events[index + 2]
            if (
                done_name != f"done_{operation}"
                or async_name != f"async_{operation}"
            ):
                continue
            ordinal = ordinals[operation]
            ordinals[operation] += 1
            result.append(DeviceOp(
                task_id=task_id,
                client=lane.split("-", 1)[0],
                lane=lane,
                device=_device_for(operation, ordinal, counts[operation]),
                operation=operation,
                ordinal=ordinal,
                exec_ns=exec_ns,
                done_ns=done_ns,
                async_ns=async_ns,
            ))
    return result


def same_device_overlap(ops: list[DeviceOp]) -> dict[str, Any]:
    """Find B's submission while A is waiting for CUDA completion."""
    examples = []
    total_ns = 0
    for waiting in ops:
        if waiting.async_ns <= waiting.done_ns:
            continue
        for submitting in ops:
            if (
                waiting.device != submitting.device
                or waiting.task_id == submitting.task_id
            ):
                continue
            start = max(waiting.done_ns, submitting.exec_ns)
            end = min(waiting.async_ns, submitting.done_ns)
            if end <= start:
                continue
            overlap = end - start
            total_ns += overlap
            examples.append({
                "device": waiting.device,
                "waiting_task": waiting.task_id,
                "waiting_operation": waiting.operation,
                "waiting_ordinal": waiting.ordinal,
                "submitting_task": submitting.task_id,
                "submitting_operation": submitting.operation,
                "submitting_ordinal": submitting.ordinal,
                "start_ns": start,
                "end_ns": end,
                "overlap_ms": overlap / 1e6,
            })
    examples.sort(key=lambda item: item["overlap_ms"], reverse=True)
    return {
        "pair_count": len(examples),
        "total_pairwise_overlap_ms": total_ns / 1e6,
        "max_overlap_ms": examples[0]["overlap_ms"] if examples else 0.0,
        "examples": examples[:50],
    }


def action_style(action: str) -> tuple[str, str] | None:
    if action == "leave->dealer-recv":
        return "net", EVENT_STYLE["net"]
    if action.startswith("exec_") and "backward" in action:
        return "backward", EVENT_STYLE["backward"]
    if action.startswith("exec_"):
        return "forward", EVENT_STYLE["forward"]
    if action.startswith("done_") and "->async_" in action:
        return "async", EVENT_STYLE["async"]
    return None


def build_timeline(
    traces: dict[str, list[list[list[Any]]]],
    brains: dict[str, list[list[Any]]],
) -> list[tuple[str, str, float, float]]:
    lines: list[tuple[str, str, float, float]] = []
    for lane, _, record in task_records(traces):
        for (first, start), (second, end) in zip(record, record[1:]):
            style = action_style(f"{first}->{second}")
            if style is None:
                continue
            lines.append((
                f"client-{lane}", style[0], int(start) / 1e9, int(end) / 1e9
            ))

    brain_style = {
        "forward": "forward",
        "backward": "backward",
        "train_head": "forward",
        "train_tail": "forward",
        "weight_update": "update",
        "swapout": "swap",
        "dump_state": "dump",
    }
    for dealer, entries in brains.items():
        lane = dealer.replace("dealer", "device")
        for _, record in entries:
            description = record.get("description", [])
            if not description:
                continue
            action = brain_style.get(description[0])
            if action is None:
                continue
            start = int(record["begin_time"]) / 1e9
            end = (
                int(record["begin_time"])
                + int(record["delta_time"])
                + int(record.get("extra_async_delta_time", 0))
            ) / 1e9
            lines.append((lane, action, start, end))
    return lines


def choose_window(
    lines: list[tuple[str, str, float, float]],
    overlap: dict[str, Any],
    width: float = 1.8,
) -> tuple[float, float]:
    origin = min(start for _, _, start, _ in lines)
    total_end = max(end for _, _, _, end in lines) - origin
    examples = overlap["examples"]
    if examples:
        center = (
            (examples[0]["start_ns"] + examples[0]["end_ns"]) / 2e9
            - origin
        )
    else:
        center = total_end * 0.45
    start = max(0.0, min(center - width / 2, max(0.0, total_end - width)))
    return start, min(total_end, start + width)


def write_static_timeline(
    lines: list[tuple[str, str, float, float]],
    path: Path,
    window: tuple[float, float],
    highlights: list[dict[str, Any]],
) -> None:
    origin = min(start for _, _, start, _ in lines)
    lanes = sorted(
        {lane for lane, _, _, _ in lines},
        key=lambda lane: (lane.startswith("device"), lane),
    )
    y_for = {lane: len(lanes) - index - 1 for index, lane in enumerate(lanes)}
    figure, axis = plt.subplots(figsize=(13, 6), dpi=180)
    legend_handles: dict[str, Any] = {}
    for lane, action, start, end in lines:
        rel_start = start - origin
        if end - origin < window[0] or rel_start > window[1]:
            continue
        bars = axis.broken_barh(
            [(rel_start, max(0.0, end - start))],
            (y_for[lane] - 0.35, 0.7),
            facecolors=EVENT_STYLE[action], edgecolors=EVENT_STYLE[action],
            linewidth=0.4, label=action,
        )
        legend_handles.setdefault(action, bars)
    for highlight in highlights:
        start = highlight["start_second"] - origin
        end = highlight["end_second"] - origin
        axis.axvspan(
            start, end, facecolor="none", edgecolor="black",
            linewidth=1.5, linestyle="--",
        )
        axis.text(
            (start + end) / 2, len(lanes) - 0.35, highlight["label"],
            horizontalalignment="center", verticalalignment="bottom",
            fontsize=13, fontweight="bold",
        )
    axis.set_xlim(*window)
    axis.set_ylim(-0.8, len(lanes) - 0.2)
    axis.set_yticks([y_for[lane] for lane in lanes], labels=lanes)
    axis.set_xlabel("Time (second)")
    axis.grid(axis="x", color="0.85", linewidth=0.6)
    axis.legend(
        legend_handles.values(), legend_handles.keys(), loc="upper center",
        bbox_to_anchor=(0.5, 1.12), ncol=len(legend_handles), frameon=False,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def write_timeline_figures(
    case_dir: Path,
    figure_dir: Path,
    case_id: str,
    overlap: dict[str, Any],
) -> dict[str, Any]:
    traces = read_json(case_dir / "traces.json")
    brains = read_json(case_dir / "brains.json")
    lines = build_timeline(traces, brains)
    if not lines:
        raise ValueError(f"no timeline records in {case_dir}")
    window = choose_window(lines, overlap)
    origin = min(start for _, _, start, _ in lines)
    highlights = []
    if case_id == "opt" and overlap["examples"]:
        example = overlap["examples"][0]
        highlights.append({
            "label": "3",
            "start_second": example["start_ns"] / 1e9,
            "end_second": example["end_ns"] / 1e9,
            "meaning": "same-device outstanding CUDA overlap",
        })
    elif case_id == "non-opt":
        intervals = extract_task_intervals(traces)
        gaps = []
        for client in sorted({interval.client for interval in intervals}):
            ordered = sorted(
                (interval for interval in intervals if interval.client == client),
                key=lambda interval: interval.start_ns,
            )
            for previous, current in zip(ordered, ordered[1:]):
                if current.start_ns > previous.end_ns:
                    gaps.append((
                        current.start_ns - previous.end_ns,
                        previous.end_ns / 1e9,
                        current.start_ns / 1e9,
                    ))
        for label, (_, start, end) in zip(("1", "2"), sorted(gaps, reverse=True)):
            highlights.append({
                "label": label,
                "start_second": start,
                "end_second": end,
                "meaning": "request-emission gap",
            })
        if highlights:
            centers = [
                (item["start_second"] + item["end_second"]) / 2 - origin
                for item in highlights
            ]
            center = sum(centers) / len(centers)
            width = max(1.8, max(centers) - min(centers) + 0.8)
            total_end = max(end for _, _, _, end in lines) - origin
            start = max(0.0, min(center - width / 2, max(0.0, total_end - width)))
            window = (start, min(total_end, start + width))
    clients = sorted({lane.rsplit("-", 1)[0] for lane, _, _, _ in lines if lane.startswith("client-")})

    def group_fn(name: str) -> str:
        if name.startswith("device"):
            return "device"
        return name.rsplit("-", 1)[0]

    figure = plot_time_line(
        time_lines=lines,
        group_fn=group_fn,
        priority=clients + ["device"],
        color_mapping=EVENT_STYLE,
        trim_start=window[0],
        trim_end=window[1],
    )
    figure.update_layout(width=1300, title=f"{case_id}: PEFT timeline")
    for highlight in highlights:
        figure.add_vrect(
            x0=highlight["start_second"] - origin,
            x1=highlight["end_second"] - origin,
            line_width=2,
            line_dash="dash",
            line_color="black",
            fillcolor="rgba(0,0,0,0)",
            annotation_text=highlight["label"],
            annotation_position="top",
        )
    html_path = figure_dir / f"{case_id}.html"
    png_path = figure_dir / f"{case_id}.png"
    figure.write_html(html_path)
    write_static_timeline(lines, png_path, window, highlights)
    return {
        "html": str(html_path),
        "png": str(png_path),
        "window_seconds": list(window),
        "highlights": highlights,
    }


def summarize_case(case_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    case = read_json(case_dir / "case.json")
    args = read_json(case_dir / "args.json")
    network = read_json(case_dir / "network.json")
    records = read_json(case_dir / "results.json")
    traces = read_json(case_dir / "traces.json")
    if not (case_dir / "brains.json").exists():
        raise ValueError(f"Dealer time trace is missing: {case_dir / 'brains.json'}")
    global_record = records[-1]
    tokens = sum(global_record["tokens"])
    start = global_record["start_times"][0]
    end = global_record["times"][-1]
    expected_minibatches = 2 * case["samples_per_client"] // 4
    if len(global_record["times"]) != expected_minibatches:
        raise ValueError(
            f"{case_dir}: expected {expected_minibatches} minibatches, "
            f"got {len(global_record['times'])}"
        )
    if args["agent"] is not None:
        raise ValueError(f"{case_dir}: expected combined Request-Agent")
    if args["async_backend"] != case["async_backend"]:
        raise ValueError(f"{case_dir}: async backend mismatch")
    task_overlap = request_overlap(extract_task_intervals(traces))
    device_overlap = same_device_overlap(extract_device_ops(traces))
    row = {
        **case,
        "network_name": network["name"],
        "network_spec_sha256": network["spec_sha256"],
        "global_minibatches": len(global_record["times"]),
        "global_tokens": tokens,
        "measurement_seconds": end - start,
        "total_tokens_per_second": tokens / (end - start),
        "request_overlap_pair_count": task_overlap["pair_count"],
        "request_max_overlap_ms": task_overlap["max_overlap_ms"],
        "same_device_overlap_pair_count": device_overlap["pair_count"],
        "same_device_max_overlap_ms": device_overlap["max_overlap_ms"],
        "same_device_total_pairwise_overlap_ms": (
            device_overlap["total_pairwise_overlap_ms"]
        ),
    }
    evidence = {
        "case_id": case["case_id"],
        "request_overlap": task_overlap,
        "same_device_cuda_outstanding_overlap": device_overlap,
    }
    return row, evidence


def relative_gain(numerator: dict[str, Any], denominator: dict[str, Any]) -> float:
    return (
        numerator["total_tokens_per_second"]
        / denominator["total_tokens_per_second"] - 1
    )


def scan_logs(run_id: str) -> dict[str, Any]:
    root = EXP_ROOT / "logs" / run_id
    patterns = (
        "Traceback", "out of memory", "CUDA error", "fallback",
        "ConnectionResetError", "ConnectionRefusedError",
    )
    hits = []
    known_shutdown_warnings = []
    for path in sorted(root.rglob("*.log")):
        content = path.read_text(encoding="utf-8", errors="replace")
        known_py_shutdown = (
            "Task was destroyed but it is pending!" in content
            and "SystemExit: 0" in content
        )
        for pattern in patterns:
            if pattern.lower() in content.lower():
                if pattern == "Traceback" and known_py_shutdown:
                    known_shutdown_warnings.append({
                        "file": str(path),
                        "warning": "py event-loop pending-task cleanup after successful SystemExit(0)",
                    })
                    continue
                hits.append({"file": str(path), "pattern": pattern})
    return {
        "root": str(root),
        "fatal_error_hits": hits,
        "known_shutdown_warnings": known_shutdown_warnings,
    }


def summarize(run_root: Path) -> dict[str, Any]:
    rows = []
    evidences = []
    for case_id in ("opt", "no-ooo", "no-async-poll", "non-opt"):
        case_dir = run_root / case_id
        if not (case_dir / "results.json").exists():
            continue
        row, evidence = summarize_case(case_dir)
        rows.append(row)
        evidences.append(evidence)
    if not rows:
        raise ValueError(f"no completed cases under {run_root}")
    by_id = {row["case_id"]: row for row in rows}
    comparisons = []
    pairs = (
        ("ooo_gain_async_on", "opt", "no-ooo"),
        ("ooo_gain_async_off", "no-async-poll", "non-opt"),
        ("async_gain_ooo_on", "opt", "no-async-poll"),
        ("async_gain_ooo_off", "no-ooo", "non-opt"),
        ("joint_gain", "opt", "non-opt"),
    )
    for name, enabled, disabled in pairs:
        if enabled in by_id and disabled in by_id:
            comparisons.append({
                "name": name,
                "enabled_case": enabled,
                "disabled_case": disabled,
                "relative_gain": relative_gain(by_id[enabled], by_id[disabled]),
                "percent_gain": relative_gain(by_id[enabled], by_id[disabled]) * 100,
            })

    figure_dir = run_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    evidence_by_id = {item["case_id"]: item for item in evidences}
    for case_id in ("opt", "non-opt"):
        if case_id in by_id:
            figures[case_id] = write_timeline_figures(
                run_root / case_id, figure_dir, case_id,
                evidence_by_id[case_id]["same_device_cuda_outstanding_overlap"],
            )

    log_scan = scan_logs(run_root.name)
    summary = {
        "run_root": str(run_root),
        "cases": rows,
        "comparisons": comparisons,
        "figures": figures,
        "log_scan": log_scan,
    }
    write_json(run_root / "summary.json", summary)
    write_json(run_root / "overlap_examples.json", evidences)
    fields = (
        "case_id", "out_of_order_dispatch", "async_poller",
        "async_backend", "global_tokens", "measurement_seconds",
        "total_tokens_per_second", "request_overlap_pair_count",
        "request_max_overlap_ms", "same_device_overlap_pair_count",
        "same_device_max_overlap_ms", "same_device_total_pairwise_overlap_ms",
    )
    with (run_root / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
