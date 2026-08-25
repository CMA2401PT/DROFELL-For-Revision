#!/usr/bin/env python3
"""Analyze throughput, batching, waits, interactions, and bounded timelines."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go  # type: ignore


FACTOR_CASES = tuple(
    f"c{c}-m{m}-d{d}"
    for c, m, d in (
        (1, 1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0),
        (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, 0),
    )
)
TIMELINE_CASES = ("c1-m1-d1", "c0-m0-d0", "historical-no-opt")
CASE_LABELS = {
    "c1-m1-d1": "all-on",
    "c0-m0-d0": "all-off",
    "historical-no-opt": "historical-no-opt",
}


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


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def distribution(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data:
        return {
            "count": 0, "mean": None, "median": None,
            "p95": None, "max": None,
        }
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "p95": percentile(data, 95),
        "max": max(data),
    }


def load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise AnalysisError(
                f"invalid JSONL at {path}:{line_number}"
            ) from error
    return records


def iter_step_transitions(
    traces: list[list[list[list[Any]]]],
    start_ns: int,
    end_ns: int,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    named: dict[str, list[float]] = defaultdict(list)
    marks: dict[str, list[float]] = defaultdict(list)
    for task in traces:
        task_hold_ms = 0.0
        task_hold_count = 0
        for raw_step in task:
            step = [(str(name), int(stamp)) for name, stamp in raw_step]
            step = [event for event in step if start_ns <= event[1] <= end_ns]
            if len(step) < 2:
                continue
            trap_starts: list[int] = []
            for name, stamp in step:
                if name == "start_trap":
                    trap_starts.append(stamp)
                elif name == "end_trap" and trap_starts:
                    hold_ms = (stamp - trap_starts.pop(0)) / 1e6
                    marks["consolidation_window_hold_ms"].append(hold_ms)
                    task_hold_ms += hold_ms
                    task_hold_count += 1
            for (left_name, left_time), (right_name, right_time) in zip(
                step, step[1:]
            ):
                duration_ms = (right_time - left_time) / 1e6
                if duration_ms < 0:
                    raise AnalysisError("trace timestamps are not monotonic")
                named[f"{left_name}->{right_name}"].append(duration_ms)

            for index, (name, stamp) in enumerate(step):
                if name == "dealer-recv":
                    target = next(
                        (
                            candidate for candidate in step[index + 1:]
                            if candidate[0].startswith("wait_")
                        ),
                        None,
                    )
                    if target is not None:
                        marks["mark1_arrival_to_mailbox_ms"].append(
                            (target[1] - stamp) / 1e6
                        )
                if name.startswith("wait_"):
                    # A mailbox transition must stay inside the current node
                    # visit. Crossing leave/dealer-recv would incorrectly fold
                    # transport and consolidation time into paper mark 2.
                    segment_start = max(
                        (
                            boundary_index + 1
                            for boundary_index, (boundary_name, _) in enumerate(
                                step[:index]
                            )
                            if boundary_name in {"leave", "dealer-recv"}
                        ),
                        default=0,
                    )
                    source = next(
                        (
                            candidate for candidate in reversed(
                                step[segment_start:index]
                            )
                            if candidate[0].startswith("done_")
                        ),
                        None,
                    )
                    if source is not None:
                        marks["mark2_binding_to_next_mailbox_ms"].append(
                            (stamp - source[1]) / 1e6
                        )
                if name == "leave":
                    source_index = next(
                        (
                            candidate_index
                            for candidate_index in range(index - 1, -1, -1)
                            if step[candidate_index][0].startswith("done_")
                        ),
                        None,
                    )
                    if source_index is not None:
                        source = step[source_index]
                        duration_ms = (stamp - source[1]) / 1e6
                        marks["mark3_completion_to_leave_ms"].append(
                            duration_ms
                        )
                        crossed_trap = any(
                            event_name in {"start_trap", "end_trap"}
                            for event_name, _ in step[source_index + 1:index]
                        )
                        if crossed_trap:
                            marks[
                                "mark3_trapped_completion_to_leave_ms"
                            ].append(duration_ms)
                        else:
                            marks[
                                "mark3_untrapped_completion_to_leave_ms"
                            ].append(duration_ms)
        if task_hold_count:
            marks["consolidation_cumulative_hold_per_task_ms"].append(
                task_hold_ms
            )
            marks["consolidation_triggers_per_task"].append(
                float(task_hold_count)
            )
    return dict(named), dict(marks)


def batch_statistics(
    metrics: list[dict[str, Any]],
    start_ns: int,
    end_ns: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [
        record for record in metrics
        if start_ns <= int(record.get("time_ns", -1)) <= end_ns
    ]
    by_operation: dict[str, list[int]] = defaultdict(list)
    consolidation_sizes: list[int] = []
    bypass_sizes: list[int] = []
    for record in selected:
        event = record.get("event")
        if event == "batch_merge":
            by_operation[str(record.get("operation"))].append(
                int(record["actual_size"])
            )
        elif event == "consolidation_flush":
            consolidation_sizes.append(int(record["actual_size"]))
        elif event == "consolidation_bypass":
            bypass_sizes.append(int(record["actual_size"]))

    def size_summary(values: list[int]) -> dict[str, Any]:
        summary = distribution(values)
        counts = Counter(values)
        summary.update({
            "counts": {str(key): counts[key] for key in sorted(counts)},
            "fraction_ge_2": (
                sum(value >= 2 for value in values) / len(values)
                if values else None
            ),
            "fraction_ge_7": (
                sum(value >= 7 for value in values) / len(values)
                if values else None
            ),
        })
        return summary

    summary = {
        "events_in_decode_window": len(selected),
        "operations": {
            operation: size_summary(values)
            for operation, values in sorted(by_operation.items())
        },
        "consolidation_flush": size_summary(consolidation_sizes),
        "consolidation_bypass": size_summary(bypass_sizes),
    }
    return summary, selected


def analyze_repeat(
    case_id: str,
    repeat_dir: Path,
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    result = read_json(repeat_dir / "result.json")
    phases = read_json(repeat_dir / "phases.json")
    traces = read_json(repeat_dir / "traces.json")
    graph = read_json(repeat_dir / "decode_graph.json")
    if not result:
        raise AnalysisError(f"empty result: {repeat_dir}")
    first_ns = int(result[0][1])
    last_ns = int(result[-1][1])
    decode_start_ns = int(phases["decode_start_ns"])
    end_ns = int(phases["end_ns"])
    observed_seconds = (last_ns - first_ns) / 1e9
    decode_wall_seconds = (end_ns - decode_start_ns) / 1e9
    task_timestamps: dict[int, list[int]] = defaultdict(list)
    for sequence_index, timestamp in result:
        task_timestamps[int(sequence_index)].append(int(timestamp))
    completion_ms = [
        (timestamps[-1] - decode_start_ns) / 1e6
        for timestamps in task_timestamps.values()
    ]
    observed_task_span_ms = [
        (timestamps[-1] - timestamps[0]) / 1e6
        for timestamps in task_timestamps.values() if len(timestamps) >= 2
    ]
    inter_token_ms = [
        (right - left) / 1e6
        for timestamps in task_timestamps.values()
        for left, right in zip(timestamps, timestamps[1:])
    ]
    named, marks = iter_step_transitions(traces, decode_start_ns, end_ns)
    batches, _ = batch_statistics(metrics, decode_start_ns, end_ns)
    return {
        "case": case_id,
        "repeat": int(repeat_dir.name.removeprefix("repeat-")),
        "tokens": len(result),
        "first_token_ns": first_ns,
        "last_token_ns": last_ns,
        "decode_start_ns": decode_start_ns,
        "end_ns": end_ns,
        "observed_seconds": observed_seconds,
        "decode_wall_seconds": decode_wall_seconds,
        "observed_tok_s": len(result) / observed_seconds,
        "decode_wall_tok_s": len(result) / decode_wall_seconds,
        "request_latency_statistics": {
            "decode_start_to_completion_ms": distribution(completion_ms),
            "measured_first_to_last_token_ms": distribution(
                observed_task_span_ms
            ),
            "inter_token_ms": distribution(inter_token_ms),
        },
        "transition_statistics": {
            name: distribution(values) for name, values in sorted(named.items())
        },
        "mark_statistics": {
            name: distribution(values) for name, values in sorted(marks.items())
        },
        "batch_statistics": batches,
        "graph": graph,
    }


def aggregate_case(case_id: str, repeats: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [float(item["observed_tok_s"]) for item in repeats]
    wall = [float(item["decode_wall_tok_s"]) for item in repeats]
    mark_names = sorted({
        name for repeat in repeats for name in repeat["mark_statistics"]
    })
    marks = {}
    for name in mark_names:
        weighted_values = []
        for repeat in repeats:
            stat = repeat["mark_statistics"].get(name)
            if stat and stat["median"] is not None:
                weighted_values.append(float(stat["median"]))
        marks[name] = distribution(weighted_values)

    latency_names = sorted({
        name
        for repeat in repeats
        for name in repeat["request_latency_statistics"]
    })
    latencies = {}
    for name in latency_names:
        repeat_medians = [
            float(repeat["request_latency_statistics"][name]["median"])
            for repeat in repeats
            if repeat["request_latency_statistics"][name]["median"] is not None
        ]
        latencies[name] = distribution(repeat_medians)

    operation_names = sorted({
        operation
        for repeat in repeats
        for operation in repeat["batch_statistics"]["operations"]
    })
    operations = {}
    for operation in operation_names:
        means = []
        near_limit = []
        for repeat in repeats:
            stat = repeat["batch_statistics"]["operations"].get(operation)
            if not stat or stat["mean"] is None:
                continue
            means.append(float(stat["mean"]))
            if stat["fraction_ge_7"] is not None:
                near_limit.append(float(stat["fraction_ge_7"]))
        operations[operation] = {
            "repeat_mean_batch_size": distribution(means),
            "repeat_fraction_ge_7": distribution(near_limit),
        }
    return {
        "case": case_id,
        "repeat_count": len(repeats),
        "observed_tok_s": distribution(observed),
        "decode_wall_tok_s": distribution(wall),
        "marks_repeat_medians_ms": marks,
        "request_latency_repeat_medians_ms": latencies,
        "batch_operations": operations,
        "repeats": repeats,
    }


def factorial_effects(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not all(case_id in cases for case_id in FACTOR_CASES):
        return {"complete": False, "reason": "not all 2^3 cases are present"}
    rates: dict[tuple[int, int, int], float] = {}
    for case_id in FACTOR_CASES:
        c = int(case_id[1])
        m = int(case_id[4])
        d = int(case_id[7])
        rates[(c, m, d)] = float(cases[case_id]["observed_tok_s"]["median"])

    conditional: dict[str, list[dict[str, Any]]] = {"C": [], "M": [], "D": []}
    for m in (0, 1):
        for d in (0, 1):
            off, on = rates[(0, m, d)], rates[(1, m, d)]
            conditional["C"].append({
                "condition": {"M": m, "D": d}, "off": off, "on": on,
                "delta_tok_s": on - off,
                "percent_vs_off": (on / off - 1) * 100,
            })
    for c in (0, 1):
        for d in (0, 1):
            off, on = rates[(c, 0, d)], rates[(c, 1, d)]
            conditional["M"].append({
                "condition": {"C": c, "D": d}, "off": off, "on": on,
                "delta_tok_s": on - off,
                "percent_vs_off": (on / off - 1) * 100,
            })
    for c in (0, 1):
        for m in (0, 1):
            off, on = rates[(c, m, 0)], rates[(c, m, 1)]
            conditional["D"].append({
                "condition": {"C": c, "M": m}, "off": off, "on": on,
                "delta_tok_s": on - off,
                "percent_vs_off": (on / off - 1) * 100,
            })

    signs = {
        "C": lambda c, m, d: 1 if c else -1,
        "M": lambda c, m, d: 1 if m else -1,
        "D": lambda c, m, d: 1 if d else -1,
    }
    contrasts = {}
    for term in ("C", "M", "D", "CM", "CD", "MD", "CMD"):
        contrast = 0.0
        for (c, m, d), rate in rates.items():
            sign = math.prod(signs[factor](c, m, d) for factor in term)
            contrast += sign * rate
        contrasts[term] = contrast / 4.0

    all_off = rates[(0, 0, 0)]
    all_on = rates[(1, 1, 1)]
    return {
        "complete": True,
        "rates": {f"c{c}-m{m}-d{d}": value for (c, m, d), value in rates.items()},
        "all_on_vs_all_off": {
            "all_off_tok_s": all_off,
            "all_on_tok_s": all_on,
            "speedup": all_on / all_off,
            "percent": (all_on / all_off - 1) * 100,
        },
        "conditional_effects": conditional,
        "factorial_contrasts_tok_s": contrasts,
        "interpretation": (
            "Contrasts and conditional effects quantify interactions; individual "
            "percentage changes must not be added."
        ),
    }


def timeline_events(
    traces: list[list[list[list[Any]]]],
    step_start: int,
    step_count: int,
) -> list[tuple[int, str, float]]:
    raw = []
    for task_index, task in enumerate(traces):
        for step in task[step_start:step_start + step_count]:
            for name, stamp in step:
                raw.append((task_index, str(name), int(stamp) / 1e6))
    if not raw:
        return []
    origin = min(stamp for _, _, stamp in raw)
    return sorted(
        ((task, name, stamp - origin) for task, name, stamp in raw),
        key=lambda item: item[2],
    )


def group_event_lines(
    events: list[tuple[int, str, float]],
    group_range_ms: float,
) -> list[tuple[str, list[tuple[int, float]]]]:
    by_name: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for task, name, stamp in events:
        by_name[name].append((task, stamp))
    grouped = []
    for name, values in by_name.items():
        current: list[tuple[int, float]] = []
        tasks: set[int] = set()
        last_stamp: float | None = None
        for task, stamp in sorted(values, key=lambda item: item[1]):
            if (
                current and (
                    task in tasks or
                    (last_stamp is not None and stamp - last_stamp > group_range_ms)
                )
            ):
                grouped.append((name, current))
                current = []
                tasks = set()
            current.append((task, stamp))
            tasks.add(task)
            last_stamp = stamp
        if current:
            grouped.append((name, current))
    return grouped


def group_event_occurrences(
    events: list[tuple[int, str, float]],
    task_count: int,
) -> list[tuple[str, list[tuple[int, float]]]]:
    """Reproduce the paper plot's visual occurrence grouping.

    For each event name, append an event to the first open group that does not
    already contain the same task. This deliberately ignores time distance; it
    exposes scheduling skew through long diagonals but is not a batch boundary.
    """
    grouped: list[tuple[str, list[tuple[int, float]]]] = []
    open_groups: dict[str, list[list[tuple[int, float]]]] = defaultdict(list)
    for task, name, stamp in sorted(events, key=lambda item: item[2]):
        candidates = open_groups[name]
        selected: list[tuple[int, float]] | None = None
        for candidate in candidates:
            if all(candidate_task != task for candidate_task, _ in candidate):
                selected = candidate
                break
        if selected is None:
            selected = []
            candidates.append(selected)
            grouped.append((name, selected))
        selected.append((task, stamp))
        if len(selected) == task_count:
            candidates.remove(selected)
    return grouped


def timeline_mark_spans(
    traces: list[list[list[list[Any]]]],
    step_start: int,
    step_count: int,
    origin_ms: float,
) -> dict[str, list[tuple[int, float, float]]]:
    spans: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for task_index, task in enumerate(traces):
        for raw_step in task[step_start:step_start + step_count]:
            step = [(str(name), int(stamp) / 1e6) for name, stamp in raw_step]
            for index, (name, stamp) in enumerate(step):
                if name == "dealer-recv":
                    target = next(
                        (
                            event for event in step[index + 1:]
                            if event[0].startswith("wait_")
                        ),
                        None,
                    )
                    if target is not None:
                        spans["1"].append(
                            (task_index, stamp - origin_ms, target[1] - origin_ms)
                        )
                elif name.startswith("wait_"):
                    segment_start = max(
                        (
                            boundary + 1
                            for boundary, (boundary_name, _) in enumerate(
                                step[:index]
                            )
                            if boundary_name in {"leave", "dealer-recv"}
                        ),
                        default=0,
                    )
                    source = next(
                        (
                            event for event in reversed(
                                step[segment_start:index]
                            )
                            if event[0].startswith("done_")
                        ),
                        None,
                    )
                    if source is not None:
                        spans["2"].append(
                            (task_index, source[1] - origin_ms, stamp - origin_ms)
                        )
                elif name == "leave":
                    source_index = next(
                        (
                            candidate
                            for candidate in range(index - 1, -1, -1)
                            if step[candidate][0].startswith("done_")
                        ),
                        None,
                    )
                    if source_index is None:
                        continue
                    if any(
                        event_name in {"start_trap", "end_trap"}
                        for event_name, _ in step[source_index + 1:index]
                    ):
                        continue
                    spans["3"].append((
                        task_index,
                        step[source_index][1] - origin_ms,
                        stamp - origin_ms,
                    ))
    return dict(spans)


def choose_zoom_range(
    events: list[tuple[int, str, float]],
    spans: dict[str, list[tuple[int, float, float]]],
    width_ms: float,
) -> tuple[float, float]:
    minimum = min(stamp for _, _, stamp in events)
    maximum = max(stamp for _, _, stamp in events)
    if maximum - minimum <= width_ms:
        return minimum, maximum
    margin_ms = min(2.0, width_ms * 0.02)
    candidates = {minimum, maximum - width_ms}
    for values in spans.values():
        for _, start, end in values:
            midpoint = (start + end) / 2
            for candidate in (
                midpoint - width_ms / 2,
                start - margin_ms,
                end + margin_ms - width_ms,
            ):
                candidates.add(max(
                    minimum, min(candidate, maximum - width_ms)
                ))

    def score(window_start: float) -> tuple[int, float]:
        window_end = window_start + width_ms
        covered_types = 0
        duration_score = 0.0
        for values in spans.values():
            visible = [
                max(0.0, end - start)
                for _, start, end in values
                if (window_start + margin_ms <= start and
                    end <= window_end - margin_ms)
            ]
            if visible:
                covered_types += 1
                duration_score += max(visible)
        return covered_types, duration_score

    start = max(candidates, key=score)
    return start, start + width_ms


def event_color(name: str) -> str:
    if name == "leave":
        return "#252A34"
    if "recv" in name:
        return "#FF2E63"
    if name.startswith("wait_"):
        return "#EABFCB"
    if name.startswith("exec_"):
        return "#00ADB5"
    if name.startswith("done_"):
        return "#63C7CC"
    if "trap" in name:
        return "#B9EBD9"
    return "#B8B8E8"


def make_timeline(
    case_id: str,
    repeat: dict[str, Any],
    result_root: Path,
    step_start: int,
    step_count: int,
    group_range_ms: float,
    zoom_width_ms: float,
) -> None:
    repeat_dir = result_root / case_id / f"repeat-{repeat['repeat']}"
    traces = read_json(repeat_dir / "traces.json")
    events = timeline_events(traces, step_start, step_count)
    if not events:
        return
    occurrence_groups = group_event_occurrences(events, len(traces))
    adjacent_groups = group_event_lines(events, group_range_ms)
    raw_stamps_ms = [
        int(stamp) / 1e6
        for task in traces
        for step in task[step_start:step_start + step_count]
        for _, stamp in step
    ]
    origin_ms = min(raw_stamps_ms)
    mark_spans = timeline_mark_spans(
        traces, step_start, step_count, origin_ms
    )
    figure_dir = result_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    label = CASE_LABELS.get(case_id, case_id)

    plotly_figure = go.Figure()
    legends: set[str] = set()
    for name, points in occurrence_groups:
        if name.startswith("async_"):
            continue
        points = sorted(points, key=lambda item: item[1])
        plotly_figure.add_trace(go.Scatter(
            x=[stamp for _, stamp in points],
            y=[task for task, _ in points],
            text=[name] * len(points),
            name=name,
            legendgroup=name,
            showlegend=name not in legends,
            mode="lines+markers",
            opacity=(1.0 if name.startswith(("exec_", "done_")) or
                     name in {"leave", "dealer-recv"} else 0.35),
            line={
                "color": event_color(name),
                "width": 2,
                "dash": "dash" if name.startswith("done_") else "solid",
            },
            marker={"size": 5},
            hovertemplate="%{text}<br>%{x:.3f} ms<br>task %{y}<extra></extra>",
        ))
        legends.add(name)
    plotly_figure.update_layout(
        title=(
            f"{label}: event-occurrence timeline (visual grouping; "
            "not batch boundaries)"
        ),
        xaxis_title="Time from bounded-window origin (ms)",
        yaxis_title="SubTask",
        template="plotly_white",
        height=650,
        hovermode="closest",
        dragmode="zoom",
        font={"family": "Times New Roman"},
    )
    plotly_figure.write_html(
        figure_dir / f"{label}.html", include_plotlyjs="cdn"
    )

    def render_static(
        groups: list[tuple[str, list[tuple[int, float]]]],
        output_name: str,
        grouping_label: str,
        x_range: tuple[float, float] | None = None,
        annotate_marks: bool = False,
    ) -> None:
        visible_names = {"start", "root", "finish"}
        key_groups = [
            (name, points) for name, points in groups
            if not name.startswith("async_") and (
                name in visible_names or name == "dealer-recv" or
                name == "leave" or
                name.startswith(("wait_", "exec_", "done_")) or
                "trap" in name
            )
        ]
        figure, axis = plt.subplots(figsize=(17, 6.5), dpi=180)
        legend_seen: set[str] = set()
        for name, points in key_groups:
            points = sorted(points, key=lambda item: item[1])
            axis.plot(
                [stamp for _, stamp in points],
                [task for task, _ in points],
                marker="o", markersize=2.8, linewidth=1.2,
                alpha=(1.0 if name.startswith(("exec_", "done_")) or
                       name in {"leave", "dealer-recv"} else 0.32),
                color=event_color(name),
                linestyle="--" if name.startswith("done_") else "-",
                label=name if name not in legend_seen else None,
            )
            legend_seen.add(name)
        if x_range is not None:
            axis.set_xlim(*x_range)
        if annotate_marks and x_range is not None:
            annotation_margin_ms = min(2.0, (x_range[1] - x_range[0]) * 0.02)
            for mark in ("1", "2", "3"):
                candidates = [
                    span for span in mark_spans.get(mark, [])
                    if (x_range[0] + annotation_margin_ms <= span[1] and
                        span[2] <= x_range[1] - annotation_margin_ms)
                ]
                if not candidates:
                    continue
                task, start, end = max(
                    candidates, key=lambda span: span[2] - span[1]
                )
                axis.annotate(
                    "", xy=(end, task), xytext=(start, task),
                    arrowprops={
                        "arrowstyle": "<->", "color": "#FF8C00",
                        "linewidth": 2.4,
                    },
                    zorder=20,
                )
                axis.annotate(
                    mark, xy=((start + end) / 2, min(task + 0.65, 15.3)),
                    ha="center", va="center", color="white", fontsize=10,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "square,pad=0.25", "facecolor": "#FF8C00",
                        "edgecolor": "#FF8C00",
                    },
                    zorder=21,
                )
        suffix = " (local zoom)" if x_range is not None else ""
        axis.set_title(
            f"{label}: {step_count} bounded decode steps per task{suffix}\n"
            f"{grouping_label}; visual connections are not batch boundaries"
        )
        axis.set_xlabel("Time from bounded-window origin (ms)")
        axis.set_ylabel("SubTask")
        axis.set_yticks(range(16))
        axis.grid(True, alpha=0.18)
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        figure.tight_layout()
        figure.savefig(figure_dir / output_name, bbox_inches="tight")
        plt.close(figure)

    render_static(
        occurrence_groups, f"{label}.png", "event-occurrence grouping"
    )
    zoom_range = choose_zoom_range(events, mark_spans, zoom_width_ms)
    render_static(
        occurrence_groups,
        f"{label}-zoom.png",
        "event-occurrence grouping",
        x_range=zoom_range,
        annotate_marks=case_id in {"c0-m0-d0", "historical-no-opt"},
    )
    render_static(
        adjacent_groups,
        f"{label}-time-adjacent.png",
        f"time-adjacent grouping (gap ≤ {group_range_ms:g} ms)",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--timeline-step-start", type=int, default=12)
    parser.add_argument("--timeline-step-count", type=int, default=24)
    parser.add_argument("--group-range-ms", type=float, default=0.25)
    parser.add_argument("--timeline-zoom-ms", type=float, default=100.0)
    args = parser.parse_args()
    result_root = args.result_root
    if not result_root.exists():
        raise AnalysisError(f"result directory does not exist: {result_root}")

    case_results: dict[str, dict[str, Any]] = {}
    repeat_rows = []
    transition_rows = []
    batch_rows = []
    for case_dir in sorted(path for path in result_root.iterdir() if path.is_dir()):
        repeat_dirs = sorted(
            path for path in case_dir.glob("repeat-*")
            if (path / "result.json").exists()
        )
        if not repeat_dirs:
            continue
        metrics = load_metrics(case_dir / "runtime_metrics.jsonl")
        repeats = [
            analyze_repeat(case_dir.name, repeat_dir, metrics)
            for repeat_dir in repeat_dirs
        ]
        aggregate = aggregate_case(case_dir.name, repeats)
        case_results[case_dir.name] = aggregate
        for repeat in repeats:
            repeat_rows.append({
                "case": case_dir.name,
                "repeat": repeat["repeat"],
                "tokens": repeat["tokens"],
                "observed_seconds": repeat["observed_seconds"],
                "observed_tok_s": repeat["observed_tok_s"],
                "decode_wall_seconds": repeat["decode_wall_seconds"],
                "decode_wall_tok_s": repeat["decode_wall_tok_s"],
                "decode_start_to_completion_median_ms": repeat[
                    "request_latency_statistics"
                ]["decode_start_to_completion_ms"]["median"],
                "inter_token_median_ms": repeat[
                    "request_latency_statistics"
                ]["inter_token_ms"]["median"],
                "cumulative_hold_per_task_median_ms": repeat[
                    "mark_statistics"
                ].get("consolidation_cumulative_hold_per_task_ms", {}).get(
                    "median"
                ),
            })
            for mark, stat in repeat["mark_statistics"].items():
                transition_rows.append({
                    "case": case_dir.name,
                    "repeat": repeat["repeat"],
                    "metric": mark,
                    "count": stat["count"],
                    "mean_ms": stat["mean"],
                    "median_ms": stat["median"],
                    "p95_ms": stat["p95"],
                    "max_ms": stat["max"],
                })
            for operation, stat in repeat["batch_statistics"]["operations"].items():
                batch_rows.append({
                    "case": case_dir.name,
                    "repeat": repeat["repeat"],
                    "operation": operation,
                    "count": stat["count"],
                    "mean_size": stat["mean"],
                    "median_size": stat["median"],
                    "fraction_ge_2": stat["fraction_ge_2"],
                    "fraction_ge_7": stat["fraction_ge_7"],
                    "size_counts": json.dumps(stat["counts"], sort_keys=True),
                })

    if not case_results:
        raise AnalysisError(f"no complete repeats under {result_root}")
    effects = factorial_effects(case_results)
    summary = {
        "schema_version": 1,
        "result_root": str(result_root),
        "cases": case_results,
        "factorial_effects": effects,
    }
    write_json(result_root / "summary.json", summary)
    write_json(result_root / "interaction_effects.json", effects)
    write_csv(result_root / "summary.csv", repeat_rows)
    write_csv(result_root / "transition_summary.csv", transition_rows)
    write_csv(result_root / "batch_summary.csv", batch_rows)

    figure_dir = result_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    ordered_cases = [case for case in FACTOR_CASES if case in case_results]
    if "historical-no-opt" in case_results:
        ordered_cases.append("historical-no-opt")
    medians = [
        float(case_results[case]["observed_tok_s"]["median"])
        for case in ordered_cases
    ]
    figure, axis = plt.subplots(figsize=(12, 5.5), dpi=180)
    bars = axis.bar(ordered_cases, medians, color="#00ADB5")
    axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axis.set_ylabel("Total observed throughput (token/s)")
    axis.set_title("Inference consolidation factorial experiment")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_dir / "throughput.png")
    plt.close(figure)

    for case_id in TIMELINE_CASES:
        if case_id not in case_results:
            continue
        median_rate = float(case_results[case_id]["observed_tok_s"]["median"])
        representative = min(
            case_results[case_id]["repeats"],
            key=lambda item: abs(float(item["observed_tok_s"]) - median_rate),
        )
        make_timeline(
            case_id, representative, result_root,
            args.timeline_step_start, args.timeline_step_count,
            args.group_range_ms, args.timeline_zoom_ms,
        )

    print(json.dumps({
        "cases": {
            case: case_results[case]["observed_tok_s"]["median"]
            for case in ordered_cases
        },
        "all_on_vs_all_off": effects.get("all_on_vs_all_off"),
    }, indent=2))


if __name__ == "__main__":
    main()
