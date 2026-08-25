#!/usr/bin/env python3
"""Validate and plot ordinary and Backbone multi-path recovery experiments."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import matplotlib.pyplot as plt


class AnalysisError(RuntimeError):
    pass


CASE_STYLE = {
    (False, "reprefill"): ("Normal / Re-Prefill", "#4c78a8", "-"),
    (True, "reprefill"): ("Recover / Re-Prefill", "#f58518", "-"),
    (False, "mix"): ("Normal / Mix", "#4c78a8", "--"),
    (True, "mix"): ("Recover / Mix", "#f58518", "--"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError as error:
            raise AnalysisError(f"invalid JSONL at {path}:{line_number}") from error
    return sorted(records, key=lambda item: int(item["time_ns"]))


def named(events: Iterable[dict[str, Any]], event: str) -> list[dict[str, Any]]:
    return [item for item in events if item.get("event") == event]


def exactly_one(events: Iterable[dict[str, Any]], event: str) -> dict[str, Any]:
    matches = named(events, event)
    if len(matches) != 1:
        raise AnalysisError(f"expected one {event}, found {len(matches)}")
    return matches[0]


def token_hash(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def validate_keypoints(topology: str, keypoints: Any) -> list[str]:
    normalized = sorted((str(item[0]), int(item[1]), int(item[2])) for item in keypoints)
    if len(normalized) != 2 or any(item[1:] != (0, 6) for item in normalized):
        raise AnalysisError(
            f"{topology} must install two complete outer paths, found {normalized}"
        )
    dealers = [item[0] for item in normalized]
    if len(set(dealers)) != 2:
        raise AnalysisError(f"outer paths are not physically distinct: {dealers}")
    expected = (
        {"T0-ordinary-fast-a", "T0-ordinary-survivor"}
        if topology == "ordinary"
        else {"T0-backbone", "T0-backbone-direct"}
    )
    if set(dealers) != expected:
        raise AnalysisError(
            f"unexpected initial {topology} paths: expected={expected}, actual={dealers}"
        )
    return dealers


def correlate_recovery(
    events: list[dict[str, Any]],
    injection: dict[str, Any],
    next_injection_time: int,
    expected_scope: str,
    expected_handler: str,
) -> dict[str, Any]:
    start = int(injection["time_ns"])
    compute_node = str(injection["compute_node"])
    window = [
        event for event in events
        if start <= int(event["time_ns"]) < next_injection_time
    ]
    snapshots = [
        event for event in named(window, "shadow_state_snapshot")
        if str(event.get("compute_node")) == compute_node
    ]
    if len(snapshots) != 1:
        raise AnalysisError(
            f"cannot correlate {compute_node} to one shadow snapshot: {snapshots}"
        )
    recovery_id = str(snapshots[0]["recovery_id"])
    recovery_events = [
        event for event in events
        if str(event.get("recovery_id")) == recovery_id
    ]
    detection_candidates = [
        event for event in named(window, "failure_detected")
        if str(event.get("compute_node")) == compute_node
    ]
    if len(detection_candidates) != 1:
        raise AnalysisError(
            f"expected one failure detection for {compute_node}, found "
            f"{len(detection_candidates)}"
        )
    detection = detection_candidates[0]
    if (
        detection.get("recovery_scope") != expected_scope
        or detection.get("handler_node") != expected_handler
    ):
        raise AnalysisError(
            f"wrong recovery domain for {compute_node}: {detection}"
        )
    lookup_start = exactly_one(recovery_events, "replacement_lookup_start")
    lookup_end = exactly_one(recovery_events, "replacement_lookup_end")
    graph_installed = exactly_one(recovery_events, "replacement_graph_installed")
    handoff = exactly_one(recovery_events, "promise_handoff_committed")
    handler_return = exactly_one(recovery_events, "recovery_handler_return")
    frontier = exactly_one(recovery_events, "frontier_materialized")
    send_returns = named(recovery_events, "recovery_packet_send_return")
    sends = named(recovery_events, "recovery_packet_send")
    if not sends or len(sends) != len(send_returns):
        raise AnalysisError(
            f"recovery {recovery_id} has unmatched send/send-return events"
        )
    ordered_times = [
        start,
        int(detection["time_ns"]),
        int(lookup_start["time_ns"]),
        int(lookup_end["time_ns"]),
        int(graph_installed["time_ns"]),
        max(int(item["time_ns"]) for item in send_returns),
        int(handoff["time_ns"]),
        int(handler_return["time_ns"]),
        int(frontier["time_ns"]),
    ]
    if ordered_times != sorted(ordered_times):
        raise AnalysisError(
            f"invalid recovery ordering for {recovery_id}: {ordered_times}"
        )
    return {
        "recovery_id": recovery_id,
        "detection": detection,
        "lookup_start": lookup_start,
        "lookup_end": lookup_end,
        "graph_installed": graph_installed,
        "handoff": handoff,
        "handler_return": handler_return,
        "frontier": frontier,
        "send_count": len(sends),
    }


def analyze_run(
    result_root: Path,
    run: dict[str, Any],
    drop_steps: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    case = str(run["case"])
    topology = str(run["topology"])
    method = str(run["method"])
    repeat = int(run["repeat"])
    failure = bool(run["failure"])
    run_dir = result_root / run["run_dir"]
    raw_dir = run_dir / "raw"
    if run.get("benchmark_returncode") != 0 or run.get("error") is not None:
        raise AnalysisError(f"manifest marks {case}/{repeat} unsuccessful")

    timestamps = [
        int(stamp) for sequence, stamp in read_json(raw_dir / "result.json")
        if int(sequence) == 0
    ]
    token_streams = read_json(raw_dir / "tokens.json")
    phases = read_json(raw_dir / "phases.json")
    if len(token_streams) != 1:
        raise AnalysisError(f"expected one token stream in {raw_dir}")
    tokens = [int(token) for token in token_streams[0]]
    expected_tokens = int(phases["parameters"]["max_tokens"])
    if (
        len(tokens) != expected_tokens
        or len(timestamps) != expected_tokens - 1
        or timestamps != sorted(timestamps)
        or int(phases["parameters"]["num_paths"]) != 2
        or bool(phases["parameters"]["retrieve"]) != bool(run["retrieve"])
    ):
        raise AnalysisError(f"invalid request shape in {raw_dir}")
    initial_paths = validate_keypoints(topology, phases["keypoints"])

    events = read_jsonl(run_dir / "runtime_metrics.jsonl")
    graph_configs = named(events, "online_inference_config")
    if len(graph_configs) != 3 or any(
        not bool(item.get("enable_cuda_graph")) for item in graph_configs
    ):
        raise AnalysisError(
            f"expected CUDA graph on all three Dealers in {case}/{repeat}"
        )

    intervals_ms = [
        (end - start) / 1e6 for start, end in zip(timestamps, timestamps[1:])
    ]
    timeline = [
        {
            "case": case,
            "topology": topology,
            "method": method,
            "failure": failure,
            "repeat": repeat,
            "output_index": index,
            "accumulate_seconds": (end - timestamps[0]) / 1e9,
            "inter_token_ms": (end - start) / 1e6,
        }
        for index, (start, end) in enumerate(zip(timestamps, timestamps[1:]), 1)
    ]
    run_row = {
        "case": case,
        "topology": topology,
        "method": method,
        "failure": failure,
        "repeat": repeat,
        "token_sha256": token_hash(tokens),
        "token_count": len(tokens),
        "decode_duration_seconds": (timestamps[-1] - timestamps[0]) / 1e9,
        "median_inter_token_ms": statistics.median(intervals_ms),
        "p95_inter_token_ms": percentile(intervals_ms, 0.95),
        "max_inter_token_ms": max(intervals_ms),
        "initial_paths": json.dumps(initial_paths, separators=(",", ":")),
        "failure_count": 0,
        "all_failed_branches_were_previous_winners": None,
        "all_same_step_outputs_precede_physical_recovery": None,
    }
    if not failure:
        if named(events, "failure_injected"):
            raise AnalysisError(f"normal run {case}/{repeat} injected a failure")
        return run_row, timeline, []

    injections = named(events, "failure_injected")
    if len(injections) != len(drop_steps):
        raise AnalysisError(
            f"{case}/{repeat} expected {len(drop_steps)} failures, found {len(injections)}"
        )
    actual_steps = [int(item["step"]) for item in injections]
    if actual_steps != list(drop_steps):
        raise AnalysisError(
            f"wrong failure steps in {case}/{repeat}: {actual_steps}"
        )
    accepted_outputs = [
        event for event in named(events, "infer_output_arrival")
        if bool(event.get("accepted"))
        and str(event.get("control_id", "")).startswith("bench-")
        and not bool(event.get("trace_invalid"))
    ]
    accepted_by_step: dict[int, dict[str, Any]] = {}
    for event in accepted_outputs:
        step = int(event["step"])
        if step in accepted_by_step:
            raise AnalysisError(f"more than one accepted output for step {step}")
        accepted_by_step[step] = event

    expected_scope = "agent" if topology == "ordinary" else "backbone"
    expected_handler = str(run["agent_name"]) if topology == "ordinary" else "T0-backbone"
    expected_survivor = (
        "T0-ordinary-survivor"
        if topology == "ordinary"
        else "T0-backbone-direct"
    )
    recoveries: list[dict[str, Any]] = []
    for index, injection in enumerate(injections):
        next_time = (
            int(injections[index + 1]["time_ns"])
            if index + 1 < len(injections)
            else 2**63 - 1
        )
        correlated = correlate_recovery(
            events,
            injection,
            next_time,
            expected_scope,
            expected_handler,
        )
        step = int(injection["step"])
        failed_dealer = str(injection["dealer_node"])
        previous = accepted_by_step.get(step - 1)
        current = accepted_by_step.get(step)
        if previous is None or current is None:
            raise AnalysisError(
                f"missing accepted output around failure step {step}"
            )
        previous_trace = [str(item) for item in previous.get("trace", [])]
        current_trace = [str(item) for item in current.get("trace", [])]
        # Backbone deliberately rewrites the logical trace to its outer node
        # name.  ``source_node`` retains the physical inner Dealer identity
        # needed to prove that the branch we injected into was the winner.
        failed_was_previous_winner = (
            str(previous.get("source_node")) == failed_dealer
        )
        survivor_served_failed_step = (
            str(current.get("source_node")) == expected_survivor
        )
        frontier_time = int(correlated["frontier"]["time_ns"])
        same_step_output_before_frontier = int(current["time_ns"]) < frontier_time
        injection_time = int(injection["time_ns"])
        injection_output_index = bisect.bisect_left(timestamps, injection_time)
        if injection_output_index <= 0 or injection_output_index >= len(timestamps):
            raise AnalysisError(f"failure step {step} is outside output timeline")
        failure_interval_ms = (
            timestamps[injection_output_index]
            - timestamps[injection_output_index - 1]
        ) / 1e6
        outputs_during_physical_recovery = sum(
            injection_time <= int(event["time_ns"]) < frontier_time
            for event in accepted_outputs
        )
        recoveries.append({
            "case": case,
            "topology": topology,
            "method": method,
            "repeat": repeat,
            "recovery_index": index,
            "step": step,
            "failed_dealer": failed_dealer,
            "recovery_id": correlated["recovery_id"],
            "recovery_scope": expected_scope,
            "handler_node": expected_handler,
            "injection_output_index": injection_output_index,
            "failure_interval_ms": failure_interval_ms,
            "physical_recovery_ms": (frontier_time - injection_time) / 1e6,
            "handoff_to_physical_ms": (
                frontier_time - int(correlated["handoff"]["time_ns"])
            ) / 1e6,
            "send_count": correlated["send_count"],
            "failed_was_previous_winner": failed_was_previous_winner,
            "survivor_served_failed_step": survivor_served_failed_step,
            "same_step_output_before_frontier": same_step_output_before_frontier,
            "outputs_during_physical_recovery": outputs_during_physical_recovery,
            "previous_winner_trace": json.dumps(previous_trace, separators=(",", ":")),
            "failed_step_winner_trace": json.dumps(current_trace, separators=(",", ":")),
            "normal_matched_interval_ms": None,
            "visible_stall_excess_ms": None,
            "stall_masked_fraction": None,
        })
    run_row["failure_count"] = len(recoveries)
    run_row["all_failed_branches_were_previous_winners"] = all(
        bool(row["failed_was_previous_winner"]) for row in recoveries
    )
    run_row["all_same_step_outputs_precede_physical_recovery"] = all(
        bool(row["same_step_output_before_frontier"]) for row in recoveries
    )
    return run_row, timeline, recoveries


def add_matched_baselines(
    timeline: list[dict[str, Any]],
    recoveries: list[dict[str, Any]],
) -> None:
    normal: dict[tuple[str, str, int, int], float] = {}
    for row in timeline:
        if not bool(row["failure"]):
            normal[(
                str(row["topology"]),
                str(row["method"]),
                int(row["repeat"]),
                int(row["output_index"]),
            )] = float(row["inter_token_ms"])
    for row in recoveries:
        key = (
            str(row["topology"]),
            str(row["method"]),
            int(row["repeat"]),
            int(row["injection_output_index"]),
        )
        if key not in normal:
            raise AnalysisError(f"no matched normal interval for {key}")
        baseline = normal[key]
        excess = max(0.0, float(row["failure_interval_ms"]) - baseline)
        physical = float(row["physical_recovery_ms"])
        masked = 1.0 - excess / physical if physical > 0 else 0.0
        row["normal_matched_interval_ms"] = baseline
        row["visible_stall_excess_ms"] = excess
        row["stall_masked_fraction"] = masked


def aggregate_summary(
    runs: list[dict[str, Any]],
    recoveries: list[dict[str, Any]],
) -> dict[str, Any]:
    hashes = sorted({str(row["token_sha256"]) for row in runs})
    expected_topologies = sorted({str(row["topology"]) for row in runs})
    topology_summary: dict[str, Any] = {}
    for topology in ("ordinary", "backbone"):
        selected = [row for row in recoveries if row["topology"] == topology]
        if not selected:
            continue
        masked = [float(row["stall_masked_fraction"]) for row in selected]
        failure_intervals = [float(row["failure_interval_ms"]) for row in selected]
        physical = [float(row["physical_recovery_ms"]) for row in selected]
        topology_summary[topology] = {
            "recovery_count": len(selected),
            "median_failure_interval_ms": statistics.median(failure_intervals),
            "max_failure_interval_ms": max(failure_intervals),
            "median_physical_recovery_ms": statistics.median(physical),
            "median_stall_masked_fraction": statistics.median(masked),
            "minimum_stall_masked_fraction": min(masked),
            "all_failed_branches_were_previous_winners": all(
                bool(row["failed_was_previous_winner"]) for row in selected
            ),
            "all_survivors_served_failed_step": all(
                bool(row["survivor_served_failed_step"]) for row in selected
            ),
            "all_same_step_outputs_precede_physical_recovery": all(
                bool(row["same_step_output_before_frontier"]) for row in selected
            ),
            "minimum_outputs_during_physical_recovery": min(
                int(row["outputs_during_physical_recovery"]) for row in selected
            ),
        }
    overall_pass = (
        len(hashes) == 1
        and sorted(topology_summary) == expected_topologies
        and all(
            item["all_failed_branches_were_previous_winners"]
            and item["all_survivors_served_failed_step"]
            and item["all_same_step_outputs_precede_physical_recovery"]
            and item["minimum_outputs_during_physical_recovery"] >= 1
            and item["median_stall_masked_fraction"] >= 0.90
            and item["minimum_stall_masked_fraction"] >= 0.75
            for item in topology_summary.values()
        )
    )
    return {
        "schema_version": 1,
        "claim": (
            "A surviving redundant path keeps service outputs advancing while "
            "the failed ordinary or Backbone-internal branch recovers."
        ),
        "recovery_replay": "chunk",
        "all_token_sequences_identical": len(hashes) == 1,
        "token_hashes": hashes,
        "topologies": topology_summary,
        "overall_pass": overall_pass,
    }


def plot_topology(
    axes: tuple[Any, Any],
    topology: str,
    timeline: list[dict[str, Any]],
    recoveries: list[dict[str, Any]],
    show_legend: bool,
) -> None:
    upper, lower = axes
    for failure, method in (
        (False, "reprefill"),
        (True, "reprefill"),
        (False, "mix"),
        (True, "mix"),
    ):
        selected = [
            row for row in timeline
            if row["topology"] == topology
            and bool(row["failure"]) == failure
            and row["method"] == method
        ]
        if not selected:
            continue
        by_index: dict[int, list[dict[str, Any]]] = {}
        for row in selected:
            by_index.setdefault(int(row["output_index"]), []).append(row)
        points = sorted(
            (
                index,
                statistics.mean(float(row["accumulate_seconds"]) for row in rows),
                statistics.median(float(row["inter_token_ms"]) for row in rows),
            )
            for index, rows in by_index.items()
        )
        label, color, line_style = CASE_STYLE[(failure, method)]
        upper.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            label=label,
            color=color,
            linestyle=line_style,
            linewidth=1.8,
        )
        lower.plot(
            [point[0] for point in points],
            [point[2] for point in points],
            color=color,
            linestyle=line_style,
            linewidth=1.2,
            alpha=0.88,
        )
    topology_recoveries = [row for row in recoveries if row["topology"] == topology]
    failure_indices = sorted({int(row["recovery_index"]) for row in topology_recoveries})
    for failure_index in failure_indices:
        positions = [
            int(row["injection_output_index"])
            for row in topology_recoveries
            if int(row["recovery_index"]) == failure_index
        ]
        position = statistics.median(positions)
        recovery_ends = [
            int(row["injection_output_index"])
            + int(row["outputs_during_physical_recovery"])
            for row in topology_recoveries
            if int(row["recovery_index"]) == failure_index
        ]
        recovery_end = statistics.median(recovery_ends)
        for axis in (upper, lower):
            axis.axvspan(
                position,
                recovery_end,
                color="#bdbdbd",
                alpha=0.16,
                linewidth=0,
                label=(
                    "Dealer physical recovery"
                    if axis is upper and failure_index == failure_indices[0]
                    else None
                ),
            )
            axis.axvline(position, color="#777777", linestyle=":", linewidth=0.9)
        upper.text(
            position,
            0.97 if failure_index % 2 == 0 else 0.89,
            f"F{failure_index + 1}",
            transform=upper.get_xaxis_transform(),
            ha="center",
            va="top",
            color="#666666",
            fontsize=7,
        )
    title = "Ordinary multi-path / Agent recovery" if topology == "ordinary" else "Backbone multi-path / local recovery"
    selected_summary = [float(row["stall_masked_fraction"]) for row in topology_recoveries]
    if selected_summary:
        title += f"\nmedian recovery stall masked: {statistics.median(selected_summary) * 100:.1f}%"
    upper.set_title(title, fontsize=10)
    upper.grid(True, alpha=0.22)
    lower.grid(True, alpha=0.22)
    if show_legend:
        upper.legend(loc="upper left", fontsize=8, frameon=True)


def render_figures(
    result_root: Path,
    timeline: list[dict[str, Any]],
    recoveries: list[dict[str, Any]],
) -> None:
    figures = result_root / "figures"
    figures.mkdir(exist_ok=True)
    individual_limits: list[tuple[float, float]] = []
    for topology in ("ordinary", "backbone"):
        fig, (upper, lower) = plt.subplots(
            2, 1, figsize=(9.4, 6.8), dpi=180, sharex=True,
            gridspec_kw={"height_ratios": [1.1, 1.0]},
        )
        plot_topology((upper, lower), topology, timeline, recoveries, True)
        upper.set_ylabel("Accumulated decode time (s)")
        lower.set_ylabel("Inter-token interval (ms)")
        lower.set_xlabel("Decode output index")
        fig.tight_layout()
        stem = f"{topology}_multipath_recovery_timeline"
        fig.savefig(figures / f"{stem}.png", dpi=330)
        fig.savefig(figures / f"{stem}.pdf")
        individual_limits.append(lower.get_ylim())
        plt.close(fig)

    fig, axes = plt.subplots(
        2, 2, figsize=(15.4, 6.8), dpi=180, sharex="col", sharey="row",
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )
    plot_topology((axes[0, 0], axes[1, 0]), "ordinary", timeline, recoveries, False)
    plot_topology((axes[0, 1], axes[1, 1]), "backbone", timeline, recoveries, False)
    axes[0, 0].set_ylabel("Accumulated decode time (s)")
    axes[1, 0].set_ylabel("Inter-token interval (ms)")
    for axis in axes[1]:
        axis.set_xlabel("Decode output index")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        "Service-visible recovery stalls are masked by a surviving path",
        fontsize=11,
        y=0.935,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    fig.savefig(figures / "multipath_recovery_timeline.png", dpi=330)
    fig.savefig(figures / "multipath_recovery_timeline.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    manifest = read_json(result_root / "manifest.json")
    drop_steps = tuple(int(step) for step in manifest["parameters"]["drop_steps"])
    if manifest["parameters"].get("recovery_replay") != "chunk":
        raise AnalysisError("formal multi-path result must use chunk replay")

    run_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    for run in manifest["runs"]:
        run_row, timeline, recoveries = analyze_run(result_root, run, drop_steps)
        run_rows.append(run_row)
        timeline_rows.extend(timeline)
        recovery_rows.extend(recoveries)
    add_matched_baselines(timeline_rows, recovery_rows)
    summary = aggregate_summary(run_rows, recovery_rows)
    if not summary["overall_pass"]:
        raise AnalysisError(
            "multi-path pause masking assertions failed:\n"
            + json.dumps(summary, indent=2, ensure_ascii=False)
        )

    write_csv(result_root / "runs.csv", run_rows)
    write_csv(result_root / "timeline.csv", timeline_rows)
    write_csv(result_root / "recoveries.csv", recovery_rows)
    write_json(result_root / "summary.json", summary)
    render_figures(result_root, timeline_rows, recovery_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
