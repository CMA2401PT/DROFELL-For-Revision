#!/usr/bin/env python3
"""Validate and plot the clean Re-Prefill single-recovery experiment."""

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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError as error:
            raise AnalysisError(f"invalid JSONL at {path}:{number}") from error
    return sorted(records, key=lambda item: int(item["time_ns"]))


def named(events: Iterable[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event") == name]


def exactly_one(events: Iterable[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = named(events, name)
    if len(matches) != 1:
        raise AnalysisError(f"expected one {name}, found {len(matches)}")
    return matches[0]


def token_hash(tokens: list[int]) -> str:
    raw = json.dumps(tokens, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


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


def correlate_recovery(
    events: list[dict[str, Any]],
    injection: dict[str, Any],
    expected_scope: str,
    expected_handler: str,
) -> dict[str, Any]:
    compute_node = str(injection["compute_node"])
    start = int(injection["time_ns"])
    snapshots = [
        event for event in named(events, "shadow_state_snapshot")
        if int(event["time_ns"]) >= start
        and str(event.get("compute_node")) == compute_node
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
    detections = [
        event for event in named(events, "failure_detected")
        if int(event["time_ns"]) >= start
        and str(event.get("compute_node")) == compute_node
    ]
    if len(detections) != 1:
        raise AnalysisError(
            f"expected one failure detection for {compute_node}, "
            f"found {len(detections)}"
        )
    detection = detections[0]
    if (
        detection.get("recovery_scope") != expected_scope
        or detection.get("handler_node") != expected_handler
    ):
        raise AnalysisError(f"wrong recovery domain: {detection}")
    lookup_start = exactly_one(recovery_events, "replacement_lookup_start")
    lookup_end = exactly_one(recovery_events, "replacement_lookup_end")
    graph = exactly_one(recovery_events, "replacement_graph_installed")
    handoff = exactly_one(recovery_events, "promise_handoff_committed")
    handler_return = exactly_one(recovery_events, "recovery_handler_return")
    frontier = exactly_one(recovery_events, "frontier_materialized")
    sends = named(recovery_events, "recovery_packet_send")
    send_returns = named(recovery_events, "recovery_packet_send_return")
    if not sends or len(sends) != len(send_returns):
        raise AnalysisError("recovery packets are missing or unmatched")
    times = [
        start,
        int(detection["time_ns"]),
        int(lookup_start["time_ns"]),
        int(lookup_end["time_ns"]),
        int(graph["time_ns"]),
        max(int(event["time_ns"]) for event in send_returns),
        int(handoff["time_ns"]),
        int(handler_return["time_ns"]),
        int(frontier["time_ns"]),
    ]
    if times != sorted(times):
        raise AnalysisError(f"invalid recovery order: {times}")
    replacement = lookup_end.get("replacement")
    replacement_dealers = sorted({
        str(edge[0]) for edge in (replacement or [])
        if isinstance(edge, (list, tuple)) and edge
    })
    return {
        "recovery_id": recovery_id,
        "handoff": handoff,
        "frontier": frontier,
        "send_count": len(sends),
        "replacement_dealers": replacement_dealers,
    }


def analyze_run(
    result_root: Path,
    run: dict[str, Any],
    drop_step: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    case = str(run["case"])
    topology = str(run["topology"])
    repeat = int(run["repeat"])
    failure = bool(run["failure"])
    run_dir = result_root / str(run["run_dir"])
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
    max_tokens = int(phases["parameters"]["max_tokens"])
    expected_paths = 2 if topology == "ordinary" else 1
    if (
        len(tokens) != max_tokens
        or len(timestamps) != max_tokens - 1
        or timestamps != sorted(timestamps)
        or int(phases["parameters"]["num_paths"]) != expected_paths
        or bool(phases["parameters"]["retrieve"])
    ):
        raise AnalysisError(f"invalid clean request shape in {raw_dir}")

    events = read_jsonl(run_dir / "runtime_metrics.jsonl")
    graph_configs = named(events, "online_inference_config")
    if len(graph_configs) != 4 or any(
        not bool(event.get("enable_cuda_graph")) for event in graph_configs
    ):
        raise AnalysisError(f"expected CUDA graph on four Dealers in {case}/{repeat}")

    intervals = [
        (end - start) / 1e6 for start, end in zip(timestamps, timestamps[1:])
    ]
    timeline = [
        {
            "case": case,
            "topology": topology,
            "failure": failure,
            "repeat": repeat,
            "output_index": index,
            "accumulated_seconds": (end - timestamps[0]) / 1e9,
            "inter_token_ms": (end - start) / 1e6,
        }
        for index, (start, end) in enumerate(zip(timestamps, timestamps[1:]), 1)
    ]
    run_row = {
        "case": case,
        "topology": topology,
        "failure": failure,
        "repeat": repeat,
        "token_sha256": token_hash(tokens),
        "token_count": len(tokens),
        "decode_duration_seconds": (timestamps[-1] - timestamps[0]) / 1e9,
        "median_inter_token_ms": statistics.median(intervals),
        "p95_inter_token_ms": percentile(intervals, 0.95),
        "max_inter_token_ms": max(intervals),
        "failure_count": 0,
    }
    if not failure:
        if named(events, "failure_injected") or named(
            events, "active_winner_failure_command_send"
        ):
            raise AnalysisError(f"normal run {case}/{repeat} injected a failure")
        return run_row, timeline, None

    injection = exactly_one(events, "failure_injected")
    command = exactly_one(events, "active_winner_failure_command_send")
    if (
        int(injection["step"]) != drop_step
        or injection.get("injection_mode")
        != "agent_observed_active_winner_teardown"
        or str(command["dealer_node"]) != str(injection["dealer_node"])
        or int(command["previous_winner_step"]) != drop_step - 1
    ):
        raise AnalysisError(f"invalid active-winner injection: {injection}")

    accepted = [
        event for event in named(events, "infer_output_arrival")
        if bool(event.get("accepted"))
        and str(event.get("control_id", "")).startswith("bench-")
        and not bool(event.get("trace_invalid"))
    ]
    accepted_by_step: dict[int, dict[str, Any]] = {}
    for event in accepted:
        step = int(event["step"])
        if step in accepted_by_step:
            raise AnalysisError(f"more than one accepted output for step {step}")
        accepted_by_step[step] = event
    previous = accepted_by_step.get(drop_step - 1)
    current = accepted_by_step.get(drop_step)
    if previous is None or current is None:
        raise AnalysisError("missing accepted output around active-winner failure")
    failed_dealer = str(injection["dealer_node"])
    if str(previous.get("source_node")) != failed_dealer:
        raise AnalysisError("injected Dealer was not the observed previous winner")
    survivor = str(current.get("source_node"))
    if survivor == failed_dealer:
        raise AnalysisError("failed Dealer served the next step")

    expected_scope = "agent" if topology == "ordinary" else "backbone"
    expected_handler = str(run["agent_name"]) if topology == "ordinary" else "T0-backbone"
    recovery = correlate_recovery(
        events, injection, expected_scope, expected_handler
    )
    replacements = recovery["replacement_dealers"]
    if len(replacements) != 1 or replacements[0] in {failed_dealer, survivor}:
        raise AnalysisError(
            "recovery did not restore a physically distinct redundant path: "
            f"failed={failed_dealer}, survivor={survivor}, replacement={replacements}"
        )

    injection_time = int(injection["time_ns"])
    frontier_time = int(recovery["frontier"]["time_ns"])
    outputs_during = sum(
        injection_time <= int(event["time_ns"]) < frontier_time
        for event in accepted
    )
    next_output_before_frontier = int(current["time_ns"]) < frontier_time
    anchor_index = bisect.bisect_left(timestamps, injection_time) - 1
    last_before_frontier = bisect.bisect_left(timestamps, frontier_time) - 1
    if anchor_index < 0 or last_before_frontier <= anchor_index:
        raise AnalysisError("physical recovery window contains no service progress")
    recovery_row = {
        "case": case,
        "topology": topology,
        "repeat": repeat,
        "step": drop_step,
        "failed_dealer": failed_dealer,
        "survivor_dealer": survivor,
        "replacement_dealer": replacements[0],
        "recovery_id": recovery["recovery_id"],
        "recovery_scope": expected_scope,
        "handler_node": expected_handler,
        "injection_time_ns": injection_time,
        "frontier_time_ns": frontier_time,
        "physical_recovery_ms": (frontier_time - injection_time) / 1e6,
        "handoff_to_physical_ms": (
            frontier_time - int(recovery["handoff"]["time_ns"])
        ) / 1e6,
        "send_count": recovery["send_count"],
        "failed_was_previous_winner": True,
        "survivor_served_next_step": True,
        "next_output_before_frontier": next_output_before_frontier,
        "outputs_during_physical_recovery": outputs_during,
        "anchor_timestamp_index": anchor_index,
        "last_pre_frontier_timestamp_index": last_before_frontier,
        "normal_window_ms": None,
        "recover_window_ms": None,
        "cumulative_visible_excess_ms": None,
        "full_window_masked_fraction": None,
        "matched_window_p95_ratio": None,
        "matched_window_max_excess_ms": None,
    }
    run_row["failure_count"] = 1
    return run_row, timeline, recovery_row


def add_paired_window_metrics(
    timelines: list[dict[str, Any]],
    recoveries: list[dict[str, Any]],
) -> None:
    intervals: dict[tuple[str, bool, int], dict[int, float]] = {}
    for row in timelines:
        intervals.setdefault((
            str(row["topology"]), bool(row["failure"]), int(row["repeat"])
        ), {})[int(row["output_index"])] = float(row["inter_token_ms"])
    for row in recoveries:
        topology = str(row["topology"])
        repeat = int(row["repeat"])
        first_interval = int(row["anchor_timestamp_index"]) + 1
        last_interval = int(row["last_pre_frontier_timestamp_index"])
        indexes = list(range(first_interval, last_interval + 1))
        normal_map = intervals[(topology, False, repeat)]
        recover_map = intervals[(topology, True, repeat)]
        normal_values = [normal_map[index] for index in indexes]
        recover_values = [recover_map[index] for index in indexes]
        normal_window = sum(normal_values)
        recover_window = sum(recover_values)
        excess = max(0.0, recover_window - normal_window)
        physical = float(row["physical_recovery_ms"])
        row["normal_window_ms"] = normal_window
        row["recover_window_ms"] = recover_window
        row["cumulative_visible_excess_ms"] = excess
        row["full_window_masked_fraction"] = (
            1.0 - excess / physical if physical > 0 else 0.0
        )
        row["matched_window_p95_ratio"] = (
            percentile(recover_values, 0.95) / percentile(normal_values, 0.95)
        )
        row["matched_window_max_excess_ms"] = max(
            recover - normal
            for normal, recover in zip(normal_values, recover_values)
        )


def aggregate(
    runs: list[dict[str, Any]],
    recoveries: list[dict[str, Any]],
) -> dict[str, Any]:
    hashes = sorted({str(row["token_sha256"]) for row in runs})
    summaries: dict[str, Any] = {}
    for topology in ("ordinary", "backbone"):
        selected = [row for row in recoveries if row["topology"] == topology]
        if not selected:
            continue
        summaries[topology] = {
            "recovery_count": len(selected),
            "median_physical_recovery_ms": statistics.median(
                float(row["physical_recovery_ms"]) for row in selected
            ),
            "median_cumulative_visible_excess_ms": statistics.median(
                float(row["cumulative_visible_excess_ms"]) for row in selected
            ),
            "maximum_cumulative_visible_excess_ms": max(
                float(row["cumulative_visible_excess_ms"]) for row in selected
            ),
            "median_full_window_masked_fraction": statistics.median(
                float(row["full_window_masked_fraction"]) for row in selected
            ),
            "minimum_full_window_masked_fraction": min(
                float(row["full_window_masked_fraction"]) for row in selected
            ),
            "maximum_matched_window_p95_ratio": max(
                float(row["matched_window_p95_ratio"]) for row in selected
            ),
            "maximum_matched_window_max_excess_ms": max(
                float(row["matched_window_max_excess_ms"]) for row in selected
            ),
            "minimum_outputs_during_physical_recovery": min(
                int(row["outputs_during_physical_recovery"]) for row in selected
            ),
            "all_active_winners_failed": all(
                bool(row["failed_was_previous_winner"]) for row in selected
            ),
            "all_next_outputs_from_survivor": all(
                bool(row["survivor_served_next_step"]) for row in selected
            ),
            "all_next_outputs_precede_physical_recovery": all(
                bool(row["next_output_before_frontier"]) for row in selected
            ),
            "all_replacements_physically_distinct": all(
                len({
                    str(row["failed_dealer"]),
                    str(row["survivor_dealer"]),
                    str(row["replacement_dealer"]),
                }) == 3
                for row in selected
            ),
        }
    expected = {str(row["topology"]) for row in runs}
    overall_pass = (
        len(hashes) == 1
        and set(summaries) == expected
        and all(
            summary["all_active_winners_failed"]
            and summary["all_next_outputs_from_survivor"]
            and summary["all_next_outputs_precede_physical_recovery"]
            and summary["all_replacements_physically_distinct"]
            and summary["minimum_outputs_during_physical_recovery"] >= 1
            and summary["minimum_full_window_masked_fraction"] >= 0.80
            and summary["maximum_matched_window_p95_ratio"] <= 1.50
            and summary["maximum_matched_window_max_excess_ms"] <= 4.0
            for summary in summaries.values()
        )
    )
    return {
        "schema_version": 1,
        "claim": (
            "With zero artificial link delay, a surviving eager path keeps "
            "Re-Prefill decode service advancing while one observed winning "
            "path is physically rebuilt."
        ),
        "profile": "Re-Prefill, one recovery per run, zero artificial delay",
        "metric_note": (
            "full_window_masked_fraction uses cumulative service-time excess "
            "across every matched inter-token interval before frontier materialization"
        ),
        "all_token_sequences_identical": len(hashes) == 1,
        "token_hashes": hashes,
        "topologies": summaries,
        "overall_pass": overall_pass,
    }


def series_points(
    timelines: list[dict[str, Any]],
    topology: str,
    failure: bool,
) -> list[tuple[int, float, float]]:
    by_index: dict[int, list[dict[str, Any]]] = {}
    for row in timelines:
        if row["topology"] == topology and bool(row["failure"]) == failure:
            by_index.setdefault(int(row["output_index"]), []).append(row)
    return [
        (
            index,
            statistics.median(float(row["accumulated_seconds"]) for row in rows),
            statistics.median(float(row["inter_token_ms"]) for row in rows),
        )
        for index, rows in sorted(by_index.items())
    ]


def plot_topology(
    axes: tuple[Any, Any],
    topology: str,
    timelines: list[dict[str, Any]],
    recoveries: list[dict[str, Any]],
    legend: bool,
) -> None:
    upper, lower = axes
    for failure, label, color in (
        (False, "Normal / Re-Prefill", "#4c78a8"),
        (True, "Recover / Re-Prefill", "#f58518"),
    ):
        points = series_points(timelines, topology, failure)
        upper.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            label=label,
            color=color,
            linewidth=1.8,
        )
        lower.plot(
            [point[0] for point in points],
            [point[2] for point in points],
            color=color,
            linewidth=1.1,
            alpha=0.9,
        )
    selected = [row for row in recoveries if row["topology"] == topology]
    position = statistics.median(int(row["step"]) for row in selected)
    end = position + statistics.median(
        int(row["outputs_during_physical_recovery"]) for row in selected
    )
    for axis in axes:
        axis.axvspan(
            position,
            end,
            color="#bdbdbd",
            alpha=0.16,
            linewidth=0,
            label="Physical recovery" if axis is upper else None,
        )
        axis.axvline(position, color="#666666", linestyle=":", linewidth=0.9)
        axis.grid(True, alpha=0.2)
    upper.text(
        position,
        0.97,
        "active winner fails",
        transform=upper.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
    )
    excess = statistics.median(
        float(row["cumulative_visible_excess_ms"]) for row in selected
    )
    title = (
        "Ordinary eager multi-path / Agent recovery"
        if topology == "ordinary"
        else "Backbone-internal eager multi-path / local recovery"
    )
    upper.set_title(f"{title}\nmedian full-window excess: {excess:.2f} ms", fontsize=10)
    if legend:
        upper.legend(loc="upper left", fontsize=8, frameon=True)


def render_figures(
    result_root: Path,
    timelines: list[dict[str, Any]],
    recoveries: list[dict[str, Any]],
) -> None:
    figures = result_root / "figures"
    figures.mkdir(exist_ok=True)
    available = [
        topology for topology in ("ordinary", "backbone")
        if any(row["topology"] == topology for row in recoveries)
    ]
    for topology in available:
        fig, (upper, lower) = plt.subplots(
            2, 1,
            figsize=(9.4, 6.8),
            dpi=180,
            sharex=False,
            gridspec_kw={"height_ratios": [1.1, 1.0]},
        )
        plot_topology((upper, lower), topology, timelines, recoveries, True)
        upper.set_ylabel("Accumulated decode time (s)")
        upper.set_xlabel("Decode output index (full run)")
        lower.set_ylabel("Inter-token interval (ms)")
        lower.set_xlabel("Decode output index (failure window)")
        failure_step = statistics.median(
            int(row["step"]) for row in recoveries
            if row["topology"] == topology
        )
        window_end = failure_step + statistics.median(
            int(row["outputs_during_physical_recovery"]) for row in recoveries
            if row["topology"] == topology
        )
        lower.set_xlim(failure_step - 15, window_end + 15)
        fig.tight_layout()
        stem = f"{topology}_clean_recovery_timeline"
        fig.savefig(figures / f"{stem}.png", dpi=330)
        fig.savefig(figures / f"{stem}.pdf")
        plt.close(fig)

    if set(available) != {"ordinary", "backbone"}:
        return
    fig, axes = plt.subplots(
        2, 2,
        figsize=(15.4, 6.8),
        dpi=180,
        sharex=False,
        sharey="row",
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )
    plot_topology((axes[0, 0], axes[1, 0]), "ordinary", timelines, recoveries, False)
    plot_topology((axes[0, 1], axes[1, 1]), "backbone", timelines, recoveries, False)
    axes[0, 0].set_ylabel("Accumulated decode time (s)")
    axes[1, 0].set_ylabel("Inter-token interval (ms)")
    for topology, upper, lower in zip(
        ("ordinary", "backbone"), axes[0], axes[1]
    ):
        upper.set_xlabel("Decode output index (full run)")
        lower.set_xlabel("Decode output index (failure window)")
        selected = [row for row in recoveries if row["topology"] == topology]
        failure_step = statistics.median(int(row["step"]) for row in selected)
        window_end = failure_step + statistics.median(
            int(row["outputs_during_physical_recovery"]) for row in selected
        )
        lower.set_xlim(failure_step - 15, window_end + 15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        "Zero-delay Re-Prefill: service continues during one physical recovery",
        fontsize=11,
        y=0.935,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    fig.savefig(figures / "clean_multipath_recovery_timeline.png", dpi=330)
    fig.savefig(figures / "clean_multipath_recovery_timeline.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    manifest = read_json(result_root / "manifest.json")
    parameters = manifest["parameters"]
    drop_steps = [int(step) for step in parameters["drop_steps"]]
    if (
        parameters.get("experiment_profile")
        != "clean_reprefill_single_recovery_zero_delay"
        or parameters.get("recovery_replay") != "chunk"
        or float(parameters.get("artificial_network_latency_ms", -1)) != 0.0
        or len(drop_steps) != 1
    ):
        raise AnalysisError("result is not the clean formal profile")

    runs: list[dict[str, Any]] = []
    timelines: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    for run in manifest["runs"]:
        run_row, timeline, recovery = analyze_run(
            result_root, run, drop_steps[0]
        )
        runs.append(run_row)
        timelines.extend(timeline)
        if recovery is not None:
            recoveries.append(recovery)
    add_paired_window_metrics(timelines, recoveries)
    summary = aggregate(runs, recoveries)
    if not summary["overall_pass"]:
        raise AnalysisError(
            "clean pause-free assertions failed:\n"
            + json.dumps(summary, indent=2, ensure_ascii=False)
        )
    write_csv(result_root / "runs.csv", runs)
    write_csv(result_root / "timeline.csv", timelines)
    write_csv(result_root / "recoveries.csv", recoveries)
    write_json(result_root / "summary.json", summary)
    render_figures(result_root, timelines, recoveries)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
