#!/usr/bin/env python3
"""Validate and plot Backbone PEFT multi-path pause masking."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import matplotlib.pyplot as plt


LOSS_ATOL = 1e-5
LOSS_RTOL = 1e-4
LOSS_SMOOTHING_WINDOW = 8
BACKBONE = "T0-peft-backbone"
INITIAL_DEALERS = {
    "T0-peft-fast-active",
    "T1-peft-delayed-survivor",
}
REPLACEMENT_MAP = {
    "T0-peft-fast-active": "T2-peft-fast-recovery",
    "T2-peft-fast-recovery": "T0-peft-fast-active",
    "T1-peft-delayed-survivor": "T3-peft-delayed-recovery",
    "T3-peft-delayed-recovery": "T1-peft-delayed-survivor",
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_events(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError as error:
            raise AnalysisError(f"invalid JSONL at {path}:{line_number}") from error
    return sorted(rows, key=lambda row: int(row["time_ns"]))


def named(events: Iterable[dict[str, Any]], event: str) -> list[dict[str, Any]]:
    return [row for row in events if row.get("event") == event]


def exactly_one(
    events: Iterable[dict[str, Any]], event: str, **fields: Any,
) -> dict[str, Any]:
    rows = [
        row for row in events
        if row.get("event") == event
        and all(row.get(key) == value for key, value in fields.items())
    ]
    if len(rows) != 1:
        raise AnalysisError(
            f"expected one {event} with {fields}, found {len(rows)}"
        )
    return rows[0]


def ns(event: dict[str, Any]) -> int:
    return int(event["time_ns"])


def result_rows(run_dir: Path) -> list[dict[str, Any]]:
    values = read_json(run_dir / "raw" / "results.json")
    if len(values) != 2:
        raise AnalysisError(f"expected instance + global result in {run_dir}")
    instance = values[0]
    keys = ("history", "times", "start_times", "ids", "tokens")
    lengths = {key: len(instance[key]) for key in keys}
    if len(set(lengths.values())) != 1:
        raise AnalysisError(f"unaligned result vectors in {run_dir}: {lengths}")
    rows = []
    for index, item in enumerate(zip(
        instance["ids"], instance["history"], instance["start_times"],
        instance["times"], instance["tokens"],
    )):
        minibatch_id, loss, start, end, tokens = item
        rows.append({
            "completion_index": index,
            "minibatch_id": str(minibatch_id),
            "request_version": int(str(minibatch_id).split("-", 1)[0]),
            "loss": float(loss),
            "start_time_s": float(start),
            "completion_time_s": float(end),
            "latency_ms": (float(end) - float(start)) * 1000,
            "tokens": int(tokens),
        })
    if len({row["minibatch_id"] for row in rows}) != len(rows):
        raise AnalysisError(f"duplicate committed minibatches in {run_dir}")
    args = read_json(run_dir / "raw" / "args.json")["benchmark"]
    expected = math.ceil(
        int(args["samples_per_instance"]) / int(args["minibatch_size"])
    )
    if len(rows) != expected or int(args.get("num_paths", 0)) != 1:
        raise AnalysisError(
            f"invalid multi-path result shape in {run_dir}: "
            f"rows={len(rows)}, expected={expected}, args={args}"
        )
    network = read_json(run_dir / "raw" / "network.json")
    paths = network.get("selected_paths", [])
    normalized = {tuple(str(node) for node in path) for path in paths}
    expected_paths = {tuple([BACKBONE] * 6)}
    if normalized != expected_paths:
        raise AnalysisError(
            f"expected one outer Backbone path, found {paths}"
        )
    return rows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def analyze_recoveries(
    events: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    normal_gap_p99_ms: float,
    expected_failure_times: list[float],
) -> list[dict[str, Any]]:
    injections = sorted(
        named(events, "failure_injected"),
        key=lambda event: int(event["planned_failure_index"]),
    )
    commands = sorted(
        named(events, "active_winner_failure_command_send"),
        key=lambda event: int(event["planned_failure_index"]),
    )
    detections = named(events, "failure_detected")
    expected_count = len(expected_failure_times)
    if not (
        len(injections) == len(commands) == len(detections) == expected_count
    ):
        raise AnalysisError(
            "failure command/injection/detection counts differ: "
            f"commands={len(commands)}, injections={len(injections)}, "
            f"detections={len(detections)}, expected={expected_count}"
        )

    accepted_outputs = [
        event for event in named(events, "train_service_output_visible")
        if event.get("accepted") is True and event.get("origin") == "normal"
    ]
    if len(accepted_outputs) != len(rows):
        raise AnalysisError(
            f"expected {len(rows)} accepted outputs, found {len(accepted_outputs)}"
        )
    accepted_by_minibatch = {
        str(event["minibatch_id"]): event for event in accepted_outputs
    }
    if len(accepted_by_minibatch) != len(accepted_outputs):
        raise AnalysisError("more than one accepted output for a minibatch")
    completion_ns = sorted(int(row["completion_time_s"] * 1e9) for row in rows)

    recoveries: list[dict[str, Any]] = []
    for index, (planned_s, command, injection) in enumerate(zip(
        expected_failure_times, commands, injections,
    )):
        failed = str(injection["dealer_node"])
        if (
            int(command["planned_failure_index"]) != index
            or int(injection["planned_failure_index"]) != index
            or not math.isclose(float(injection["planned_elapsed_s"]), planned_s)
            or str(command["dealer_node"]) != failed
            or injection.get("injection_mode")
            != "agent_observed_active_winner_teardown"
        ):
            raise AnalysisError(f"invalid active-winner failure {index}: {injection}")
        winner_minibatch = str(command["previous_winner_minibatch_id"])
        winner = accepted_by_minibatch.get(winner_minibatch)
        if winner is None or str(winner.get("dealer_node")) != failed:
            raise AnalysisError(
                f"failure {index} did not target the observed accepted winner"
            )
        matching_detections = [
            event for event in detections
            if event.get("compute_node") == injection.get("compute_node")
            and ns(event) >= ns(injection)
        ]
        if len(matching_detections) != 1:
            raise AnalysisError(
                f"failure {index} has {len(matching_detections)} detections"
            )
        detection = matching_detections[0]
        if (
            detection.get("handler_node") != BACKBONE
            or detection.get("recovery_scope") != "backbone"
        ):
            raise AnalysisError(f"failure escaped Backbone: {detection}")
        recovery_id = str(detection["recovery_id"])
        snapshot = exactly_one(
            events, "shadow_state_snapshot", recovery_id=recovery_id,
            state_type="train",
        )
        lookup_start = exactly_one(
            events, "replacement_lookup_start", recovery_id=recovery_id
        )
        lookup_end = exactly_one(
            events, "replacement_lookup_end", recovery_id=recovery_id
        )
        graph = exactly_one(
            events, "replacement_graph_installed", recovery_id=recovery_id
        )
        handoff = exactly_one(
            events, "promise_handoff_committed", recovery_id=recovery_id
        )
        handler_return = exactly_one(
            events, "recovery_handler_return", recovery_id=recovery_id
        )
        state_load_end = exactly_one(
            events, "replacement_state_load_end", recovery_id=recovery_id,
            state_type="train",
        )
        physical = exactly_one(
            events, "train_physical_recovery_complete", recovery_id=recovery_id
        )
        replacement_dealer = REPLACEMENT_MAP.get(failed)
        if lookup_end.get("replacement") != [[replacement_dealer, 0, 6]]:
            raise AnalysisError(
                f"failure {index} selected wrong replacement: {lookup_end}"
            )
        ordered = [
            ns(injection), ns(detection), ns(snapshot), ns(lookup_start),
            ns(lookup_end), ns(graph), ns(handoff), ns(handler_return),
            ns(physical),
        ]
        if ordered != sorted(ordered) or ns(state_load_end) > ns(physical):
            raise AnalysisError(f"invalid recovery order for failure {index}")

        sends = [
            event for event in named(events, "recovery_packet_send_start")
            if event.get("recovery_id") == recovery_id
        ]
        enqueues = [
            event for event in named(events, "recovery_packet_enqueue_accepted")
            if event.get("recovery_id") == recovery_id
        ]
        if (
            not enqueues
            or len(sends) != len(enqueues)
            or len(enqueues) != int(handoff["required_packet_count"])
            or max(ns(event) for event in enqueues) > ns(handoff)
        ):
            raise AnalysisError(f"incomplete handoff for failure {index}")

        survivor_outputs = [
            event for event in accepted_outputs
            if ns(injection) <= ns(event) < ns(physical)
            and str(event.get("dealer_node")) != failed
        ]
        if not survivor_outputs:
            raise AnalysisError(
                f"no surviving-path output during recovery {index}"
            )
        window_points = [
            value for value in completion_ns
            if ns(injection) <= value < ns(physical)
        ]
        boundary_points = [
            max((value for value in completion_ns if value < ns(injection)), default=None),
            *window_points,
            min((value for value in completion_ns if value >= ns(physical)), default=None),
        ]
        boundary_points = [value for value in boundary_points if value is not None]
        gaps_ms = [
            (end - start) / 1e6
            for start, end in zip(boundary_points, boundary_points[1:])
        ]
        max_gap_ms = max(gaps_ms)
        physical_ms = (ns(physical) - ns(injection)) / 1e6
        visible_excess_ms = max(0.0, max_gap_ms - normal_gap_p99_ms)
        masked_fraction = max(0.0, 1.0 - visible_excess_ms / physical_ms)
        if max_gap_ms >= physical_ms or masked_fraction < 0.75:
            raise AnalysisError(
                f"recovery {index} was not masked: gap={max_gap_ms:.3f}ms, "
                f"physical={physical_ms:.3f}ms, masked={masked_fraction:.3f}"
            )
        survivor_dealers = sorted({
            str(event["dealer_node"]) for event in survivor_outputs
        })
        recoveries.append({
            "failure_index": index,
            "planned_failure_time_s": planned_s,
            "recovery_id": recovery_id,
            "failed_dealer": failed,
            "replacement_dealer": replacement_dealer,
            "surviving_dealers": survivor_dealers,
            "failure_injected_time_ns": ns(injection),
            "failure_detected_time_ns": ns(detection),
            "promise_handoff_time_ns": ns(handoff),
            "physical_recovery_complete_time_ns": ns(physical),
            "injection_to_detection_ms": (ns(detection) - ns(injection)) / 1e6,
            "handoff_to_physical_ms": (ns(physical) - ns(handoff)) / 1e6,
            "physical_recovery_ms": physical_ms,
            "outputs_during_physical_recovery": len(window_points),
            "accepted_survivor_outputs_during_recovery": len(survivor_outputs),
            "normal_completion_gap_p99_ms": normal_gap_p99_ms,
            "max_completion_gap_during_recovery_ms": max_gap_ms,
            "visible_gap_excess_ms": visible_excess_ms,
            "stall_masked_fraction": masked_fraction,
            "failed_was_observed_winner": True,
            "failure_local_to_backbone": True,
            "replacement_isolated_from_survivor_device": True,
        })
    return recoveries


def plot_overview(
    result_root: Path,
    minibatches: list[dict[str, Any]],
    recoveries: list[dict[str, Any]],
    max_loss_error: float,
) -> None:
    normal = [row for row in minibatches if row["condition"] == "normal"]
    recovered = [row for row in minibatches if row["condition"] == "recover"]
    selected = (
        ("Normal — Backbone + 2 paths", normal, "#4c78a8", "-"),
        ("Recover — Backbone + 2 paths", recovered, "#f58518", "--"),
    )
    fig, (loss_axis, completed, latency) = plt.subplots(
        3, 1, figsize=(11.5, 8.5), sharex=True,
        gridspec_kw={"height_ratios": (1.15, 1.0, 1.0)},
    )
    bases = {
        condition: min(float(row["start_time_s"]) for row in rows)
        for condition, rows in (("normal", normal), ("recover", recovered))
    }
    for label, rows, color, style in selected:
        condition = str(rows[0]["condition"])
        base = bases[condition]
        timed = sorted(rows, key=lambda row: float(row["completion_time_s"]))
        xs = [float(row["completion_time_s"]) - base for row in timed]
        losses = [float(row["loss"]) for row in timed]
        loss_axis.plot(
            xs, losses, ".", color=color, alpha=.20, markersize=2,
            label=f"{label} raw",
        )
        window = min(LOSS_SMOOTHING_WINDOW, len(timed))
        rolling = [
            statistics.mean(losses[max(0, index - window + 1):index + 1])
            for index in range(len(losses))
        ]
        loss_axis.plot(
            xs, rolling, color=color, linestyle=style, linewidth=2,
            label=f"{label} rolling mean (window={window})",
        )
        completed.step(
            sorted(xs), range(1, len(xs) + 1), where="post",
            color=color, linestyle=style, linewidth=1.8, label=label,
        )
        latency.plot(
            xs, [float(row["latency_ms"]) for row in timed], ".",
            color=color, alpha=.72, markersize=3, label=label,
        )

    recover_base = bases["recover"]
    for recovery in recoveries:
        failure_x = recovery["failure_injected_time_ns"] / 1e9 - recover_base
        physical_x = (
            recovery["physical_recovery_complete_time_ns"] / 1e9 - recover_base
        )
        for axis in (loss_axis, completed, latency):
            axis.axvspan(
                failure_x, physical_x, color="#bdbdbd", alpha=.17, linewidth=0,
            )
            axis.axvline(
                failure_x, color="#e45756", linestyle="--", alpha=.72,
            )
        loss_axis.annotate(
            f"F{int(recovery['failure_index']) + 1}",
            (failure_x, loss_axis.get_ylim()[1]),
            xytext=(3, -14), textcoords="offset points",
            color="#b22222", fontsize=9,
        )
        completed.text(
            (failure_x + physical_x) / 2,
            0.05,
            (
                f"{recovery['physical_recovery_ms'] / 1000:.1f}s\n"
                f"{recovery['accepted_survivor_outputs_during_recovery']} outputs"
            ),
            transform=completed.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=7, color="#555555",
        )
    loss_axis.set_ylabel("Training loss")
    completed.set_ylabel("Completed minibatches")
    latency.set_ylabel("Minibatch latency (ms)")
    latency.set_xlabel("Wall time from each run start (s)")
    loss_axis.legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
    completed.legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle(
        "Backbone multi-path masks the failed branch's physical recovery stall — "
        f"max semantic |Δloss| = {max_loss_error:.2g}, "
        f"worst stall masked = "
        f"{min(row['stall_masked_fraction'] for row in recoveries) * 100:.1f}%",
        fontsize=12,
    )
    for axis in (loss_axis, completed, latency):
        axis.grid(axis="y", alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .96))
    figures = result_root / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "peft_recovery_overview.png", dpi=330)
    fig.savefig(figures / "peft_recovery_overview.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    manifest = read_json(result_root / "manifest.json")
    if len(manifest["runs"]) != 2:
        raise AnalysisError("formal result requires one Normal and one Recover run")

    all_rows: list[dict[str, Any]] = []
    events_by_condition: dict[str, list[dict[str, Any]]] = {}
    for run in manifest["runs"]:
        if run.get("benchmark_returncode") != 0 or run.get("error") is not None:
            raise AnalysisError(f"manifest marks run unsuccessful: {run}")
        condition = str(run["condition"])
        run_dir = result_root / run["run_dir"]
        rows = result_rows(run_dir)
        for row in rows:
            row["condition"] = condition
            row["repeat"] = int(run["repeat"])
        all_rows.extend(rows)
        events = read_events(run_dir / "events.jsonl")
        events_by_condition[condition] = events
        if condition == "normal" and named(events, "failure_injected"):
            raise AnalysisError("Normal unexpectedly injected a failure")

    if set(events_by_condition) != {"normal", "recover"}:
        raise AnalysisError("missing Normal or Recover condition")
    normal = [row for row in all_rows if row["condition"] == "normal"]
    recovered = [row for row in all_rows if row["condition"] == "recover"]
    normal_by_id = {row["minibatch_id"]: row for row in normal}
    recovered_by_id = {row["minibatch_id"]: row for row in recovered}
    if set(normal_by_id) != set(recovered_by_id):
        raise AnalysisError("Normal and Recover committed different minibatches")
    loss_errors = [
        abs(float(normal_by_id[key]["loss"]) - float(recovered_by_id[key]["loss"]))
        for key in normal_by_id
    ]
    violations = [
        key for key in normal_by_id
        if not math.isclose(
            float(normal_by_id[key]["loss"]),
            float(recovered_by_id[key]["loss"]),
            rel_tol=LOSS_RTOL, abs_tol=LOSS_ATOL,
        )
    ]
    if violations:
        raise AnalysisError(f"loss equivalence failed for {len(violations)} minibatches")

    normal_times = sorted(float(row["completion_time_s"]) for row in normal)
    normal_gaps_ms = [
        (end - start) * 1000
        for start, end in zip(normal_times, normal_times[1:])
    ]
    normal_gap_p99_ms = percentile(normal_gaps_ms, .99)
    expected_failure_times = [
        float(value) for value in manifest["parameters"]["failure_times_s"]
    ]
    recoveries = analyze_recoveries(
        events_by_condition["recover"], recovered, normal_gap_p99_ms,
        expected_failure_times,
    )
    normal_base = min(float(row["start_time_s"]) for row in normal)
    recover_base = min(float(row["start_time_s"]) for row in recovered)
    for recovery in recoveries:
        window_start = (
            float(recovery["failure_injected_time_ns"]) / 1e9 - recover_base
        )
        window_end = (
            float(recovery["physical_recovery_complete_time_ns"]) / 1e9
            - recover_base
        )
        recover_latencies = [
            float(row["latency_ms"]) for row in recovered
            if window_start
            <= float(row["completion_time_s"]) - recover_base
            < window_end
        ]
        normal_latencies = [
            float(row["latency_ms"]) for row in normal
            if window_start
            <= float(row["completion_time_s"]) - normal_base
            < window_end
        ]
        if recover_latencies and normal_latencies:
            recover_median = statistics.median(recover_latencies)
            normal_median = statistics.median(normal_latencies)
            recovery["recover_window_median_latency_ms"] = recover_median
            recovery["normal_aligned_window_median_latency_ms"] = normal_median
            recovery["median_latency_increase_ms"] = (
                recover_median - normal_median
            )
    summary = {
        "schema_version": 1,
        "claim": (
            "A surviving Backbone-internal path keeps PEFT minibatches "
            "completing while each failed branch is rebuilt and replayed."
        ),
        "validated": True,
        "all_minibatches_equal": True,
        "max_loss_abs_error": max(loss_errors, default=0.0),
        "recovery_count": len(recoveries),
        "worst_stall_masked_fraction": min(
            row["stall_masked_fraction"] for row in recoveries
        ),
        "recoveries": recoveries,
    }
    write_csv(result_root / "tables" / "minibatches.csv", all_rows)
    write_csv(result_root / "tables" / "recoveries.csv", recoveries)
    write_json(result_root / "summary.json", summary)
    write_json(result_root / "correctness.json", {
        "minibatch_set_equal": True,
        "loss_atol": LOSS_ATOL,
        "loss_rtol": LOSS_RTOL,
        "max_loss_abs_error": max(loss_errors, default=0.0),
        "passed": True,
    })
    plot_overview(
        result_root, all_rows, recoveries, max(loss_errors, default=0.0)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
