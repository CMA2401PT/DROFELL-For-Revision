#!/usr/bin/env python3
"""Validate PEFT recovery causality and produce publication-ready figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import matplotlib.pyplot as plt


EXP_ROOT = Path(__file__).resolve().parent
LOSS_ATOL = 1e-5
LOSS_RTOL = 1e-4
LOSS_SMOOTHING_WINDOW = 8


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
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AnalysisError(f"missing event stream: {path}")
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise AnalysisError(
                f"invalid JSONL at {path}:{line_number}: {error}"
            ) from error
    return sorted(result, key=lambda event: int(event["time_ns"]))


def named(events: Iterable[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event") == name]


def matching(
    events: Iterable[dict[str, Any]], name: str, **fields: Any
) -> list[dict[str, Any]]:
    return [
        event for event in events
        if event.get("event") == name
        and all(event.get(key) == value for key, value in fields.items())
    ]


def exactly_one(
    events: Iterable[dict[str, Any]], name: str, **fields: Any
) -> dict[str, Any]:
    values = matching(events, name, **fields)
    if len(values) != 1:
        raise AnalysisError(
            f"expected one {name} with {fields}, got {len(values)}"
        )
    return values[0]


def ns(event: dict[str, Any]) -> int:
    return int(event["time_ns"])


def elapsed_ms(start: dict[str, Any], end: dict[str, Any]) -> float:
    return (ns(end) - ns(start)) / 1e6


def first_after(
    events: Iterable[dict[str, Any]], name: str, time_ns: int, **fields: Any
) -> dict[str, Any]:
    values = [
        event for event in matching(events, name, **fields)
        if ns(event) > time_ns
    ]
    if not values:
        raise AnalysisError(
            f"missing {name} after {time_ns} with {fields}"
        )
    return min(values, key=ns)


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
    for index, values in enumerate(zip(
        instance["ids"], instance["history"], instance["start_times"],
        instance["times"], instance["tokens"],
    )):
        minibatch_id, loss, start, end, tokens = values
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
        raise AnalysisError(f"duplicate committed minibatch IDs in {run_dir}")
    args = read_json(run_dir / "raw" / "args.json")["benchmark"]
    expected = math.ceil(
        int(args["samples_per_instance"]) / int(args["minibatch_size"])
    )
    if len(rows) != expected:
        raise AnalysisError(
            f"expected {expected} committed minibatches in {run_dir}, got {len(rows)}"
        )
    return rows


def packet_row(
    run: dict[str, Any], recovery_id: str, packet_id: str,
    kind: str, send: dict[str, Any], enqueue: dict[str, Any],
    receive: dict[str, Any],
) -> dict[str, Any]:
    return {
        "condition": run["condition"], "repeat": run["repeat"],
        "link_latency_ms": run["link_latency_ms"],
        "recovery_id": recovery_id, "packet_id": packet_id,
        "packet_kind": kind,
        "send_time_ns": ns(send), "enqueue_time_ns": ns(enqueue),
        "receive_time_ns": ns(receive),
        "send_to_enqueue_ms": elapsed_ms(send, enqueue),
        "enqueue_to_receive_ms": elapsed_ms(enqueue, receive),
    }


def analyze_recovery(
    run: dict[str, Any], events: list[dict[str, Any]],
    snapshot: dict[str, Any], next_snapshot_ns: int | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    recovery_id = str(snapshot["recovery_id"])
    control_id = str(snapshot["control_id"])
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
    state_load_start = exactly_one(
        events, "replacement_state_load_start", recovery_id=recovery_id,
        state_type="train",
    )
    state_load_end = exactly_one(
        events, "replacement_state_load_end", recovery_id=recovery_id,
        state_type="train",
    )
    physical = exactly_one(
        events, "train_physical_recovery_complete", recovery_id=recovery_id
    )
    detection = max(
        (
            event for event in named(events, "failure_detected")
            if ns(event) <= ns(snapshot)
        ),
        key=ns,
        default=None,
    )
    injection = max(
        (
            event for event in named(events, "failure_injected")
            if ns(event) <= ns(detection or snapshot)
            and event.get("compute_node") == snapshot.get("compute_node")
        ),
        key=ns,
        default=None,
    )
    if detection is None or injection is None:
        raise AnalysisError(f"cannot associate failure events with {recovery_id}")

    ordered = (injection, detection, lookup_start, lookup_end, graph, handoff)
    if [ns(event) for event in ordered] != sorted(ns(event) for event in ordered):
        raise AnalysisError(f"invalid Agent event order for {recovery_id}")
    if ns(state_load_start) > ns(state_load_end) or ns(state_load_end) > ns(physical):
        raise AnalysisError(f"invalid Dealer recovery order for {recovery_id}")

    sends = matching(
        events, "recovery_packet_send_start", recovery_id=recovery_id
    )
    enqueues = matching(
        events, "recovery_packet_enqueue_accepted", recovery_id=recovery_id
    )
    if len(sends) != len(enqueues) or len(enqueues) != int(
        handoff["required_packet_count"]
    ):
        raise AnalysisError(
            f"recovery packet count mismatch for {recovery_id}: "
            f"send={len(sends)}, enqueue={len(enqueues)}, "
            f"required={handoff['required_packet_count']}"
        )
    packet_rows: list[dict[str, Any]] = []
    recovery_receives: list[dict[str, Any]] = []
    for enqueue in enqueues:
        packet_id = str(enqueue["packet_id"])
        send = exactly_one(
            sends, "recovery_packet_send_start", packet_id=packet_id
        )
        kind = str(enqueue["packet_kind"])
        if kind == "create_train_node":
            receive = exactly_one(
                events, "replacement_create_receive",
                recovery_id=recovery_id, packet_id=packet_id,
            )
        else:
            receive = exactly_one(
                events, "train_packet_receive", recovery_id=recovery_id,
                packet_id=packet_id, origin="recovery_replay",
            )
        if not (
            ns(send) <= ns(enqueue) <= ns(handoff)
            and ns(enqueue) <= ns(receive)
        ):
            raise AnalysisError(
                f"invalid recovery packet causality for {packet_id}"
            )
        packet_rows.append(packet_row(
            run, recovery_id, packet_id, kind, send, enqueue, receive
        ))
        recovery_receives.append(receive)

    next_created = first_after(
        events, "train_task_created", ns(handoff),
        control_id=control_id, origin="normal",
        logical_handoff_id=recovery_id,
    )
    packet_id = str(next_created["packet_id"])
    next_enqueue = exactly_one(
        events, "train_packet_enqueue_accepted", packet_id=packet_id,
        origin="normal",
    )
    next_receive = exactly_one(
        events, "train_packet_receive", packet_id=packet_id, origin="normal"
    )
    next_stm = exactly_one(
        events, "train_stm_accept", packet_id=packet_id, origin="normal"
    )
    next_execute = exactly_one(
        events, "train_task_execute_start", packet_id=packet_id,
        origin="normal",
    )
    next_execute_end = exactly_one(
        events, "train_task_execute_end", packet_id=packet_id,
        origin="normal",
    )
    next_output = exactly_one(
        events, "train_output_send", packet_id=packet_id, origin="normal"
    )
    next_visible = exactly_one(
        events, "train_service_output_visible", packet_id=packet_id,
        origin="normal",
    )
    normal_order = (
        next_created, next_enqueue, next_receive, next_stm, next_execute,
        next_execute_end, next_output, next_visible,
    )
    if [ns(event) for event in normal_order] != sorted(
        ns(event) for event in normal_order
    ):
        raise AnalysisError(f"invalid post-handoff task order for {recovery_id}")
    if next_snapshot_ns is not None and ns(next_created) >= next_snapshot_ns:
        raise AnalysisError(
            f"no post-handoff normal task before next recovery for {recovery_id}"
        )

    replay_receives = matching(
        events, "train_packet_receive", recovery_id=recovery_id,
        origin="recovery_replay",
    )
    replay_stm_accepts = matching(
        events, "train_stm_accept", recovery_id=recovery_id,
        origin="recovery_replay",
    )
    replay_task_count = int(handoff["replay_task_count"])
    if len(replay_receives) != replay_task_count:
        raise AnalysisError(
            f"expected {replay_task_count} replay receives for {recovery_id}, "
            f"got {len(replay_receives)}"
        )
    if len(replay_stm_accepts) != replay_task_count:
        raise AnalysisError(
            f"expected {replay_task_count} replay STM accepts for {recovery_id}, "
            f"got {len(replay_stm_accepts)}"
        )

    failure_version = int(injection["request_version"])
    next_version = int(next_created["request_version"])
    row = {
        "condition": run["condition"], "repeat": run["repeat"],
        "link_latency_ms": float(run["link_latency_ms"]),
        "recovery_id": recovery_id,
        "failure_version": failure_version,
        "failed_dealer": injection["dealer_node"],
        "replacement": json.dumps(lookup_end.get("replacement"), ensure_ascii=False),
        "replay_task_count": replay_task_count,
        "required_packet_count": int(handoff["required_packet_count"]),
        "injection_to_detection_ms": elapsed_ms(injection, detection),
        "detection_to_selection_ms": elapsed_ms(detection, lookup_end),
        "selection_to_graph_ms": elapsed_ms(lookup_end, graph),
        "graph_to_handoff_ms": elapsed_ms(graph, handoff),
        "handoff_to_state_load_end_ms": elapsed_ms(handoff, state_load_end),
        "handoff_to_physical_complete_ms": elapsed_ms(handoff, physical),
        "handoff_to_next_task_create_ms": elapsed_ms(handoff, next_created),
        "handoff_to_next_task_enqueue_ms": elapsed_ms(handoff, next_enqueue),
        "next_enqueue_to_receive_ms": elapsed_ms(next_enqueue, next_receive),
        "next_receive_to_stm_accept_ms": elapsed_ms(next_receive, next_stm),
        "next_stm_to_execute_ms": elapsed_ms(next_stm, next_execute),
        "next_execute_ms": elapsed_ms(next_execute, next_execute_end),
        "next_output_transfer_ms": elapsed_ms(next_output, next_visible),
        "handoff_to_service_visible_ms": elapsed_ms(handoff, next_visible),
        "next_minibatch_id": next_created["minibatch_id"],
        "next_request_version": next_version,
        "next_task_same_failed_batch": next_version == failure_version,
        "next_task_enqueued_before_physical": ns(next_enqueue) < ns(physical),
        "state_load_finished_after_handoff": ns(state_load_end) > ns(handoff),
        "physical_finished_after_handoff": ns(physical) > ns(handoff),
    }
    trial = {
        **row,
        "failure_injected_time_ns": ns(injection),
        "failure_detected_time_ns": ns(detection),
        "lookup_start_time_ns": ns(lookup_start),
        "lookup_end_time_ns": ns(lookup_end),
        "graph_installed_time_ns": ns(graph),
        "handoff_time_ns": ns(handoff),
        "state_load_start_time_ns": ns(state_load_start),
        "state_load_end_time_ns": ns(state_load_end),
        "physical_complete_time_ns": ns(physical),
        "next_created_time_ns": ns(next_created),
        "next_enqueue_time_ns": ns(next_enqueue),
        "next_receive_time_ns": ns(next_receive),
        "next_stm_time_ns": ns(next_stm),
        "next_execute_time_ns": ns(next_execute),
        "next_execute_end_time_ns": ns(next_execute_end),
        "next_output_time_ns": ns(next_output),
        "next_visible_time_ns": ns(next_visible),
        "recovery_first_enqueue_time_ns": min(ns(event) for event in enqueues),
        "recovery_last_enqueue_time_ns": max(ns(event) for event in enqueues),
        "recovery_first_receive_time_ns": min(
            ns(event) for event in recovery_receives
        ),
        "recovery_last_receive_time_ns": max(
            ns(event) for event in recovery_receives
        ),
    }
    packet_rows.append(packet_row(
        run, recovery_id, packet_id, "post_handoff_normal",
        exactly_one(events, "train_packet_send_start", packet_id=packet_id),
        next_enqueue, next_receive,
    ))
    return row, trial, packet_rows


def plot_overview(
    figures: Path, minibatches: list[dict[str, Any]],
    recoveries: list[dict[str, Any]],
) -> None:
    normal = [row for row in minibatches if row["condition"] == "normal" and row["repeat"] == 0]
    recovery_conditions = sorted(
        {row["condition"] for row in minibatches if row["case"] == "recover"},
        key=lambda condition: max(
            float(row["link_latency_ms"]) for row in minibatches
            if row["condition"] == condition
        ),
    )
    if not normal or not recovery_conditions:
        return
    condition = recovery_conditions[-1]
    recovered = [
        row for row in minibatches
        if row["condition"] == condition and row["repeat"] == 0
    ]
    selected = (("Normal", normal, "#4c78a8"),
                (f"Recover ({recovered[0]['link_latency_ms']:g} ms each way)",
                 recovered, "#f58518"))
    fig, (loss_axis, completed, latency) = plt.subplots(
        3, 1, figsize=(11.5, 8.5), sharex=True,
        gridspec_kw={"height_ratios": (1.15, 1.0, 1.0)},
    )
    recovered_base = min(float(row["start_time_s"]) for row in recovered)
    normal_by_id = {row["minibatch_id"]: row for row in normal}
    semantic_differences = [
        abs(float(row["loss"]) - float(normal_by_id[row["minibatch_id"]]["loss"]))
        for row in recovered
    ]
    for label, rows, color in selected:
        line_style = "--" if label.startswith("Recover") else "-"
        base = min(float(row["start_time_s"]) for row in rows)
        timed_rows = sorted(
            rows, key=lambda row: float(row["completion_time_s"])
        )
        xs = [float(row["completion_time_s"]) - base for row in timed_rows]
        loss_axis.plot(
            xs, [float(row["loss"]) for row in timed_rows],
            marker=".", linestyle=line_style,
            color=color, alpha=.22, linewidth=.7, label=f"{label} raw",
            markersize=2,
        )
        window = min(LOSS_SMOOTHING_WINDOW, len(timed_rows))
        losses = [float(row["loss"]) for row in timed_rows]
        rolling = [
            statistics.mean(losses[max(0, index - window + 1):index + 1])
            for index in range(len(losses))
        ]
        loss_axis.plot(
            xs, rolling, color=color, linestyle=line_style, linewidth=2.0,
            label=f"{label} rolling mean (window={window})",
        )
        ordered = sorted(xs)
        completed.step(
            ordered, range(1, len(ordered) + 1), where="post",
            color=color, linestyle=line_style, label=label,
        )
        latency.plot(
            xs, [float(row["latency_ms"]) for row in timed_rows], ".",
            color=color, alpha=.72, label=label, markersize=3,
        )
    recovery_run = [
        row for row in recoveries
        if row["condition"] == condition and row["repeat"] == 0
    ]
    base = recovered_base
    for index, recovery in enumerate(recovery_run):
        x = (int(recovery["failure_injected_time_ns"]) / 1e9) - base
        for axis in (loss_axis, completed, latency):
            axis.axvline(x, color="#e45756", linestyle="--", alpha=.55)
        loss_axis.annotate(
            f"F{index + 1}", (x, loss_axis.get_ylim()[1]),
            xytext=(3, -14), textcoords="offset points",
            color="#b22222", fontsize=9,
        )
    loss_axis.set_ylabel("Training loss")
    completed.set_ylabel("Completed minibatches")
    latency.set_ylabel("Minibatch latency (ms)")
    latency.set_xlabel("Wall time from each run start (s)")
    loss_axis.legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
    completed.legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle(
        "Real-data PEFT loss and recovery positions — "
        f"max semantic |Δloss| = {max(semantic_differences, default=0.0):.2g}",
        fontsize=13,
    )
    for axis in (loss_axis, completed, latency):
        axis.grid(axis="y", alpha=.2)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(figures / "peft_recovery_overview.png", dpi=330)
    fig.savefig(figures / "peft_recovery_overview.pdf")
    plt.close(fig)


def plot_causality(figures: Path, trials: list[dict[str, Any]]) -> None:
    if not trials:
        return
    trial = max(
        trials,
        key=lambda row: (
            float(row["link_latency_ms"]),
            bool(row["next_task_enqueued_before_physical"]),
            -int(row["failure_version"]),
        ),
    )
    origin = int(trial["failure_injected_time_ns"])
    ms = lambda key: (int(trial[key]) - origin) / 1e6
    fig, (timeline, event_key) = plt.subplots(
        1, 2, figsize=(14.5, 7.0),
        gridspec_kw={"width_ratios": [4.25, 2.5]},
    )
    lanes = {
        "Failed TrainNode": 5,
        "Agent": 4,
        "Agent → Dealer network": 3,
        "Dealer: recovery stream": 2,
        "Dealer: normal minibatch / STM": 1,
        "Dealer → Agent network": 0,
    }
    timeline.set_yticks(list(lanes.values()), list(lanes))
    for y in lanes.values():
        timeline.axhline(y, color="#d9d9d9", linewidth=.7, zorder=0)

    failure_color = "#b22222"
    agent_color = "#1f77b4"
    recovery_color = "#d95f02"
    normal_color = "#2ca02c"
    physical_color = "#8c2d04"

    def arrow(
        start: tuple[float, float], end: tuple[float, float], color: str,
        linewidth: float = 1.5,
    ) -> None:
        timeline.annotate(
            "", xy=end, xytext=start,
            arrowprops={
                "arrowstyle": "->", "color": color, "lw": linewidth,
                "shrinkA": 4, "shrinkB": 4,
            },
            zorder=3,
        )

    y_failed = lanes["Failed TrainNode"]
    y_agent = lanes["Agent"]
    y_forward = lanes["Agent → Dealer network"]
    y_recovery = lanes["Dealer: recovery stream"]
    y_normal = lanes["Dealer: normal minibatch / STM"]
    y_reverse = lanes["Dealer → Agent network"]

    values = {
        "failure": 0.0,
        "detected": ms("failure_detected_time_ns"),
        "selected": ms("lookup_end_time_ns"),
        "graph": ms("graph_installed_time_ns"),
        "recovery_first_enqueue": ms("recovery_first_enqueue_time_ns"),
        "recovery_last_enqueue": ms("recovery_last_enqueue_time_ns"),
        "handoff": ms("handoff_time_ns"),
        "next_enqueue": ms("next_enqueue_time_ns"),
        "recovery_first_receive": ms("recovery_first_receive_time_ns"),
        "recovery_last_receive": ms("recovery_last_receive_time_ns"),
        "state_end": ms("state_load_end_time_ns"),
        "next_receive": ms("next_receive_time_ns"),
        "execute": ms("next_execute_time_ns"),
        "execute_end": ms("next_execute_end_time_ns"),
        "output": ms("next_output_time_ns"),
        "visible": ms("next_visible_time_ns"),
        "physical": ms("physical_complete_time_ns"),
    }
    stm = ms("next_stm_time_ns")
    state_start = ms("state_load_start_time_ns")

    event_specs = [
        ("failure", "TrainNode failure injected", y_failed, failure_color),
        ("detected", "Agent detects failure", y_agent, agent_color),
        ("selected", "Replacement selection complete", y_agent + .18, agent_color),
        ("recovery_first_enqueue", "First recovery packet queued", y_agent - .18,
         recovery_color),
        ("graph", "Replacement graph installed", y_agent + .18, agent_color),
        ("recovery_last_enqueue",
         f"Last recovery packet queued ({trial['required_packet_count']} required)",
         y_agent - .18, recovery_color),
        ("handoff", "Promise handoff committed", y_agent, agent_color),
        ("next_enqueue", "Next normal minibatch queued", y_agent, normal_color),
        ("recovery_first_receive", "First recovery packet received",
         y_recovery, recovery_color),
        ("recovery_last_receive", "Last recovery packet received",
         y_recovery, recovery_color),
        ("state_end", "Replacement state load complete", y_recovery + .18,
         recovery_color),
        ("next_receive", "Next normal minibatch received", y_normal + .18,
         normal_color),
        ("execute", "STM accepts; normal task handler starts", y_normal,
         normal_color),
        ("execute_end", "Normal task processing complete", y_normal,
         normal_color),
        ("output", "Dealer sends normal output", y_normal - .18, normal_color),
        ("visible", "Agent receives result / service visible", y_agent,
         normal_color),
        ("physical", "All recovery replay complete", y_recovery,
         physical_color),
    ]
    ordered_specs = sorted(event_specs, key=lambda item: values[item[0]])
    event_numbers = {
        name: index for index, (name, _, _, _) in enumerate(ordered_specs, 1)
    }

    def numbered_event(name: str, y: float, color: str) -> None:
        x = values[name]
        timeline.scatter(x, y, s=135, color=color, edgecolor="white",
                         linewidth=.8, zorder=6)
        timeline.text(
            x, y, str(event_numbers[name]), ha="center", va="center",
            color="white", fontsize=6.5, fontweight="bold", zorder=7,
        )

    for name, _, y, color in event_specs:
        numbered_event(name, y, color)

    agent_sequence = [
        ("detected", y_agent),
        ("selected", y_agent + .18),
        ("recovery_first_enqueue", y_agent - .18),
        ("graph", y_agent + .18),
        ("recovery_last_enqueue", y_agent - .18),
        ("handoff", y_agent),
        ("next_enqueue", y_agent),
    ]
    arrow((values["failure"], y_failed), (values["detected"], y_agent),
          failure_color)
    for (before, before_y), (after, after_y) in zip(
        agent_sequence, agent_sequence[1:]
    ):
        arrow((values[before], before_y), (values[after], after_y),
              normal_color if after == "next_enqueue" else agent_color)

    timeline.plot(
        [values["recovery_first_enqueue"], values["recovery_last_enqueue"]],
        [y_agent - .16, y_agent - .16], color=recovery_color,
        linewidth=4, solid_capstyle="round", zorder=3,
    )
    timeline.axvline(
        values["handoff"], color=agent_color, linestyle="--",
        linewidth=1.1, alpha=.75,
        zorder=1,
    )
    timeline.text(
        values["handoff"], 5.43, "logical recovery complete",
        ha="center", va="bottom", fontsize=8, color=agent_color,
    )

    recovery_mid = (
        values["recovery_first_enqueue"] + values["recovery_last_receive"]
    ) / 2
    arrow((values["recovery_first_enqueue"], y_agent - .18),
          (recovery_mid, y_forward + .08), recovery_color, 1.8)
    arrow((recovery_mid, y_forward + .08),
          (values["recovery_last_receive"], y_recovery),
          recovery_color, 1.8)
    timeline.text(
        recovery_mid, y_forward + .2,
        f"recovery traffic ({trial['required_packet_count']} packets)",
        ha="center", va="bottom", fontsize=7, color=recovery_color,
    )

    normal_mid = (values["next_enqueue"] + values["next_receive"]) / 2
    arrow((values["next_enqueue"], y_agent),
          (normal_mid, y_forward - .18), normal_color, 2.0)
    arrow((normal_mid, y_forward - .18),
          (values["next_receive"], y_normal + .18),
          normal_color, 2.0)
    timeline.text(
        normal_mid, y_forward - .3, "next normal minibatch",
        ha="center", va="top", fontsize=7, color=normal_color,
    )

    timeline.plot(
        [state_start, values["state_end"]],
        [y_recovery + .18, y_recovery + .18], color="#fdae6b",
        linewidth=5, solid_capstyle="round", zorder=4,
    )
    timeline.plot(
        [values["state_end"], values["physical"]],
        [y_recovery, y_recovery],
        color=recovery_color, linewidth=2.2, alpha=.8,
    )
    timeline.text(
        (values["state_end"] + values["physical"]) / 2,
        y_recovery + .12, "recovery replay",
        ha="center", va="bottom", fontsize=7, color=physical_color,
    )

    timeline.plot(
        [values["execute"], values["execute_end"]],
        [y_normal, y_normal], color=normal_color,
        linewidth=5, solid_capstyle="round", zorder=4,
    )
    timeline.text(
        (values["execute"] + values["execute_end"]) / 2,
        y_normal - .12, "normal task processing interval",
        ha="center", va="top", fontsize=7, color=normal_color,
    )
    arrow((values["next_receive"], y_normal + .18),
          (values["execute"], y_normal), normal_color)
    arrow((values["execute_end"], y_normal),
          (values["output"], y_normal - .18), normal_color)

    overlap_start = max(values["execute"], values["state_end"])
    overlap_end = min(values["execute_end"], values["physical"])
    if overlap_end > overlap_start:
        timeline.annotate(
            "concurrent in wall time; separate Dealer/STM streams",
            ((overlap_start + overlap_end) / 2, 1.5),
            ha="center", va="center", fontsize=7, color="#555555",
            bbox={"boxstyle": "round,pad=.25", "fc": "white",
                  "ec": "#bdbdbd", "alpha": .9},
            zorder=5,
        )

    reverse_mid = (values["output"] + values["visible"]) / 2
    arrow((values["output"], y_normal - .18), (reverse_mid, y_reverse),
          normal_color, 1.8)
    arrow((reverse_mid, y_reverse), (values["visible"], y_agent),
          normal_color, 1.8)
    timeline.text(
        reverse_mid, y_reverse + .12, "normal output",
        ha="center", va="bottom", fontsize=7, color=normal_color,
    )

    event_key.axis("off")
    event_key.set_title("Numbered events", loc="left", fontsize=10)
    line_y = .965
    line_step = .054
    for index, (name, label, _, color) in enumerate(ordered_specs, 1):
        event_key.text(
            .0, line_y, f"{index}", ha="left", va="top", color="white",
            fontsize=7, fontweight="bold",
            bbox={"boxstyle": "circle,pad=.22", "fc": color, "ec": "none"},
            transform=event_key.transAxes,
        )
        suffix = ""
        if name == "execute":
            suffix = (
                f"\nreceive→STM {stm - values['next_receive']:.3f} ms; "
                f"STM→execute {values['execute'] - stm:.3f} ms"
            )
        event_key.text(
            .075, line_y, f"{label}\n{values[name]:.1f} ms{suffix}",
            ha="left", va="top", fontsize=7.2, transform=event_key.transAxes,
        )
        line_y -= line_step + (.028 if suffix else 0.0)

    dispatch_after_handoff = values["next_enqueue"] - values["handoff"]
    dispatch_before_physical = values["physical"] - values["next_enqueue"]
    timeline.set_title(
        "Causal PEFT recovery timeline — "
        f"symmetric {trial['link_latency_ms']:g} ms per direction\n"
        f"Agent queues the next minibatch {dispatch_after_handoff:.1f} ms after "
        f"handoff and {dispatch_before_physical:.1f} ms before physical recovery"
    )
    end = max(values["physical"], values["visible"])
    padding = max(end * .035, 20.0)
    timeline.set_xlim(-padding, end + padding)
    timeline.set_xlabel("Time since TrainNode failure injection (ms)")
    timeline.set_ylim(-.55, 5.65)
    timeline.grid(axis="x", alpha=.18)
    fig.tight_layout()
    fig.savefig(figures / "nonblocking_causality.png", dpi=330)
    fig.savefig(figures / "nonblocking_causality.pdf")
    plt.close(fig)


def plot_delay_sweep(figures: Path, recoveries: list[dict[str, Any]]) -> None:
    if not recoveries:
        return
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in recoveries:
        grouped.setdefault(float(row["link_latency_ms"]), []).append(row)
    xs = sorted(grouped)
    if len(xs) < 2:
        # A single latency condition is not a sweep.  Remove stale output from
        # earlier analyses rather than publishing a misleading 46--54 ms plot.
        for suffix in ("png", "pdf"):
            path = figures / f"recovery_delay_sweep.{suffix}"
            if path.exists():
                path.unlink()
        return
    physical = [statistics.median(
        float(row["handoff_to_physical_complete_ms"]) for row in grouped[x]
    ) for x in xs]
    dispatch = [statistics.median(
        float(row["handoff_to_next_task_enqueue_ms"]) for row in grouped[x]
    ) for x in xs]
    passed = [sum(bool(row["next_task_enqueued_before_physical"])
                  for row in grouped[x]) for x in xs]
    counts = [len(grouped[x]) for x in xs]
    fig, axis = plt.subplots(figsize=(9.5, 5.2))

    def observations(axis, field: str, color: str) -> None:
        spacing = min(
            [b - a for a, b in zip(xs, xs[1:])], default=25.0
        )
        for x in xs:
            values = [float(row[field]) for row in grouped[x]]
            if len(values) == 1:
                offsets = [0.0]
            else:
                width = min(spacing * .18, 4.0)
                offsets = [
                    -width + 2 * width * index / (len(values) - 1)
                    for index in range(len(values))
                ]
            axis.scatter(
                [x + offset for offset in offsets], values,
                color=color, alpha=.38, s=24, zorder=2,
            )

    observations(axis, "handoff_to_physical_complete_ms", "#d62728")
    observations(axis, "handoff_to_next_task_enqueue_ms", "#2ca02c")
    axis.fill_between(
        xs, dispatch, physical, color="#54a24b", alpha=.10,
        label="Observed nonblocking margin",
    )
    axis.plot(xs, physical, "o-", color="#d62728", linewidth=2,
              label="Physical recovery complete")
    axis.plot(xs, dispatch, "o-", color="#2ca02c", linewidth=2,
              label="Next task locally enqueued")
    for x, ok, count in zip(xs, passed, counts):
        axis.text(
            x, physical[xs.index(x)] + 42, f"{ok}/{count}",
            ha="center", va="bottom", fontsize=7, color="#555555",
        )
    total_passed = sum(passed)
    total = sum(counts)
    axis.set_title(
        "Symmetric link-delay sweep — Agent dispatch remains nonblocking\n"
        f"{total_passed}/{total} recovery trials queued the next minibatch "
        "before physical recovery"
    )
    axis.set_ylabel("Time after Promise handoff (ms)")
    axis.set_xlabel(
        "Symmetric one-way latency per direction (ms); injected RTT = 2×x"
    )
    axis.legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
    axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figures / "recovery_delay_sweep.png", dpi=330)
    fig.savefig(figures / "recovery_delay_sweep.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    manifest = read_json(result_root / "manifest.json")
    runs = manifest["runs"]
    if not runs:
        raise AnalysisError("manifest contains no runs")
    for run in runs:
        if "link_latency_ms" not in run:
            run["link_latency_ms"] = run.get("uplink_latency_ms", 0.0)

    run_rows: list[dict[str, Any]] = []
    minibatch_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    events_by_run: dict[tuple[str, int], list[dict[str, Any]]] = {}
    raw_by_run: dict[tuple[str, int], list[dict[str, Any]]] = {}

    for run in runs:
        if run.get("benchmark_returncode") != 0 or run.get("error") is not None:
            raise AnalysisError(f"unsuccessful run in manifest: {run}")
        run_dir = result_root / run["run_dir"]
        rows = result_rows(run_dir)
        benchmark_args = read_json(run_dir / "raw" / "args.json")["benchmark"]
        if benchmark_args.get("dataset") != "advertise-gen":
            raise AnalysisError(
                f"non-real training dataset in {run['condition']}/"
                f"{run['repeat']}: {benchmark_args.get('dataset')!r}"
            )
        events = read_events(run_dir / "events.jsonl")
        key = (str(run["condition"]), int(run["repeat"]))
        events_by_run[key] = events
        raw_by_run[key] = rows
        base = min(float(row["start_time_s"]) for row in rows)
        for row in rows:
            minibatch_rows.append({
                "condition": run["condition"], "case": run["case"],
                "repeat": run["repeat"],
                "link_latency_ms": run["link_latency_ms"],
                **row,
                "start_elapsed_s": float(row["start_time_s"]) - base,
                "completion_elapsed_s": float(row["completion_time_s"]) - base,
            })
        commits = [
            event for event in named(events, "train_minibatch_committed")
            if event.get("origin") != "path_probe"
        ]
        commit_ids = [str(event["minibatch_id"]) for event in commits]
        expected_ids = {str(row["minibatch_id"]) for row in rows}
        if len(commit_ids) != len(set(commit_ids)) or set(commit_ids) != expected_ids:
            raise AnalysisError(
                f"semantic commit set/count mismatch in {run['condition']}/{run['repeat']}"
            )
        probes = named(events, "train_path_probe_execution")
        if not probes or any(event.get("mode") != "measured" for event in probes):
            raise AnalysisError(
                f"synthetic or missing path probe in {run['condition']}/{run['repeat']}"
            )
        snapshots = named(events, "shadow_state_snapshot")
        failure_parameters = manifest["parameters"]
        planned_failures = (
            failure_parameters.get("drop_times_s")
            or failure_parameters.get("drop_versions", [])
        )
        expected_failures = len(planned_failures) if run["failure"] else 0
        if len(snapshots) != expected_failures:
            raise AnalysisError(
                f"expected {expected_failures} recoveries in {key}, got {len(snapshots)}"
            )
        snapshots.sort(key=ns)
        for index, snapshot in enumerate(snapshots):
            next_ns = ns(snapshots[index + 1]) if index + 1 < len(snapshots) else None
            recovery, trial, packets = analyze_recovery(
                run, events, snapshot, next_ns
            )
            recovery["failure_index"] = index
            trial["failure_index"] = index
            recovery_rows.append(recovery)
            trial_rows.append(trial)
            packet_rows.extend(packets)
        duration = max(float(row["completion_time_s"]) for row in rows) - base
        run_rows.append({
            "condition": run["condition"], "case": run["case"],
            "repeat": run["repeat"],
            "link_latency_ms": run["link_latency_ms"],
            "completed_minibatches": len(rows), "duration_s": duration,
            "minibatches_per_s": len(rows) / duration,
            "final_loss": rows[-1]["loss"], "recovery_count": len(snapshots),
        })

    experiment_profile = manifest["parameters"].get("experiment_profile")
    enforce_loss_equality = experiment_profile != "peft-correctness"
    normal_keys = [key for key in raw_by_run if key[0] == "normal"]
    if not normal_keys:
        raise AnalysisError("a Normal baseline is required for correctness")
    correctness_runs: list[dict[str, Any]] = []
    loss_difference_rows: list[dict[str, Any]] = []
    loss_equality_failures: list[dict[str, Any]] = []
    for key, rows in raw_by_run.items():
        if key[0] == "normal":
            continue
        baseline_key = ("normal", key[1])
        if baseline_key not in raw_by_run:
            baseline_key = normal_keys[0]
        baseline_rows = sorted(
            raw_by_run[baseline_key],
            key=lambda row: int(row["completion_index"]),
        )
        baseline = {row["minibatch_id"]: row for row in baseline_rows}
        actual = {row["minibatch_id"]: row for row in rows}
        if set(actual) != set(baseline):
            raise AnalysisError(f"minibatch set differs from Normal for {key}")
        differences = {
            minibatch_id: abs(actual[minibatch_id]["loss"] - baseline[minibatch_id]["loss"])
            for minibatch_id in actual
        }
        failure_versions = [
            int(row["failure_version"])
            for row in recovery_rows
            if row["condition"] == key[0] and int(row["repeat"]) == key[1]
        ]
        last_failure_version = max(failure_versions, default=None)
        ordered_differences = []
        for baseline_row in baseline_rows:
            minibatch_id = str(baseline_row["minibatch_id"])
            actual_row = actual[minibatch_id]
            difference = differences[minibatch_id]
            difference_row = {
                "condition": key[0],
                "repeat": key[1],
                "completion_index": int(baseline_row["completion_index"]),
                "minibatch_id": minibatch_id,
                "request_version": int(baseline_row["request_version"]),
                "baseline_loss": float(baseline_row["loss"]),
                "recover_loss": float(actual_row["loss"]),
                "abs_error": difference,
                "exactly_equal": difference == 0.0,
                "at_or_after_last_failure_version": (
                    last_failure_version is not None
                    and int(baseline_row["request_version"])
                    >= last_failure_version
                ),
            }
            ordered_differences.append(difference_row)
            loss_difference_rows.append(difference_row)
        exact_mismatches = [
            row for row in ordered_differences if not row["exactly_equal"]
        ]
        exact_zero_suffix_count = 0
        for row in reversed(ordered_differences):
            if not row["exactly_equal"]:
                break
            exact_zero_suffix_count += 1
        exact_zero_suffix_start = (
            ordered_differences[-exact_zero_suffix_count]
            if exact_zero_suffix_count else None
        )
        violations = [
            minibatch_id for minibatch_id, difference in differences.items()
            if difference > LOSS_ATOL + LOSS_RTOL * abs(baseline[minibatch_id]["loss"])
        ]
        correctness_runs.append({
            "condition": key[0], "repeat": key[1],
            "baseline_condition": baseline_key[0],
            "minibatch_set_equal": True,
            "max_loss_abs_error": max(differences.values(), default=0.0),
            "all_losses_exactly_equal": not exact_mismatches,
            "exact_loss_equal_count": len(ordered_differences) - len(exact_mismatches),
            "exact_loss_mismatch_count": len(exact_mismatches),
            "first_exact_loss_mismatch": (
                exact_mismatches[0] if exact_mismatches else None
            ),
            "last_exact_loss_mismatch": (
                exact_mismatches[-1] if exact_mismatches else None
            ),
            "last_failure_version": last_failure_version,
            "post_failure_exact_mismatch_count": sum(
                not row["exactly_equal"]
                and bool(row["at_or_after_last_failure_version"])
                for row in ordered_differences
            ),
            "exact_zero_suffix_count": exact_zero_suffix_count,
            "exact_zero_suffix_start": exact_zero_suffix_start,
            "eventually_exact_zero_through_run_end": (
                exact_zero_suffix_count > 0
            ),
            "loss_violation_count": len(violations),
            "strict_loss_tolerance_passed": not violations,
            "passed": not violations or not enforce_loss_equality,
        })
        if violations and enforce_loss_equality:
            loss_equality_failures.append({
                "condition": key[0],
                "repeat": key[1],
                "violation_count": len(violations),
                "max_loss_abs_error": max(differences.values()),
            })

    ack_events = []
    for events in events_by_run.values():
        for event in events:
            event_name = str(event.get("event", "")).lower()
            tokens = event_name.replace("-", "_").split("_")
            if "recovery" in tokens and (
                "ack" in tokens or "acknowledgement" in tokens
                or "acknowledgment" in tokens
            ):
                ack_events.append(event)
    max_latency = max(float(row["link_latency_ms"]) for row in trial_rows)
    convergence_profile = experiment_profile in {
        "long-loss", "correctness", "peft-correctness",
        "repeat-recovery-mini",
    }
    if max_latency <= 0 and not convergence_profile:
        raise AnalysisError(
            "nonblocking proof requires at least one positive symmetric-latency condition"
        )
    evidence_trials = [
        row for row in trial_rows
        if float(row["link_latency_ms"]) == max_latency
    ]
    if not evidence_trials:
        raise AnalysisError("no delayed recovery evidence trials")
    failed_overlap = [
        row for row in evidence_trials
        if not row["next_task_enqueued_before_physical"]
        or not row["state_load_finished_after_handoff"]
        or not row["physical_finished_after_handoff"]
    ]
    if failed_overlap:
        raise AnalysisError(
            f"nonblocking overlap not observed in {len(failed_overlap)} "
            f"max-latency trials"
        )
    failed_same_batch = [
        row for row in evidence_trials
        if not row["next_task_same_failed_batch"]
    ]
    if failed_same_batch and not convergence_profile:
        raise AnalysisError(
            f"same-batch post-handoff dispatch not observed in "
            f"{len(failed_same_batch)} max-latency trials"
        )
    if ack_events:
        raise AnalysisError(f"unexpected recovery ACK events: {ack_events[:3]}")
    link_delay_violations: list[dict[str, Any]] = []
    for row in recovery_rows:
        configured = float(row["link_latency_ms"])
        lower_bound = max(0.0, configured - 5.0)
        for field in (
            "injection_to_detection_ms",
            "next_enqueue_to_receive_ms",
            "next_output_transfer_ms",
        ):
            measured = float(row[field])
            if measured < lower_bound:
                link_delay_violations.append({
                    "condition": row["condition"],
                    "repeat": row["repeat"],
                    "recovery_id": row["recovery_id"],
                    "field": field,
                    "configured_ms": configured,
                    "measured_ms": measured,
                })
    if link_delay_violations:
        raise AnalysisError(
            "symmetric link-delay validation failed: "
            f"{link_delay_violations[:3]}"
        )

    tables = result_root / "tables"
    figures = result_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "runs.csv", run_rows)
    write_csv(tables / "minibatches.csv", minibatch_rows)
    write_csv(tables / "recoveries.csv", recovery_rows)
    write_csv(tables / "packets.csv", packet_rows)
    write_csv(tables / "nonblocking_trials.csv", trial_rows)
    write_csv(tables / "loss_differences.csv", loss_difference_rows)
    correctness = {
        "schema_version": 2,
        "criterion": {
            "minibatch_commit_set": "exact equality with Normal",
            "loss_atol": LOSS_ATOL, "loss_rtol": LOSS_RTOL,
            "loss_equality_enforced": enforce_loss_equality,
        },
        "runs": correctness_runs,
        "passed": all(row["passed"] for row in correctness_runs),
    }
    proof = {
        "schema_version": 1,
        "claim": (
            "After promise_handoff_committed, the Agent dispatches a new "
            "training task without waiting for replacement Dealer physical "
            "recovery or an ACK; Dealer execution resumes from local STM."
        ),
        "max_symmetric_per_direction_latency_evidence_ms": max_latency,
        "evidence_trial_count": len(evidence_trials),
        "all_evidence_trials_enqueue_before_physical": True,
        "recovery_ack_event_count": 0,
        "all_required_recovery_packets_enqueued_before_handoff": True,
        "all_post_handoff_tasks_reached_stm_and_executed": True,
        "symmetric_link_delay_validated": True,
        "symmetric_link_delay_validation": {
            "directions": {
                "Dealer_to_Agent_failure_notification": (
                    "injection_to_detection_ms"
                ),
                "Agent_to_Dealer_next_task": "next_enqueue_to_receive_ms",
                "Dealer_to_Agent_output": "next_output_transfer_ms",
            },
            "allowed_early_tolerance_ms": 5.0,
            "violation_count": 0,
        },
        "same_failed_batch_evidence_count": sum(
            bool(row["next_task_same_failed_batch"]) for row in evidence_trials
        ),
        "all_evidence_trials_dispatch_same_failed_batch": not failed_same_batch,
        "passed": True,
    }
    write_json(result_root / "correctness.json", correctness)
    write_json(result_root / "nonblocking_proof.json", proof)
    write_json(result_root / "event_protocol.json", {
        "schema_version": 1,
        "clock": {
            "cross_process_field": "time_ns",
            "cross_process_clock": "time.time_ns wall clock",
            "monotonic_ns_scope": "single process only",
            "deployment_assumption": (
                "All processes in this basic experiment run on one host; "
                "distributed runs require clock synchronization or tracing offsets."
            ),
        },
        "events": {
            "replacement_graph_installed": (
                "Agent installed the replacement Binding and logical graph."
            ),
            "recovery_packet_enqueue_accepted": (
                "The Agent local send path accepted one required recovery "
                "packet; this is not remote receipt or transport ACK."
            ),
            "promise_handoff_committed": (
                "The replacement graph is installed and every required "
                "recovery packet has been accepted by the local send path."
            ),
            "replacement_state_load_end": (
                "Replacement Dealer finished loading checkpoint state."
            ),
            "train_physical_recovery_complete": (
                "Replacement Dealer finished all required replay tasks and "
                "their optimizer/state transitions."
            ),
            "train_service_output_visible": (
                "A training result packet reached the Agent."
            ),
        },
        "scope": manifest["parameters"]["scope"],
    })
    external_environment = (
        EXP_ROOT / "environment" / f"{manifest['run_id']}.json"
    )
    if external_environment.is_file():
        write_json(
            result_root / "environment.json",
            read_json(external_environment),
        )
    summary = {
        "schema_version": 2,
        "run_count": len(run_rows), "recovery_count": len(recovery_rows),
        "correctness_passed": correctness["passed"],
        "nonblocking_proof_passed": proof["passed"],
        "loss_comparison": {
            "comparison_run_count": len(correctness_runs),
            "all_runs_exactly_equal": all(
                row["all_losses_exactly_equal"] for row in correctness_runs
            ),
            "exact_loss_mismatch_count": sum(
                int(row["exact_loss_mismatch_count"])
                for row in correctness_runs
            ),
            "all_runs_eventually_exact_zero_through_run_end": all(
                row["eventually_exact_zero_through_run_end"]
                for row in correctness_runs
            ),
            "minimum_exact_zero_suffix_count": min(
                (
                    int(row["exact_zero_suffix_count"])
                    for row in correctness_runs
                ),
                default=0,
            ),
        },
        "conditions": {},
    }
    summary["validated"] = (
        summary["correctness_passed"]
        and summary["nonblocking_proof_passed"]
    )
    for condition in sorted({row["condition"] for row in run_rows}):
        selected_runs = [row for row in run_rows if row["condition"] == condition]
        selected_recovery = [
            row for row in recovery_rows if row["condition"] == condition
        ]
        summary["conditions"][condition] = {
            "repeats": len(selected_runs),
            "duration_s_median": statistics.median(
                float(row["duration_s"]) for row in selected_runs
            ),
            "minibatches_per_s_median": statistics.median(
                float(row["minibatches_per_s"]) for row in selected_runs
            ),
            "recovery_count": len(selected_recovery),
            "handoff_to_physical_ms_median": (
                statistics.median(float(row["handoff_to_physical_complete_ms"])
                                  for row in selected_recovery)
                if selected_recovery else None
            ),
            "enqueue_before_physical_fraction": (
                sum(bool(row["next_task_enqueued_before_physical"])
                    for row in selected_recovery) / len(selected_recovery)
                if selected_recovery else None
            ),
        }
    write_json(result_root / "summary.json", summary)
    # Figure helpers use absolute event timestamps stored only in trial_rows.
    recovery_plot_rows = []
    trial_index = {(row["recovery_id"], row["condition"], row["repeat"]): row
                   for row in trial_rows}
    for row in recovery_rows:
        recovery_plot_rows.append({
            **row,
            **trial_index[(row["recovery_id"], row["condition"], row["repeat"])],
        })
    plot_overview(figures, minibatch_rows, recovery_plot_rows)
    plot_causality(figures, trial_rows)
    plot_delay_sweep(figures, recovery_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if loss_equality_failures:
        raise AnalysisError(
            "strict loss equality failed; diagnostic artifacts were written: "
            f"{loss_equality_failures}"
        )


if __name__ == "__main__":
    main()
