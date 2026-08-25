#!/usr/bin/env python3
"""Validate and analyze the deterministic basic recovery experiment."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt


CASE_ORDER = (
    "normal-reprefill",
    "recover-reprefill",
    "recover-split-reprefill",
    "normal-mix",
    "recover-mix",
    "recover-split-mix",
)
CASE_STYLE = {
    "normal-reprefill": ("Normal / Re-Prefill", "#4c78a8", "-"),
    "recover-reprefill": ("Recover / Re-Prefill", "#f58518", "-"),
    "recover-split-reprefill": ("Recover+Split / Re-Prefill", "#e45756", "-"),
    "normal-mix": ("Normal / Mix", "#4c78a8", "--"),
    "recover-mix": ("Recover / Mix", "#f58518", "--"),
    "recover-split-mix": ("Recover+Split / Mix", "#e45756", "--"),
}
RECOVERY_FIELDS = (
    "failure_interval_ms",
    "failure_stall_excess_ms",
    "injection_to_detection_ms",
    "detection_to_lookup_start_ms",
    "lookup_ms",
    "path_probe_ms",
    "lookup_end_to_graph_installed_ms",
    "graph_installed_to_handoff_commit_ms",
    "handoff_commit_to_first_normal_output_ms",
    "lookup_end_to_rebuild_end_ms",
    "rebuild_end_to_replay_send_end_ms",
    "replay_send_to_first_normal_output_ms",
    "detection_to_first_normal_output_ms",
    "injection_to_first_normal_output_ms",
    "replay_task_count",
    "replay_input_count",
    "shadow_total_retained_bytes",
    "shadow_kv_bytes",
    "shadow_update_cpu_ns",
    "checkpoint_block_count",
    "shadow_retained_replay_task_count",
)


class AnalysisError(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError as error:
            raise AnalysisError(
                f"invalid JSONL at {path}:{line_number}: {error}"
            ) from error
    return sorted(records, key=lambda record: int(record["time_ns"]))


def events_named(
    events: Iterable[dict[str, Any]], name: str
) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event") == name]


def exactly_one(
    events: Iterable[dict[str, Any]], name: str
) -> dict[str, Any]:
    matches = events_named(events, name)
    if len(matches) != 1:
        raise AnalysisError(
            f"expected exactly one {name!r} event, found {len(matches)}"
        )
    return matches[0]


def zero_or_one(
    events: Iterable[dict[str, Any]], name: str
) -> Optional[dict[str, Any]]:
    matches = events_named(events, name)
    if len(matches) > 1:
        raise AnalysisError(
            f"expected at most one {name!r} event, found {len(matches)}"
        )
    return matches[0] if matches else None


def event_at(time_ns: int, event: str) -> dict[str, Any]:
    """Construct an event-shaped timestamp for legacy derived boundaries."""
    return {"event": event, "time_ns": int(time_ns)}


def resolve_handoff_timing(
    recovery_events: list[dict[str, Any]],
    sends: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resolve graph installation and handoff commit for old and new traces.

    New traces contain an independently observed handoff event plus per-send
    return events.  Legacy traces contain neither, so ``replay_send_end`` is
    retained only as the closest available proxy for local send-path
    acceptance.  The k=0 definition reduces to graph installation (or the
    later replacement-create send return, if a future implementation reorders
    those operations).
    """
    graph_installed = zero_or_one(
        recovery_events, "replacement_graph_installed"
    )
    graph_event_name = "replacement_graph_installed"
    if graph_installed is None:
        graph_installed = zero_or_one(recovery_events, "graph_update_end")
        graph_event_name = "graph_update_end_legacy_proxy"
    if graph_installed is None:
        graph_installed = exactly_one(recovery_events, "promise_graph_commit")
        graph_event_name = "promise_graph_commit_legacy_proxy"

    replay_send_end = zero_or_one(recovery_events, "replay_send_end")
    send_returns = sorted(
        events_named(recovery_events, "recovery_packet_send_return"),
        key=lambda event: int(event["task_index"]),
    )
    create_send_return = zero_or_one(
        recovery_events, "replacement_create_send_return"
    )

    send_ids = {str(event["packet_id"]) for event in sends}
    return_ids = {str(event["packet_id"]) for event in send_returns}
    if send_returns and (
        len(send_returns) != len(sends) or return_ids != send_ids
    ):
        raise AnalysisError(
            "recovery send-return events do not match send-start packet IDs: "
            f"starts={send_ids}, returns={return_ids}"
        )
    if sends and replay_send_end is None:
        raise AnalysisError("recovery packets exist but replay_send_end is missing")

    acceptance_events: list[dict[str, Any]] = []
    if create_send_return is not None:
        acceptance_events.append(create_send_return)
    if send_returns:
        acceptance_events.extend(send_returns)
        last_enqueue_source = "recovery_packet_send_return"
    elif sends:
        assert replay_send_end is not None
        acceptance_events.append(replay_send_end)
        last_enqueue_source = "replay_send_end_legacy_proxy"
    elif create_send_return is not None:
        last_enqueue_source = "replacement_create_send_return"
    else:
        last_enqueue_source = "none_k0_legacy"

    graph_time_ns = int(graph_installed["time_ns"])
    last_enqueue_time_ns = (
        max(int(event["time_ns"]) for event in acceptance_events)
        if acceptance_events else graph_time_ns
    )
    requirement_time_ns = max(graph_time_ns, last_enqueue_time_ns)
    observed_commit = zero_or_one(
        recovery_events, "promise_handoff_committed"
    )
    handoff_commit = event_at(
        requirement_time_ns, "defined_promise_handoff_commit"
    )
    handoff_source = (
        "max_of_observed_graph_and_send_returns"
        if observed_commit is not None and send_returns
        else "derived_legacy_proxy"
    )

    return {
        "graph_installed": graph_installed,
        "graph_event_source": graph_event_name,
        "replay_send_end": replay_send_end,
        "send_returns": send_returns,
        "create_send_return": create_send_return,
        "last_enqueue_time_ns": last_enqueue_time_ns,
        "last_enqueue_source": last_enqueue_source,
        "requirement_time_ns": requirement_time_ns,
        "handoff_commit": handoff_commit,
        "handoff_commit_event": observed_commit,
        "handoff_source": handoff_source,
        "handoff_observed": observed_commit is not None,
    }


def milliseconds(start: dict[str, Any], end: dict[str, Any]) -> float:
    return (int(end["time_ns"]) - int(start["time_ns"])) / 1e6


def token_hash(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def median_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    return statistics.mean(present) if present else None


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(fraction * len(ordered)) - 1)])


def normalized_replacement(value: Any) -> list[list[Any]]:
    replacement = [list(edge) for edge in value]
    replacement.sort(key=lambda edge: (str(edge[0]), int(edge[1]), int(edge[2])))
    return replacement


def expected_initial(topology: str, keypoints: Any) -> str:
    if (
        not isinstance(keypoints, list)
        or len(keypoints) != 1
        or len(keypoints[0]) != 3
        or keypoints[0][1:] != [0, 6]
    ):
        raise AnalysisError(
            f"expected one complete head+8/8/8/8+tail provider, found {keypoints!r}"
        )
    dealer = str(keypoints[0][0])
    allowed = (
        {"T0-ordinary-a", "T0-ordinary-b", "T0-ordinary-c"}
        if topology == "ordinary"
        else {"T0-split-active"}
    )
    if dealer not in allowed:
        raise AnalysisError(
            f"unexpected initial dealer {dealer!r} for {topology} topology"
        )
    return dealer


def expected_failure_path(
    topology: str,
    initial_dealer: str,
    recovery_index: int,
) -> tuple[str, list[list[Any]]]:
    if topology == "ordinary":
        raise AnalysisError(
            "ordinary replacement depends on the preceding measured choice"
        )
    if topology == "split":
        expected = (
            (
                "T0-split-active",
                [["T1-split-left", 0, 3], ["T1-split-right", 3, 3]],
            ),
            ("T1-split-left", [["T0-split-active", 0, 3]]),
            ("T0-split-active", [["T1-split-left", 0, 3]]),
        )
        return expected[recovery_index]
    raise AnalysisError(f"unknown topology: {topology}")


def build_nonblocking_row(
    *,
    case_name: str,
    repeat: int,
    recovery_index: int,
    recovery_id: str,
    recovery_events: list[dict[str, Any]],
    configured_agent_to_dealer_latency_ms: float,
    configured_dealer_to_agent_latency_ms: float,
    injected: dict[str, Any],
    detected: dict[str, Any],
) -> dict[str, Any]:
    """Validate one complete Agent→Dealer non-blocking causal chain."""
    lookup_start = exactly_one(recovery_events, "replacement_lookup_start")
    lookup_end = exactly_one(recovery_events, "replacement_lookup_end")
    handler_return = exactly_one(recovery_events, "recovery_handler_return")
    create_send = exactly_one(recovery_events, "replacement_create_send")
    create_receive = exactly_one(
        recovery_events, "replacement_create_receive"
    )
    state_start = exactly_one(
        recovery_events, "replacement_state_load_start"
    )
    state_end = exactly_one(recovery_events, "replacement_state_load_end")
    first_complete = exactly_one(
        recovery_events, "first_recovery_task_complete"
    )
    frontier = exactly_one(recovery_events, "frontier_materialized")
    sends = sorted(
        events_named(recovery_events, "recovery_packet_send"),
        key=lambda event: int(event["task_index"]),
    )
    receives = events_named(recovery_events, "recovery_packet_receive")
    executes_start = events_named(
        recovery_events, "recovery_task_execute_start"
    )
    executes_end = events_named(recovery_events, "recovery_task_execute_end")
    output_sends = events_named(recovery_events, "recovery_output_send")
    output_arrivals = events_named(recovery_events, "replay_output_observed")
    if not sends:
        raise AnalysisError(
            f"recovery {recovery_id} has no recovery packets; the logical "
            "handoff resolver supports k=0, but this Dealer execution proof "
            "requires at least one replay/frontier packet"
        )
    expected_count = len(sends)
    handoff = resolve_handoff_timing(recovery_events, sends)
    graph_installed = handoff["graph_installed"]
    handoff_commit = handoff["handoff_commit"]
    handoff_commit_event = (
        handoff["handoff_commit_event"] or handoff_commit
    )
    replay_send_end = handoff["replay_send_end"]
    assert replay_send_end is not None
    last_enqueue = event_at(
        int(handoff["last_enqueue_time_ns"]),
        str(handoff["last_enqueue_source"]),
    )
    counted = {
        "receive": len(receives),
        "execute_start": len(executes_start),
        "execute_end": len(executes_end),
        "output_send": len(output_sends),
    }
    if any(count != expected_count for count in counted.values()):
        raise AnalysisError(
            f"incomplete packet causal chain for {recovery_id}: "
            f"sent={expected_count}, observed={counted}"
        )
    send_by_packet = {str(event["packet_id"]): event for event in sends}
    receive_by_packet = {
        str(event["packet_id"]): event for event in receives
    }
    observed_packet_sets = {
        "receive": set(receive_by_packet),
        "execute_start": {
            str(event["packet_id"]) for event in executes_start
        },
        "execute_end": {
            str(event["packet_id"]) for event in executes_end
        },
        "output_send": {
            str(event["packet_id"]) for event in output_sends
        },
    }
    if (
        len(send_by_packet) != expected_count
        or any(
            packet_ids != set(send_by_packet)
            for packet_ids in observed_packet_sets.values()
        )
    ):
        raise AnalysisError(
            f"packet IDs differ across causal stages for {recovery_id}: "
            f"sent={set(send_by_packet)}, observed={observed_packet_sets}"
        )
    indices = [int(event["task_index"]) for event in sends]
    if indices != list(range(expected_count)):
        raise AnalysisError(
            f"non-contiguous recovery task indices for {recovery_id}: {indices}"
        )
    if any(int(event["task_count"]) != expected_count for event in sends):
        raise AnalysisError(f"inconsistent recovery task count for {recovery_id}")

    last_send = max(sends, key=lambda event: int(event["time_ns"]))
    first_execute = min(
        executes_start, key=lambda event: int(event["time_ns"])
    )
    frontier_output_send = exactly_one(
        [event for event in output_sends if event.get("role") == "frontier_reinject"],
        "recovery_output_send",
    )
    frontier_output_arrival = exactly_one(
        [event for event in output_arrivals if event.get("role") == "frontier_reinject"],
        "replay_output_observed",
    )
    if (
        frontier_output_arrival.get("packet_id")
        != frontier_output_send.get("packet_id")
    ):
        raise AnalysisError(
            f"frontier output packet ID changed before Agent arrival for "
            f"{recovery_id}"
        )
    receive_latencies_ms = [
        milliseconds(send_by_packet[packet_id], receive_by_packet[packet_id])
        for packet_id in sorted(send_by_packet)
    ]
    output_return_latency_ms = milliseconds(
        frontier_output_send, frontier_output_arrival
    )
    assertions = {
        "replacement_graph_installed_before_dispatch_start": (
            int(graph_installed["time_ns"])
            <= min(int(event["time_ns"]) for event in sends)
        ),
        "handoff_commit_after_graph_install": (
            int(handoff_commit["time_ns"])
            >= int(graph_installed["time_ns"])
        ),
        "handoff_commit_after_last_enqueue_proxy": (
            int(handoff_commit["time_ns"]) >= int(last_enqueue["time_ns"])
        ),
        "all_recovery_send_calls_return_before_first_completion": (
            int(last_enqueue["time_ns"]) < int(first_complete["time_ns"])
        ),
        "handoff_commit_before_first_recovery_completion": (
            int(handoff_commit["time_ns"]) < int(first_complete["time_ns"])
        ),
        "handler_returns_before_frontier_materialized": (
            int(handler_return["time_ns"]) < int(frontier["time_ns"])
        ),
        "handler_returns_after_handoff_commit_boundary": (
            int(handler_return["time_ns"])
            >= int(handoff_commit["time_ns"])
        ),
        "dealer_executes_after_state_load": (
            int(first_execute["time_ns"]) >= int(state_end["time_ns"])
        ),
        "frontier_output_sent_without_agent_confirmation": (
            int(frontier_output_send["time_ns"]) >= int(frontier["time_ns"])
            and int(frontier_output_arrival["time_ns"])
            >= int(frontier_output_send["time_ns"])
        ),
        "packet_ids_match_end_to_end": True,
    }
    if handoff["create_send_return"] is not None:
        assertions["replacement_create_send_returned_before_graph_install"] = (
            int(handoff["create_send_return"]["time_ns"])
            <= int(graph_installed["time_ns"])
        )
    if handoff["handoff_commit_event"] is not None:
        assertions["observed_handoff_event_after_defined_boundary"] = (
            int(handoff_commit_event["time_ns"])
            >= int(handoff_commit["time_ns"])
        )
        assertions["handler_returns_after_observed_handoff_event"] = (
            int(handler_return["time_ns"])
            >= int(handoff_commit_event["time_ns"])
        )
    if configured_agent_to_dealer_latency_ms > 0:
        assertions["configured_agent_to_dealer_latency_observed"] = (
            min(receive_latencies_ms)
            >= configured_agent_to_dealer_latency_ms - 5.0
        )
    if configured_dealer_to_agent_latency_ms > 0:
        assertions["configured_dealer_to_agent_latency_observed"] = (
            output_return_latency_ms
            >= configured_dealer_to_agent_latency_ms - 5.0
        )
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise AnalysisError(
            f"non-blocking proof failed for {case_name}/{repeat}/"
            f"{recovery_index}: {failed}"
        )

    return {
        "case": case_name,
        "repeat": repeat,
        "recovery_index": recovery_index,
        "recovery_id": recovery_id,
        "configured_agent_to_dealer_latency_ms": (
            configured_agent_to_dealer_latency_ms
        ),
        "configured_dealer_to_agent_latency_ms": (
            configured_dealer_to_agent_latency_ms
        ),
        "recovery_task_count": expected_count,
        "packet_payload_retained_bytes": sum(
            int(event["payload_retained_bytes"]) for event in sends
        ),
        "dispatch_start_span_ms": milliseconds(sends[0], last_send),
        "graph_install_to_handoff_commit_ms": milliseconds(
            graph_installed, handoff_commit
        ),
        "last_enqueue_proxy_to_first_complete_ms": milliseconds(
            last_enqueue, first_complete
        ),
        "handoff_commit_to_first_complete_ms": milliseconds(
            handoff_commit, first_complete
        ),
        "handoff_commit_to_handler_return_ms": milliseconds(
            handoff_commit, handler_return
        ),
        "handoff_commit_event_to_first_complete_ms": (
            None
            if handoff["handoff_commit_event"] is None
            else milliseconds(handoff_commit_event, first_complete)
        ),
        "handoff_commit_event_to_handler_return_ms": (
            None
            if handoff["handoff_commit_event"] is None
            else milliseconds(handoff_commit_event, handler_return)
        ),
        "handoff_commit_event_lag_ms": (
            None
            if handoff["handoff_commit_event"] is None
            else milliseconds(handoff_commit, handoff_commit_event)
        ),
        "handler_return_to_frontier_ms": milliseconds(
            handler_return, frontier
        ),
        "state_load_ms": milliseconds(state_start, state_end),
        "state_load_to_first_execute_ms": milliseconds(
            state_end, first_execute
        ),
        "frontier_to_output_send_ms": milliseconds(
            frontier, frontier_output_send
        ),
        "output_send_to_agent_arrival_ms": output_return_latency_ms,
        "agent_to_dealer_packet_latency_ms_min": min(receive_latencies_ms),
        "agent_to_dealer_packet_latency_ms_median": statistics.median(
            receive_latencies_ms
        ),
        "all_assertions_pass": True,
        "assertions": assertions,
        "origin_time_ns": int(injected["time_ns"]),
        "failure_time_ns": int(injected["time_ns"]),
        "detection_time_ns": int(detected["time_ns"]),
        "lookup_start_time_ns": int(lookup_start["time_ns"]),
        "lookup_end_time_ns": int(lookup_end["time_ns"]),
        "replacement_graph_installed_time_ns": int(
            graph_installed["time_ns"]
        ),
        "replacement_graph_installed_source": str(
            handoff["graph_event_source"]
        ),
        "last_enqueue_proxy_time_ns": int(last_enqueue["time_ns"]),
        "last_enqueue_proxy_source": str(handoff["last_enqueue_source"]),
        "handoff_commit_time_ns": int(handoff_commit["time_ns"]),
        "handoff_commit_source": str(handoff["handoff_source"]),
        "handoff_commit_observed": bool(handoff["handoff_observed"]),
        "handoff_commit_event_time_ns": (
            None
            if handoff["handoff_commit_event"] is None
            else int(handoff["handoff_commit_event"]["time_ns"])
        ),
        "create_send_time_ns": int(create_send["time_ns"]),
        "create_send_return_time_ns": (
            None
            if handoff["create_send_return"] is None
            else int(handoff["create_send_return"]["time_ns"])
        ),
        "create_receive_time_ns": int(create_receive["time_ns"]),
        "state_start_time_ns": int(state_start["time_ns"]),
        "state_end_time_ns": int(state_end["time_ns"]),
        "first_send_time_ns": int(sends[0]["time_ns"]),
        "last_send_start_time_ns": int(last_send["time_ns"]),
        "replay_send_end_time_ns": int(replay_send_end["time_ns"]),
        "first_execute_time_ns": int(first_execute["time_ns"]),
        "first_complete_time_ns": int(first_complete["time_ns"]),
        "handler_return_time_ns": int(handler_return["time_ns"]),
        "frontier_time_ns": int(frontier["time_ns"]),
        "frontier_output_send_time_ns": int(frontier_output_send["time_ns"]),
        "frontier_output_arrival_time_ns": int(
            frontier_output_arrival["time_ns"]
        ),
        "packet_send_start_times_ns": [
            int(event["time_ns"]) for event in sends
        ],
        "packet_send_return_times_ns": [
            int(event["time_ns"]) for event in handoff["send_returns"]
        ],
        "packet_receive_times_ns": [
            int(receive_by_packet[str(event["packet_id"])]["time_ns"])
            for event in sends
        ],
        "execute_start_times_ns": sorted(
            int(event["time_ns"]) for event in executes_start
        ),
        "execute_end_times_ns": sorted(
            int(event["time_ns"]) for event in executes_end
        ),
    }


def validate_graph(raw_dir: Path, initial_dealer: str) -> tuple[bool, int, int]:
    decode_graph = read_json(raw_dir / "decode_graph.json")
    if initial_dealer not in decode_graph:
        raise AnalysisError(
            f"decode graph omits initial provider {initial_dealer!r} in {raw_dir}"
        )
    stats = decode_graph[initial_dealer]
    enabled = bool(stats.get("enabled"))
    replays = int(stats.get("replays", 0))
    fallbacks = len(dict(stats.get("fallbacks", {})))
    if not enabled or replays <= 0 or fallbacks:
        raise AnalysisError(
            f"CUDA graph validation failed in {raw_dir}: {stats!r}"
        )
    return enabled, replays, fallbacks


def analyze_run(
    result_root: Path,
    run: dict[str, Any],
    drop_steps: tuple[int, ...],
    agent_to_dealer_latency_ms: float,
    dealer_to_agent_latency_ms: float,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    case_name = str(run["case"])
    repeat = int(run["repeat"])
    topology = str(run["topology"])
    run_dir = result_root / Path(run["run_dir"])
    raw_dir = run_dir / "raw"
    if run.get("benchmark_returncode") != 0 or run.get("error") is not None:
        raise AnalysisError(
            f"manifest marks {case_name}/{repeat} unsuccessful: {run!r}"
        )

    result = read_json(raw_dir / "result.json")
    timestamps = [int(stamp) for sequence, stamp in result if int(sequence) == 0]
    if len(timestamps) < 2 or timestamps != sorted(timestamps):
        raise AnalysisError(f"invalid output timestamps in {raw_dir}")
    output_tokens = read_json(raw_dir / "tokens.json")
    if len(output_tokens) != 1:
        raise AnalysisError(f"expected one output token stream in {raw_dir}")
    tokens = [int(token) for token in output_tokens[0]]
    phases = read_json(raw_dir / "phases.json")
    expected_tokens = int(phases["parameters"]["max_tokens"])
    if (
        len(timestamps) != expected_tokens - 1
        or len(tokens) != expected_tokens
        or int(phases["parameters"]["num_paths"]) != 1
        or bool(phases["parameters"]["retrieve"]) != bool(run["retrieve"])
    ):
        raise AnalysisError(f"request shape/config validation failed in {raw_dir}")
    initial_dealer = expected_initial(topology, phases["keypoints"])
    graph_enabled, graph_replays, graph_fallbacks = validate_graph(
        raw_dir, initial_dealer
    )

    events = read_jsonl(run_dir / "runtime_metrics.jsonl")
    injected_events = events_named(events, "failure_injected")
    signal_events = events_named(events, "failure_signal_sent")
    detected_events = events_named(events, "failure_detected")
    shadow_events = events_named(events, "shadow_state_snapshot")
    lookup_start_events = events_named(events, "replacement_lookup_start")
    expected_failure_count = len(drop_steps) if bool(run["failure"]) else 0
    counted = {
        "failure_injected": len(injected_events),
        "failure_signal_sent": len(signal_events),
        "failure_detected": len(detected_events),
        "shadow_state_snapshot": len(shadow_events),
        "replacement_lookup_start": len(lookup_start_events),
    }
    if any(count != expected_failure_count for count in counted.values()):
        raise AnalysisError(
            f"recovery event counts differ in {case_name}/{repeat}: {counted}"
        )

    intervals_ns = [end - start for start, end in zip(timestamps, timestamps[1:])]
    duration_seconds = (timestamps[-1] - timestamps[0]) / 1e9
    timeline_rows = [
        {
            "case": case_name,
            "repeat": repeat,
            "output_index": output_index,
            "accumulate_seconds": (end - timestamps[0]) / 1e9,
            "inter_token_ms": (end - start) / 1e6,
        }
        for output_index, (start, end) in enumerate(
            zip(timestamps, timestamps[1:]), start=1
        )
    ]
    run_row: dict[str, Any] = {
        "case": case_name,
        "repeat": repeat,
        "failure": bool(run["failure"]),
        "retrieve": bool(run["retrieve"]),
        "topology": topology,
        "timestamp_count": len(timestamps),
        "token_count": len(tokens),
        "token_sha256": token_hash(tokens),
        "decode_duration_seconds": duration_seconds,
        "decode_tokens_per_second": (len(timestamps) - 1) / duration_seconds,
        "median_inter_token_ms": statistics.median(intervals_ns) / 1e6,
        "p95_inter_token_ms": percentile(intervals_ns, 0.95) / 1e6,
        "max_inter_token_ms": max(intervals_ns) / 1e6,
        "network_name": phases["network_name"],
        "network_spec_hash": phases["network_spec_hash"],
        "keypoints": json.dumps(phases["keypoints"], separators=(",", ":")),
        "initial_dealer": initial_dealer,
        "cuda_graph_enabled": graph_enabled,
        "cuda_graph_replays": graph_replays,
        "cuda_graph_fallback_count": graph_fallbacks,
        "failure_count": expected_failure_count,
        "failure_interval_ms_max": None,
        "failure_stall_excess_ms_median": None,
        "injection_to_first_normal_output_ms_median": None,
    }
    if not bool(run["failure"]):
        return run_row, timeline_rows, [], [], []

    injection_indices = [
        bisect.bisect_left(timestamps, int(event["time_ns"]))
        for event in injected_events
    ]
    if any(index <= 0 or index >= len(timestamps) for index in injection_indices):
        raise AnalysisError(
            f"a failure is outside the output timeline in {case_name}/{repeat}"
        )
    baseline_ms_values = [
        interval / 1e6
        for index, interval in enumerate(intervals_ns, start=1)
        if all(abs(index - failure_index) > 3 for failure_index in injection_indices)
    ]
    if not baseline_ms_values:
        raise AnalysisError(f"no steady intervals remain in {case_name}/{repeat}")
    baseline_ms = statistics.median(baseline_ms_values)

    recovery_rows: list[dict[str, Any]] = []
    recovery_event_rows: list[dict[str, Any]] = []
    nonblocking_rows: list[dict[str, Any]] = []
    current_ordinary_dealer = initial_dealer
    for recovery_index, (
        injected,
        signal,
        detected,
        shadow,
        lookup_start,
        output_index,
    ) in enumerate(
        zip(
            injected_events,
            signal_events,
            detected_events,
            shadow_events,
            lookup_start_events,
            injection_indices,
        )
    ):
        recovery_id = str(lookup_start["recovery_id"])
        recovery_events = [
            event for event in events if event.get("recovery_id") == recovery_id
        ]
        lookup_end = exactly_one(recovery_events, "replacement_lookup_end")
        lookup_probe_events = [
            event for event in events_named(events, "path_probe")
            if int(lookup_start["time_ns"])
            <= int(event["time_ns"])
            <= int(lookup_end["time_ns"])
            and event.get("task_type") == "infer"
        ]
        lookup_probe = exactly_one(lookup_probe_events, "path_probe")
        probe_execution_events = [
            event
            for event in events_named(events, "infer_path_probe_execution")
            if int(lookup_start["time_ns"])
            <= int(event["time_ns"])
            <= int(lookup_end["time_ns"])
        ]
        probe_execution = exactly_one(
            probe_execution_events, "infer_path_probe_execution"
        )
        if (
            not bool(lookup_probe.get("success"))
            or probe_execution.get("mode") != "measured"
            or int(probe_execution.get("candidate_edge_count", 0)) < 2
        ):
            raise AnalysisError(
                f"replacement lookup did not run a measured multipath probe "
                f"for {case_name}/{repeat}/{recovery_index}"
            )
        rebuild_start = exactly_one(recovery_events, "state_rebuild_start")
        rebuild_end = exactly_one(recovery_events, "state_rebuild_end")
        recovery_sends = sorted(
            events_named(recovery_events, "recovery_packet_send"),
            key=lambda event: int(event["task_index"]),
        )
        handoff = resolve_handoff_timing(recovery_events, recovery_sends)
        graph_installed = handoff["graph_installed"]
        handoff_commit = handoff["handoff_commit"]
        handoff_commit_event = handoff["handoff_commit_event"]
        replay_start = exactly_one(recovery_events, "replay_start")
        replay_send_end = handoff["replay_send_end"]
        if replay_send_end is None:
            raise AnalysisError(
                f"formal recovery {recovery_id} has no replay_send_end"
            )
        first_normal = exactly_one(
            recovery_events, "first_output_after_recovery"
        )
        ordered = (
            injected,
            signal,
            detected,
            shadow,
            lookup_start,
            lookup_end,
            rebuild_start,
            rebuild_end,
            graph_installed,
            replay_start,
            handoff_commit,
            replay_send_end,
            *(
                (handoff_commit_event,)
                if handoff_commit_event is not None else ()
            ),
            first_normal,
        )
        ordered_times = [int(event["time_ns"]) for event in ordered]
        if ordered_times != sorted(ordered_times):
            raise AnalysisError(
                f"recovery {recovery_index} is not ordered in {case_name}/{repeat}"
            )
        actual_replacement = normalized_replacement(lookup_end["replacement"])
        if topology == "ordinary":
            expected_failed = current_ordinary_dealer
            ordinary_dealers = {
                "T0-ordinary-a", "T0-ordinary-b", "T0-ordinary-c"
            }
            if (
                len(actual_replacement) != 1
                or actual_replacement[0][0] not in ordinary_dealers
                or actual_replacement[0][0] == expected_failed
                or actual_replacement[0][1:] != [0, 6]
            ):
                raise AnalysisError(
                    f"invalid ordinary replacement in {case_name}/{repeat}: "
                    f"failed={expected_failed!r}, replacement={actual_replacement!r}"
                )
            expected_replacement = actual_replacement
            current_ordinary_dealer = str(actual_replacement[0][0])
        else:
            expected_failed, expected_replacement = expected_failure_path(
                topology, initial_dealer, recovery_index
            )
        if (
            int(injected["step"]) != drop_steps[recovery_index]
            or int(injected.get("planned_failure_index", -1)) != recovery_index
            or injected.get("injection_mode")
            != "deterministic_compute_node_teardown"
            or injected.get("dealer_node") != expected_failed
            or detected.get("dealer_node") != expected_failed
            or lookup_start.get("failed_dealer") != expected_failed
            or actual_replacement != normalized_replacement(expected_replacement)
            or bool(first_normal.get("trace_invalid", True))
        ):
            raise AnalysisError(
                f"unexpected recovery {recovery_index} in {case_name}/{repeat}: "
                f"failed={injected.get('dealer_node')!r}, "
                f"step={injected.get('step')!r}, "
                f"replacement={actual_replacement!r}"
            )
        if topology == "ordinary":
            nonblocking_rows.append(
                build_nonblocking_row(
                    case_name=case_name,
                    repeat=repeat,
                    recovery_index=recovery_index,
                    recovery_id=recovery_id,
                    recovery_events=recovery_events,
                    configured_agent_to_dealer_latency_ms=(
                        agent_to_dealer_latency_ms
                    ),
                    configured_dealer_to_agent_latency_ms=(
                        dealer_to_agent_latency_ms
                    ),
                    injected=injected,
                    detected=detected,
                )
            )
        detailed_events = [
            injected,
            signal,
            detected,
            shadow,
            lookup_probe,
            probe_execution,
            *recovery_events,
        ]
        detailed_events.sort(key=lambda event: int(event["time_ns"]))
        for event_ordinal, event in enumerate(detailed_events):
            replacement = event.get("replacement")
            recovery_event_rows.append(
                {
                    "case": case_name,
                    "repeat": repeat,
                    "recovery_index": recovery_index,
                    "recovery_id": recovery_id,
                    "event_ordinal": event_ordinal,
                    "event": event["event"],
                    "time_from_injection_ms": milliseconds(injected, event),
                    "control_id": event.get("control_id", ""),
                    "step": event.get("step", ""),
                    "current_model_index": event.get(
                        "current_model_index", ""
                    ),
                    "trace_invalid": event.get("trace_invalid", ""),
                    "dealer_node": event.get("dealer_node", ""),
                    "failed_dealer": event.get("failed_dealer", ""),
                    "compute_node": event.get("compute_node", ""),
                    "packet_id": event.get("packet_id", ""),
                    "task_index": event.get("task_index", ""),
                    "task_count": event.get("task_count", ""),
                    "role": event.get("role", ""),
                    "acceptance_semantics": event.get(
                        "acceptance_semantics", ""
                    ),
                    "commit_semantics": event.get(
                        "commit_semantics", ""
                    ),
                    "replacement": (
                        "" if replacement is None else json.dumps(
                            normalized_replacement(replacement),
                            separators=(",", ":"),
                        )
                    ),
                    "replay_task_count": event.get(
                        "replay_task_count", ""
                    ),
                    "replay_input_count": event.get(
                        "replay_input_count", ""
                    ),
                }
            )
        failure_interval_ms = (
            timestamps[output_index] - timestamps[output_index - 1]
        ) / 1e6
        shadow_metrics = dict(shadow["metrics"])
        recovery_rows.append(
            {
                "case": case_name,
                "repeat": repeat,
                "topology": topology,
                "retrieve": bool(run["retrieve"]),
                "recovery_index": recovery_index,
                "recovery_id": recovery_id,
                "failed_dealer": expected_failed,
                "replacement": json.dumps(
                    actual_replacement, separators=(",", ":")
                ),
                "injected_step": int(injected["step"]),
                "injection_output_index": output_index,
                "steady_inter_token_baseline_ms": baseline_ms,
                "failure_interval_ms": failure_interval_ms,
                "failure_stall_excess_ms": max(
                    0.0, failure_interval_ms - baseline_ms
                ),
                "injection_to_detection_ms": milliseconds(injected, detected),
                "detection_to_lookup_start_ms": milliseconds(
                    detected, lookup_start
                ),
                "lookup_ms": milliseconds(lookup_start, lookup_end),
                "path_probe_ms": float(lookup_probe["duration_ns"]) / 1e6,
                "path_probe_candidate_count": int(
                    lookup_probe["candidate_count"]
                ),
                "path_probe_mode": str(probe_execution["mode"]),
                "lookup_end_to_graph_installed_ms": milliseconds(
                    lookup_end, graph_installed
                ),
                "graph_installed_to_handoff_commit_ms": milliseconds(
                    graph_installed, handoff_commit
                ),
                "handoff_commit_to_first_normal_output_ms": milliseconds(
                    handoff_commit, first_normal
                ),
                "lookup_end_to_rebuild_end_ms": milliseconds(
                    lookup_end, rebuild_end
                ),
                "rebuild_end_to_replay_send_end_ms": milliseconds(
                    rebuild_end, replay_send_end
                ),
                "replay_send_to_first_normal_output_ms": milliseconds(
                    replay_send_end, first_normal
                ),
                "detection_to_first_normal_output_ms": milliseconds(
                    detected, first_normal
                ),
                "injection_to_first_normal_output_ms": milliseconds(
                    injected, first_normal
                ),
                "replay_task_count": int(rebuild_end["replay_task_count"]),
                "replay_input_count": int(rebuild_end["replay_input_count"]),
                "shadow_total_retained_bytes": int(
                    shadow_metrics["total_retained_bytes"]
                ),
                "shadow_kv_bytes": int(shadow_metrics["kv_bytes"]),
                "shadow_update_cpu_ns": int(shadow_metrics["update_cpu_ns"]),
                "checkpoint_block_count": int(
                    shadow_metrics["checkpoint_block_count"]
                ),
                "shadow_retained_replay_task_count": int(
                    shadow_metrics["replay_task_count"]
                ),
            }
        )

    run_row.update(
        {
            "failure_interval_ms_max": max(
                row["failure_interval_ms"] for row in recovery_rows
            ),
            "failure_stall_excess_ms_median": statistics.median(
                row["failure_stall_excess_ms"] for row in recovery_rows
            ),
            "injection_to_first_normal_output_ms_median": statistics.median(
                row["injection_to_first_normal_output_ms"]
                for row in recovery_rows
            ),
        }
    )
    return (
        run_row,
        timeline_rows,
        recovery_rows,
        recovery_event_rows,
        nonblocking_rows,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentage_change(new: float, old: float) -> Optional[float]:
    return None if old == 0 else ((new / old) - 1.0) * 100.0


def summarize_runs(
    rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
    planned_failure_count: int,
) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for case_name in CASE_ORDER:
        selected = [row for row in rows if row["case"] == case_name]
        selected_recoveries = [
            row for row in recovery_rows if row["case"] == case_name
        ]
        if not selected:
            continue
        cases[case_name] = {
            "runs": len(selected),
            "recovery_events": len(selected_recoveries),
            "decode_tokens_per_second_mean": statistics.mean(
                float(row["decode_tokens_per_second"]) for row in selected
            ),
            "decode_tokens_per_second_median": statistics.median(
                float(row["decode_tokens_per_second"]) for row in selected
            ),
            "decode_tokens_per_second_min": min(
                float(row["decode_tokens_per_second"]) for row in selected
            ),
            "decode_tokens_per_second_max": max(
                float(row["decode_tokens_per_second"]) for row in selected
            ),
            "decode_duration_seconds_mean": statistics.mean(
                float(row["decode_duration_seconds"]) for row in selected
            ),
            "median_inter_token_ms_mean": statistics.mean(
                float(row["median_inter_token_ms"]) for row in selected
            ),
            "max_inter_token_ms_median": statistics.median(
                float(row["max_inter_token_ms"]) for row in selected
            ),
            "token_hashes": sorted({row["token_sha256"] for row in selected}),
            "keypoints": sorted({row["keypoints"] for row in selected}),
            "initial_dealers": sorted({row["initial_dealer"] for row in selected}),
            "recovery_medians": {
                field: median_or_none(row[field] for row in selected_recoveries)
                for field in RECOVERY_FIELDS
            },
            "recovery_means": {
                field: mean_or_none(row[field] for row in selected_recoveries)
                for field in RECOVERY_FIELDS
            },
            "recovery_by_failure_index": {
                str(index): {
                    field: median_or_none(
                        row[field]
                        for row in selected_recoveries
                        if int(row["recovery_index"]) == index
                    )
                    for field in RECOVERY_FIELDS
                }
                for index in range(planned_failure_count)
            },
            "recovery_means_by_failure_index": {
                str(index): {
                    field: mean_or_none(
                        row[field]
                        for row in selected_recoveries
                        if int(row["recovery_index"]) == index
                    )
                    for field in RECOVERY_FIELDS
                }
                for index in range(planned_failure_count)
            },
        }

    comparisons: dict[str, Any] = {
        "all_runs_token_identical": len(
            {row["token_sha256"] for row in rows}
        ) == 1,
        "all_token_hashes": sorted({row["token_sha256"] for row in rows}),
        "all_runs_cuda_graph_enabled": all(
            bool(row["cuda_graph_enabled"]) for row in rows
        ),
        "all_runs_cuda_graph_fallback_free": all(
            int(row["cuda_graph_fallback_count"]) == 0 for row in rows
        ),
        "planned_failure_count_per_recovery_run": planned_failure_count,
        "all_recovery_runs_have_planned_failures": all(
            int(row["failure_count"]) == planned_failure_count
            for row in rows
            if bool(row["failure"])
        ),
    }
    if "normal-reprefill" in cases and "normal-mix" in cases:
        comparisons["mix_steady_throughput_change_percent"] = percentage_change(
            cases["normal-mix"]["decode_tokens_per_second_mean"],
            cases["normal-reprefill"]["decode_tokens_per_second_mean"],
        )
    for prefix in ("recover", "recover-split"):
        reprefill = f"{prefix}-reprefill"
        mix = f"{prefix}-mix"
        if reprefill in cases and mix in cases:
            comparison_prefix = prefix.replace("-", "_")
            comparisons[f"{comparison_prefix}_mix_throughput_change_percent"] = (
                percentage_change(
                    cases[mix]["decode_tokens_per_second_mean"],
                    cases[reprefill]["decode_tokens_per_second_mean"],
                )
            )
            comparisons[f"{comparison_prefix}_mix_stall_change_percent"] = (
                percentage_change(
                    cases[mix]["recovery_medians"]["failure_stall_excess_ms"],
                    cases[reprefill]["recovery_medians"]["failure_stall_excess_ms"],
                )
            )
    for method in ("reprefill", "mix"):
        ordinary = f"recover-{method}"
        split = f"recover-split-{method}"
        normal = f"normal-{method}"
        if ordinary in cases and split in cases:
            comparisons[f"split_{method}_throughput_change_percent"] = (
                percentage_change(
                    cases[split]["decode_tokens_per_second_mean"],
                    cases[ordinary]["decode_tokens_per_second_mean"],
                )
            )
            comparisons[f"split_{method}_stall_change_percent"] = (
                percentage_change(
                    cases[split]["recovery_medians"]["failure_stall_excess_ms"],
                    cases[ordinary]["recovery_medians"]["failure_stall_excess_ms"],
                )
            )
        if normal in cases and ordinary in cases:
            comparisons[f"ordinary_failure_{method}_throughput_change_percent"] = (
                percentage_change(
                    cases[ordinary]["decode_tokens_per_second_mean"],
                    cases[normal]["decode_tokens_per_second_mean"],
                )
            )
    return {"schema_version": 5, "cases": cases, "comparisons": comparisons}


def plot_timeline(
    result_root: Path,
    timeline_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
) -> None:
    figures = result_root / "figures"
    figures.mkdir(exist_ok=True)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in timeline_rows:
        grouped.setdefault((row["case"], int(row["output_index"])), []).append(row)
    fig, (upper, lower) = plt.subplots(
        2, 1, figsize=(9.4, 6.8), dpi=180, sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )
    for case_name in CASE_ORDER:
        points = []
        for (name, index), values in grouped.items():
            if name == case_name:
                points.append(
                    (
                        index,
                        statistics.mean(
                            float(value["accumulate_seconds"]) for value in values
                        ),
                        statistics.median(
                            float(value["inter_token_ms"]) for value in values
                        ),
                    )
                )
        if not points:
            continue
        points.sort()
        label, color, line_style = CASE_STYLE[case_name]
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
            alpha=0.85,
        )
    recovery_indices = sorted({
        int(row["recovery_index"]) for row in recovery_rows
    })
    for recovery_index in recovery_indices:
        indices = [
            int(row["injection_output_index"])
            for row in recovery_rows
            if int(row["recovery_index"]) == recovery_index
        ]
        if not indices:
            continue
        location = statistics.median(indices)
        for axis in (upper, lower):
            axis.axvline(location, color="gray", linestyle=":", linewidth=1.0)
        upper.text(
            location,
            0.97 if recovery_index % 2 == 0 else 0.88,
            f"F{recovery_index + 1}",
            transform=upper.get_xaxis_transform(), ha="center", va="top",
            color="gray", fontsize=7,
        )
    upper.set_ylabel("Accumulated decode time (s)")
    lower.set_ylabel("Inter-token interval (ms)")
    lower.set_xlabel("Decode output index")
    for axis in (upper, lower):
        axis.grid(True, alpha=0.22)
    upper.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=8,
        ncol=3,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(figures / "basic_recovery_timeline.png", dpi=330)
    fig.savefig(figures / "basic_recovery_timeline.pdf")
    plt.close(fig)


def plot_breakdown(result_root: Path, summary: dict[str, Any]) -> None:
    cases = summary["cases"]
    names = [
        name
        for name in (
            "recover-reprefill",
            "recover-mix",
            "recover-split-reprefill",
            "recover-split-mix",
        )
        if name in cases
    ]
    if not names:
        return
    failure_count = int(
        summary["comparisons"]["planned_failure_count_per_recovery_run"]
    )
    fields = (
        ("Inject to detect", "injection_to_detection_ms"),
        ("Detect to lookup", "detection_to_lookup_start_ms"),
        ("Replacement lookup", "lookup_ms"),
        ("State rebuild + graph install", "lookup_end_to_graph_installed_ms"),
        (
            "Recovery enqueue + handoff commit",
            "graph_installed_to_handoff_commit_ms",
        ),
        (
            "Post-commit Dealer recovery + next normal output",
            "handoff_commit_to_first_normal_output_ms",
        ),
    )
    labels = {
        "recover-reprefill": "Recover\nRe-Prefill",
        "recover-mix": "Recover\nMix",
        "recover-split-reprefill": "Recover+Split\nRe-Prefill",
        "recover-split-mix": "Recover+Split\nMix",
    }
    colors = (
        "#b3de69", "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3"
    )
    column_count = min(4, failure_count)
    row_count = math.ceil(failure_count / column_count)
    fig, axes_grid = plt.subplots(
        row_count,
        column_count,
        figsize=(3.6 * column_count, 3.4 * row_count + 1.2),
        dpi=180,
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_grid.flat)
    for recovery_index in range(failure_count):
        axis = axes[recovery_index]
        bottoms = [0.0] * len(names)
        for (label, field), color in zip(fields, colors):
            values = [
                float(
                    cases[name]["recovery_means_by_failure_index"][
                        str(recovery_index)
                    ][field]
                    or 0.0
                )
                for name in names
            ]
            axis.bar(
                range(len(names)), values, bottom=bottoms,
                label=label, color=color,
            )
            bottoms = [
                bottom + value for bottom, value in zip(bottoms, values)
            ]
        axis.set_title(f"Failure {recovery_index + 1}")
        axis.set_xticks(
            range(len(names)), [labels[name] for name in names], fontsize=8
        )
        axis.grid(axis="y", alpha=0.25)
    for axis in axes[failure_count:]:
        axis.set_visible(False)
    axes[0].set_ylabel("Mean latency for each failure index (ms)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    run_counts = {int(cases[name]["runs"]) for name in names}
    repeat_suffix = (
        f" across {next(iter(run_counts))} repeats"
        if len(run_counts) == 1
        else ""
    )
    fig.suptitle(
        f"Mean recovery-stage latency{repeat_suffix}",
        fontsize=11,
        y=0.995,
    )
    fig.legend(
        handles, legend_labels, loc="upper center", ncol=3,
        fontsize=8, bbox_to_anchor=(0.5, 0.955),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    figures = result_root / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "recovery_stage_breakdown.png", dpi=330)
    fig.savefig(figures / "recovery_stage_breakdown.pdf")
    plt.close(fig)


def plot_nonblocking_causality(
    result_root: Path,
    nonblocking_rows: list[dict[str, Any]],
) -> None:
    if not nonblocking_rows:
        return
    selected_rows = [
        row for row in nonblocking_rows
        if row["case"] == "recover-mix"
        and int(row["recovery_index"]) == 0
    ]
    if not selected_rows:
        anchor = nonblocking_rows[0]
        selected_rows = [
            row for row in nonblocking_rows
            if row["case"] == anchor["case"]
            and int(row["recovery_index"])
            == int(anchor["recovery_index"])
        ]
    case_name = str(selected_rows[0]["case"])
    recovery_index = int(selected_rows[0]["recovery_index"])

    def mean_duration(start_field: str, end_field: str) -> float:
        return statistics.mean(
            (int(row[end_field]) - int(row[start_field])) / 1e6
            for row in selected_rows
        )

    def mean_from_lookup(field: str) -> float:
        return statistics.mean(
            (
                int(row[field]) - int(row["lookup_end_time_ns"])
            ) / 1e6
            for row in selected_rows
        )

    fig, (stages, dispatch, causal) = plt.subplots(
        1, 3, figsize=(15.8, 5.1), dpi=180,
        gridspec_kw={"width_ratios": [0.8, 0.95, 1.5]},
    )
    stage_bounds = (
        ("Failure signal", "failure_time_ns", "detection_time_ns"),
        ("Detect → lookup", "detection_time_ns", "lookup_start_time_ns"),
        ("Measured path lookup", "lookup_start_time_ns", "lookup_end_time_ns"),
        (
            "Local graph rebuild",
            "lookup_end_time_ns",
            "replacement_graph_installed_time_ns",
        ),
        (
            "Enqueue → handoff commit",
            "replacement_graph_installed_time_ns",
            "handoff_commit_time_ns",
        ),
    )
    durations = [
        mean_duration(start, end)
        for _, start, end in stage_bounds
    ]
    stages.barh(
        range(len(stage_bounds)), durations,
        color=("#9ecae1", "#c6dbef", "#fdae6b", "#a1d99b", "#bcbddc"),
    )
    stages.set_yticks(
        range(len(stage_bounds)), [label for label, _, _ in stage_bounds]
    )
    stages.invert_yaxis()
    stages.set_xlabel("Mean duration (ms)")
    stages.set_title("Mean path to handoff commit")
    stages.grid(axis="x", alpha=0.25)
    for index, duration in enumerate(durations):
        stages.text(duration, index, f" {duration:.1f}", va="center", fontsize=8)

    dispatch_events = (
        ("Create send", "create_send_time_ns"),
        ("Replacement graph installed", "replacement_graph_installed_time_ns"),
        ("First recovery send start", "first_send_time_ns"),
        (
            "Frontier send start\n(last recovery packet)",
            "last_send_start_time_ns",
        ),
        ("All sends returned", "last_enqueue_proxy_time_ns"),
        ("Handoff committed", "handoff_commit_time_ns"),
        ("Handler return", "handler_return_time_ns"),
    )
    dispatch_values = [
        statistics.mean(
            (
                int(row[field])
                - min(int(row[item_field]) for _, item_field in dispatch_events)
            ) / 1e6
            for row in selected_rows
        )
        for _, field in dispatch_events
    ]
    dispatch.scatter(
        dispatch_values,
        range(len(dispatch_events)),
        color="#3182bd",
        s=34,
        zorder=3,
    )
    dispatch.plot(
        dispatch_values,
        range(len(dispatch_events)),
        color="#9ecae1",
        linewidth=1.2,
    )
    dispatch.set_yticks(
        range(len(dispatch_events)),
        [label for label, _ in dispatch_events],
        fontsize=8,
    )
    dispatch.invert_yaxis()
    dispatch.set_xlabel("Mean time since first local event (ms)")
    dispatch.set_title("Mean Agent graph install + handoff")
    dispatch.grid(axis="x", alpha=0.25)
    for value, index in zip(dispatch_values, range(len(dispatch_events))):
        dispatch.text(value, index, f"  {value:.3f}", va="center", fontsize=7)

    lanes = {"Agent": 2.0, "Network": 1.0, "Dealer": 0.0}
    for label, lane in lanes.items():
        causal.axhline(lane, color="#dddddd", linewidth=0.8, zorder=0)
    causal.axvline(0.0, color="#969696", linestyle="--", linewidth=0.8)
    # These Agent-local events are less than a millisecond apart, while the
    # causal panel spans the full network and Dealer recovery interval. Keep
    # their true x coordinates and offset them slightly within the Agent lane
    # so no marker is hidden by another.
    graph_installed_x = mean_from_lookup(
        "replacement_graph_installed_time_ns"
    )
    causal.scatter(
        graph_installed_x, lanes["Agent"] + 0.18,
        color="#6a51a3", marker="*", s=54, zorder=5,
        label="replacement graph installed",
    )
    causal.annotate(
        "replacement graph installed\n(recovery packets not sent yet)",
        xy=(graph_installed_x, lanes["Agent"] + 0.18),
        xytext=(8.0, lanes["Agent"] + 0.30),
        fontsize=7,
        va="center",
        arrowprops={"arrowstyle": "->", "color": "#6a51a3", "lw": 0.8},
    )
    handoff_x = mean_from_lookup("handoff_commit_time_ns")
    handoff_is_observed = all(
        bool(row["handoff_commit_observed"]) for row in selected_rows
    )
    handoff_label = (
        "handoff committed (logical recovery complete)"
        if handoff_is_observed
        else "handoff commit (legacy-derived proxy)"
    )
    causal.scatter(
        mean_from_lookup("last_enqueue_proxy_time_ns"),
        lanes["Agent"] - 0.13,
        color="#3182bd", s=30, zorder=3,
        label="all required sends returned",
    )
    causal.scatter(
        handoff_x, lanes["Agent"] + 0.05,
        color="#54278f", marker="D", s=34, zorder=5,
        label=handoff_label,
    )
    causal.annotate(
        "logical recovery complete\n(no Dealer completion/ACK awaited)",
        xy=(handoff_x, lanes["Agent"] + 0.05),
        xytext=(8.0, lanes["Agent"] - 0.31),
        fontsize=7,
        va="center",
        arrowprops={"arrowstyle": "->", "color": "#54278f", "lw": 0.8},
    )
    causal.scatter(
        mean_from_lookup("handler_return_time_ns"), lanes["Agent"] - 0.02,
        color="#de2d26", marker="v", s=34, zorder=4,
        label="recovery handler return",
    )
    state_start = mean_from_lookup("state_start_time_ns")
    state_end = mean_from_lookup("state_end_time_ns")
    causal.plot(
        [state_start, state_end], [lanes["Dealer"]] * 2,
        color="#e6550d", linewidth=7, solid_capstyle="butt",
        label="state load",
    )
    execute_start = mean_from_lookup("first_execute_time_ns")
    frontier = mean_from_lookup("frontier_time_ns")
    causal.plot(
        [execute_start, frontier], [lanes["Dealer"]] * 2,
        color="#31a354", linewidth=7, solid_capstyle="butt",
        label="chunk replay + frontier task",
    )
    dealer_points = (
        (mean_from_lookup("create_receive_time_ns"), "create receive", "#756bb1", "o"),
        (state_end, "state ready", "#e6550d", "s"),
        (mean_from_lookup("first_complete_time_ns"), "first task complete", "#31a354", "^"),
        (frontier, "frontier materialized", "#006d2c", "D"),
        (mean_from_lookup("frontier_output_send_time_ns"), "frontier output send", "#636363", "P"),
    )
    for time_value, label, color, marker in dealer_points:
        causal.scatter(
            time_value, lanes["Dealer"], color=color, marker=marker,
            s=30, zorder=3, label=label,
        )
    packet_counts = {
        len(row["packet_send_start_times_ns"]) for row in selected_rows
    } | {
        len(row["packet_receive_times_ns"]) for row in selected_rows
    }
    if len(packet_counts) != 1:
        raise AnalysisError(
            "cannot average causal rows with different packet counts"
        )
    packet_count = next(iter(packet_counts))
    for packet_index in range(packet_count):
        sent = statistics.mean(
            (
                int(row["packet_send_start_times_ns"][packet_index])
                - int(row["lookup_end_time_ns"])
            ) / 1e6
            for row in selected_rows
        )
        received = statistics.mean(
            (
                int(row["packet_receive_times_ns"][packet_index])
                - int(row["lookup_end_time_ns"])
            ) / 1e6
            for row in selected_rows
        )
        causal.annotate(
            "", xy=(received, lanes["Dealer"] + 0.06),
            xytext=(sent, lanes["Agent"] - 0.06),
            arrowprops={"arrowstyle": "->", "color": "#756bb1", "lw": 0.8},
        )
    frontier_output_send = mean_from_lookup("frontier_output_send_time_ns")
    frontier_output_arrival = mean_from_lookup(
        "frontier_output_arrival_time_ns"
    )
    causal.annotate(
        "", xy=(frontier_output_arrival, lanes["Agent"] - 0.06),
        xytext=(frontier_output_send, lanes["Dealer"] + 0.06),
        arrowprops={"arrowstyle": "->", "color": "#636363", "lw": 1.0},
    )
    causal.scatter(
        frontier_output_arrival, lanes["Agent"],
        color="#636363", marker="P", s=30, zorder=3,
        label="frontier output arrives",
    )
    causal.set_yticks(list(lanes.values()), list(lanes))
    causal.set_xlabel("Mean time since path lookup complete (ms)")
    causal.set_title("Mean Agent ↔ Dealer chain")
    causal.grid(axis="x", alpha=0.25)
    causal.set_xlim(left=0.0)
    causal.set_ylim(-0.12, 2.38)
    causal.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1,
        fontsize=7, frameon=True,
    )
    fig.suptitle(
        f"{case_name} / failure {recovery_index + 1} / mean of "
        f"{len(selected_rows)} repeats",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.88, 0.95))
    figures = result_root / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "nonblocking_causality.png", dpi=330)
    fig.savefig(figures / "nonblocking_causality.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    manifest = read_json(result_root / "manifest.json")
    drop_steps = tuple(int(step) for step in manifest["parameters"]["drop_steps"])
    agent_to_dealer_latency_ms = float(
        manifest["parameters"].get("agent_to_dealer_latency_ms", 0.0)
    )
    dealer_to_agent_latency_ms = float(
        manifest["parameters"].get("dealer_to_agent_latency_ms", 0.0)
    )
    recovery_replay = str(
        manifest["parameters"].get("recovery_replay", "sequential")
    )
    if not drop_steps:
        raise AnalysisError("expected at least one configured drop step")
    selected_topologies = {
        str(run["topology"])
        for run in manifest["runs"]
    }
    if "split" in selected_topologies and len(drop_steps) != 3:
        raise AnalysisError(
            "split topology results require exactly three drop steps"
        )
    rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    recovery_event_rows: list[dict[str, Any]] = []
    nonblocking_rows: list[dict[str, Any]] = []
    for run in manifest["runs"]:
        row, timeline, recoveries, recovery_events, nonblocking = analyze_run(
            result_root,
            run,
            drop_steps,
            agent_to_dealer_latency_ms,
            dealer_to_agent_latency_ms,
        )
        rows.append(row)
        timeline_rows.extend(timeline)
        recovery_rows.extend(recoveries)
        recovery_event_rows.extend(recovery_events)
        nonblocking_rows.extend(nonblocking)
    rows.sort(key=lambda row: (CASE_ORDER.index(row["case"]), row["repeat"]))
    timeline_rows.sort(
        key=lambda row: (
            CASE_ORDER.index(row["case"]), row["repeat"], row["output_index"]
        )
    )
    recovery_rows.sort(
        key=lambda row: (
            CASE_ORDER.index(row["case"]), row["repeat"], row["recovery_index"]
        )
    )
    recovery_event_rows.sort(
        key=lambda row: (
            CASE_ORDER.index(row["case"]),
            row["repeat"],
            row["recovery_index"],
            row["event_ordinal"],
        )
    )
    nonblocking_rows.sort(
        key=lambda row: (
            CASE_ORDER.index(row["case"]),
            row["repeat"],
            row["recovery_index"],
        )
    )
    summary = summarize_runs(rows, recovery_rows, len(drop_steps))
    write_csv(result_root / "runs.csv", rows)
    write_csv(result_root / "timeline.csv", timeline_rows)
    write_csv(result_root / "recoveries.csv", recovery_rows)
    write_csv(result_root / "recovery_events.csv", recovery_event_rows)
    csv_nonblocking_rows = [
        {
            key: (
                json.dumps(value, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else value
            )
            for key, value in row.items()
        }
        for row in nonblocking_rows
    ]
    write_csv(
        result_root / "nonblocking_recoveries.csv", csv_nonblocking_rows
    )
    write_json(result_root / "summary.json", summary)
    proof = {
        "schema_version": 2,
        "claim": (
            "Shadow recovery installs the replacement graph, hands every "
            f"required {recovery_replay} recovery/frontier packet to the local "
            "send path, commits the Promise handoff, and returns without "
            "waiting for Dealer completion or acknowledgement."
        ),
        "recovery_replay": recovery_replay,
        "agent_to_dealer_latency_ms": agent_to_dealer_latency_ms,
        "dealer_to_agent_latency_ms": dealer_to_agent_latency_ms,
        "planned_failure_count_per_recovery_run": len(drop_steps),
        "recovery_count": len(nonblocking_rows),
        "all_recoveries_pass": bool(nonblocking_rows) and all(
            bool(row["all_assertions_pass"]) for row in nonblocking_rows
        ),
        "final_token_sequence_matches_greedy_baseline": bool(
            summary["comparisons"]["all_runs_token_identical"]
        ),
        "minimum_last_enqueue_proxy_to_first_complete_ms": (
            min(
                float(row["last_enqueue_proxy_to_first_complete_ms"])
                for row in nonblocking_rows
            )
            if nonblocking_rows else None
        ),
        "minimum_handoff_commit_to_first_complete_ms": (
            min(
                float(row["handoff_commit_to_first_complete_ms"])
                for row in nonblocking_rows
            )
            if nonblocking_rows else None
        ),
        "minimum_observed_handoff_event_to_first_complete_ms": (
            min(
                float(row["handoff_commit_event_to_first_complete_ms"])
                for row in nonblocking_rows
                if bool(row["handoff_commit_observed"])
            )
            if any(
                bool(row["handoff_commit_observed"])
                for row in nonblocking_rows
            ) else None
        ),
        "minimum_handler_return_to_frontier_ms": (
            min(
                float(row["handler_return_to_frontier_ms"])
                for row in nonblocking_rows
            )
            if nonblocking_rows else None
        ),
        "handoff_commit_sources": sorted({
            str(row["handoff_commit_source"])
            for row in nonblocking_rows
        }),
        "recoveries": nonblocking_rows,
    }
    proof["overall_pass"] = (
        proof["all_recoveries_pass"]
        and proof["final_token_sequence_matches_greedy_baseline"]
    )
    write_json(result_root / "nonblocking_proof.json", proof)
    plot_timeline(result_root, timeline_rows, recovery_rows)
    plot_breakdown(result_root, summary)
    plot_nonblocking_causality(result_root, nonblocking_rows)
    if not proof["overall_pass"]:
        raise AnalysisError(
            "non-blocking causal checks or greedy token correctness failed; "
            "see nonblocking_proof.json"
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
