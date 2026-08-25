#!/usr/bin/env python3
"""Compare opt-in PEFT checkpoint fingerprints between Normal and Recover."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    root = args.result_root.resolve()
    manifest = read_json(root / "manifest.json")

    by_run: dict[
        tuple[str, int],
        dict[tuple[int, str], list[dict[str, Any]]],
    ] = {}
    event_streams: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in manifest["runs"]:
        key = (str(run["condition"]), int(run["repeat"]))
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        events = read_events(root / run["run_dir"] / "events.jsonl")
        event_streams[key] = events
        for event in events:
            if event.get("event") != "train_checkpoint_state_fingerprint":
                continue
            grouped[(
                int(event["checkpoint_version"]),
                str(event["user_model"]),
            )].append(event)
        by_run[key] = dict(grouped)

    normal_keys = sorted(key for key in by_run if key[0] == "normal")
    if not normal_keys:
        raise RuntimeError("missing Normal run")
    comparisons = []
    for run_key, actual in sorted(by_run.items()):
        if run_key[0] == "normal":
            continue
        baseline_key = ("normal", run_key[1])
        if baseline_key not in by_run:
            baseline_key = normal_keys[0]
        baseline = by_run[baseline_key]
        shared = sorted(set(baseline) & set(actual))
        mismatches = []
        conflicts = []
        for state_key in shared:
            normal_events = baseline[state_key]
            actual_events = actual[state_key]
            normal_hashes = {event["state_fingerprint"] for event in normal_events}
            actual_hashes = {event["state_fingerprint"] for event in actual_events}
            if len(normal_hashes) != 1 or len(actual_hashes) != 1:
                conflicts.append({
                    "checkpoint_version": state_key[0],
                    "user_model": state_key[1],
                    "normal_hashes": sorted(normal_hashes),
                    "recover_hashes": sorted(actual_hashes),
                })
            if normal_hashes != actual_hashes:
                component_names = sorted(set().union(*(
                    set(event.get("component_fingerprints", {}))
                    for event in [*normal_events, *actual_events]
                )))
                component_mismatches = [
                    component
                    for component in component_names
                    if {
                        event.get("component_fingerprints", {}).get(component)
                        for event in normal_events
                    } != {
                        event.get("component_fingerprints", {}).get(component)
                        for event in actual_events
                    }
                ]
                mismatches.append({
                    "checkpoint_version": state_key[0],
                    "user_model": state_key[1],
                    "component_mismatches": component_mismatches,
                    "normal_hashes": sorted(normal_hashes),
                    "recover_hashes": sorted(actual_hashes),
                })
        comparisons.append({
            "condition": run_key[0],
            "repeat": run_key[1],
            "normal_condition": baseline_key[0],
            "normal_fingerprint_event_count": sum(map(len, baseline.values())),
            "recover_fingerprint_event_count": sum(map(len, actual.values())),
            "shared_checkpoint_model_count": len(shared),
            "normal_only_keys": [list(key) for key in sorted(set(baseline) - set(actual))],
            "recover_only_keys": [list(key) for key in sorted(set(actual) - set(baseline))],
            "intra_run_conflicts": conflicts,
            "mismatches": mismatches,
            "first_mismatch": mismatches[0] if mismatches else None,
            "all_shared_checkpoint_states_exactly_equal": not mismatches,
        })

    replay_orders = []
    for run_key, events in sorted(event_streams.items()):
        if run_key[0] == "normal":
            continue
        recovery_ids = [
            str(event["recovery_id"])
            for event in events
            if event.get("event") == "replay_start"
        ]
        for recovery_id in recovery_ids:
            sent = [
                str(event["minibatch_id"])
                for event in events
                if event.get("event") == "recovery_packet_send_start"
                and event.get("packet_kind") == "recovery_replay"
                and event.get("recovery_id") == recovery_id
            ]
            completed = [
                str(event["minibatch_id"])
                for event in events
                if event.get("event") == "train_task_execute_end"
                and event.get("origin") == "recovery_replay"
                and event.get("recovery_id") == recovery_id
            ]
            replay_orders.append({
                "condition": run_key[0],
                "repeat": run_key[1],
                "recovery_id": recovery_id,
                "sent_minibatch_order": sent,
                "completed_minibatch_order": completed,
                "completion_preserved_send_order": completed == sent,
            })

    report = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "comparisons": comparisons,
        "replay_orders": replay_orders,
        "any_replay_completion_reordered": any(
            not item["completion_preserved_send_order"]
            for item in replay_orders
        ),
        "all_shared_checkpoint_states_exactly_equal": bool(comparisons) and all(
            item["all_shared_checkpoint_states_exactly_equal"]
            for item in comparisons
        ),
    }
    output = root / "state_fingerprint_comparison.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
