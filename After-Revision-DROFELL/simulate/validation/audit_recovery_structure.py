#!/usr/bin/env python3
"""Compare one hardware and one simulated recovery at event/packet granularity."""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Any

DIAGNOSTIC_ONLY = {
    "batch_merge",
    "infer_output_arrival",
    "recovery_intermediate_arrival",
    "recovery_shadow_snapshot",
}
PACKET_EVENTS = (
    "recovery_packet_send",
    "recovery_packet_send_return",
    "recovery_packet_receive",
    "recovery_task_execute_start",
    "recovery_task_execute_end",
    "recovery_output_send",
)


def packet_set(rows: list[dict[str, Any]], event: str) -> set[str]:
    return {
        str(row.get("packet_id", ""))
        for row in rows
        if row["event"] == event and row.get("packet_id")
    }


def task_signature(rows: list[dict[str, Any]]) -> list[tuple[int, str, int]]:
    signatures = []
    for row in rows:
        if row["event"] != "recovery_packet_send":
            continue
        signatures.append((
            int(row.get("task_index") or 0),
            str(row.get("role") or ""),
            int(row.get("task_count") or 0),
        ))
    return sorted(signatures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--case", default="recover-reprefill")
    parser.add_argument("--repeat", default="0")
    parser.add_argument("--recovery-index", default="0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    real = [
        dict(row) for row in csv.DictReader(args.real.open(newline=""))
        if row["case"] == args.case
        and row["repeat"] == args.repeat
        and row["recovery_index"] == args.recovery_index
    ]
    sim_all = [
        json.loads(line) for line in args.sim.read_text().splitlines()
        if line.strip()
    ]
    first_failure = next(
        index for index, row in enumerate(sim_all)
        if row["event"] == "failure_injected"
    )
    sim = sim_all[first_failure:]
    real_types = collections.Counter(row["event"] for row in real)
    sim_types = collections.Counter(row["event"] for row in sim)
    missing = sorted(set(real_types) - set(sim_types))
    count_mismatches = {
        event: {"real": count, "sim": sim_types.get(event, 0)}
        for event, count in real_types.items()
        if sim_types.get(event, 0) != count
    }

    real_sequence = [
        row["event"] for row in real
        if row["event"] not in DIAGNOSTIC_ONLY
        and row["event"] != "shadow_state_snapshot"
    ]
    sim_sequence = [
        row["event"] for row in sim
        if row["event"] not in DIAGNOSTIC_ONLY
        and row["event"] != "shadow_state_snapshot"
        and row["event"] in real_types
    ]
    sequence_differences = [
        {"ordinal": index, "real": real_event, "sim": sim_event}
        for index, (real_event, sim_event) in enumerate(
            zip(real_sequence, sim_sequence)
        )
        if real_event != sim_event
    ]
    if len(real_sequence) != len(sim_sequence):
        sequence_differences.append({
            "ordinal": min(len(real_sequence), len(sim_sequence)),
            "real_remaining": real_sequence[len(sim_sequence):],
            "sim_remaining": sim_sequence[len(real_sequence):],
        })

    packet_invariants: dict[str, Any] = {}
    for label, rows in (("real", real), ("sim", sim)):
        sets = {event: packet_set(rows, event) for event in PACKET_EVENTS}
        sent = sets["recovery_packet_send"]
        packet_invariants[label] = {
            "sent_count": len(sent),
            "send_equals_send_return": (
                sent == sets["recovery_packet_send_return"]
            ),
            "send_equals_receive": sent == sets["recovery_packet_receive"],
            "send_equals_execute_start": (
                sent == sets["recovery_task_execute_start"]
            ),
            "send_equals_execute_end": (
                sent == sets["recovery_task_execute_end"]
            ),
            "send_equals_output": sent == sets["recovery_output_send"],
            "task_signature": task_signature(rows),
        }
    packet_pass = all(
        value is True
        for result in packet_invariants.values()
        for key, value in result.items()
        if key.startswith("send_equals")
    ) and (
        packet_invariants["real"]["task_signature"]
        == packet_invariants["sim"]["task_signature"]
    )

    checks = {
        "all_hardware_event_types_present": not missing,
        "event_multiplicity_equal": not count_mismatches,
        "event_order_equal": not sequence_differences,
        "packet_relations_and_roles_equal": packet_pass,
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "evidence_id": "g1-recovery-structure-hardware-vs-sim-v1",
        "passed": passed,
        "paper_eligible": passed,
        "checks": checks,
        "missing_event_types": missing,
        "event_count_mismatches": count_mismatches,
        "event_sequence_differences": sequence_differences,
        "packet_invariants": packet_invariants,
        "real_source": str(args.real),
        "sim_source": str(args.sim),
        "reason": (
            None if passed else
            "archived hardware recovery is not structurally identical to the "
            "current-code simulation; rerun hardware before claiming G1"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"G1={'PASS' if passed else 'FAIL'}")
    for name, ok in checks.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"wrote {args.out}")
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()

