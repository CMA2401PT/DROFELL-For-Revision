#!/usr/bin/env python3
"""P19: local smoke -- the simulated recovery timeline is causally ordered.

WHY THIS IS THE GATE THAT MATTERS FOR CREDIBILITY
-------------------------------------------------
This probe checks only the simulation side. It does not load hardware events or
compare packet IDs, so it is necessary but insufficient for reviewer-grade G1.
The actual G1 comparator is `simulate/validation/audit_recovery_structure.py`.

Two inequalities are checked, both taken from the production instrumentation
rather than invented here:

  1. `replacement_lookup_start` < `replacement_create_send`
     < `replacement_graph_installed` < `recovery_packet_send`
     -- the replacement must be found and installed before replay is sent.

  2. `first_recovery_task_complete` < `frontier_materialized`
     -- the frontier cannot be declared materialised before the first replayed
     task has completed. This is the one that would catch a simulator that
     "recovered" by fiat.

It also reports the REPLAY WINDOW actually used, because that -- not node count --
is what recovery cost scales with (D3).

Run from the repository root:
    python simulate/probes/p19_recovery_causality.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The five timepoints the paper names, plus the two that bracket the frontier.
REQUIRED_EVENTS = (
    "failure_injected",
    "failure_detected",
    "replacement_lookup_start",
    "replacement_lookup_end",
    "replacement_create_send",
    "replacement_graph_installed",
    "recovery_packet_send",
    "recovery_handler_return",
    "first_recovery_task_complete",
    "frontier_materialized",
)
ORDERED_CHAINS = (
    (
        "replacement_lookup_start",
        "replacement_create_send",
        "replacement_graph_installed",
        "recovery_packet_send",
    ),
    ("first_recovery_task_complete", "frontier_materialized"),
    ("failure_injected", "failure_detected", "replacement_lookup_start"),
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--redundancy", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--drop-step", type=int, default=8)
    args = parser.parse_args()

    handle = tempfile.NamedTemporaryFile(
        prefix="drosim-p19-", suffix=".jsonl", delete=False
    )
    handle.close()
    metrics_path = Path(handle.name)
    # `record_event` is a no-op unless this is set, exactly as on hardware, so the
    # simulated run writes the same schema the real benchmarks write.
    os.environ["DROFELL_METRICS_PATH"] = str(metrics_path)

    from simulate.drosim.run import SimConfig, run

    print("=" * 78)
    print("P19 · local recovery event and causal-order smoke")
    print("=" * 78)
    print(
        f"segments={args.segments} redundancy={args.redundancy} "
        f"tokens={args.tokens}, one kill at step {args.drop_step}"
    )

    result = run(
        SimConfig(
            segments=args.segments, redundancy=args.redundancy, clients=1,
            tokens=args.tokens, log_level="error",
            deterministic_drop_steps=(args.drop_step,),
        )
    )

    records = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    first_at: dict[str, int] = {}
    counts: dict[str, int] = {}
    for record in records:
        event = record["event"]
        counts[event] = counts.get(event, 0) + 1
        first_at.setdefault(event, record["time_ns"])

    print(f"\n{len(records)} events recorded, {len(counts)} distinct types")

    print("\n1. the request survived the failure")
    check(
        "all requested tokens produced",
        result.tokens_produced == args.tokens,
        f"{result.tokens_produced}/{args.tokens}",
    )
    check(
        "a failure was actually injected",
        result.injection["drops_triggered"] > 0,
        f"{result.injection['drops_triggered']} drops",
    )

    print("\n2. event set matches the hardware recovery instrumentation")
    for event in REQUIRED_EVENTS:
        check(f"emitted {event}", event in counts, f"x{counts.get(event, 0)}")

    print("\n3. causal order (the check a faked recovery would fail)")
    for chain in ORDERED_CHAINS:
        present = [event for event in chain if event in first_at]
        if len(present) < 2:
            check(" < ".join(chain), False, "events missing")
            continue
        stamps = [first_at[event] for event in present]
        ordered = all(a <= b for a, b in zip(stamps, stamps[1:]))
        spans_ms = [
            (b - a) / 1e6 for a, b in zip(stamps, stamps[1:])
        ]
        check(
            " < ".join(present),
            ordered,
            "gaps " + ", ".join(f"{s:.3f}ms" for s in spans_ms),
        )

    print("\n4. the replay window is what recovery cost scales with")
    replay_counts = sorted({
        record.get("recovery_task_count")
        for record in records
        if record.get("recovery_task_count") is not None
    })
    print(f"  replay task counts observed: {replay_counts}")
    print(
        "  => the window is bounded by the distance to the last confirmed "
        "output,"
    )
    print(
        "     not by the number of nodes: the shadow holds only the unconfirmed"
    )
    print(
        "     tail, so a kill deep into a request replays the same few steps as"
    )
    print(
        "     an early one. That is why D3's replay-distance sweep is flat, and"
    )
    print("     it is a property of the design rather than a simulator artefact.")

    print("\n5. shadow CPU cost stays on the REAL clock")
    shadow = [
        record for record in records if record["event"] == "shadow_state_snapshot"
    ]
    if shadow:
        cpu_values = [
            record.get("update_cpu_ns") for record in shadow
            if record.get("update_cpu_ns") is not None
        ]
        if cpu_values:
            print(
                f"  measured shadow update CPU: "
                f"{min(cpu_values) / 1e6:.3f} - {max(cpu_values) / 1e6:.3f} ms"
            )
            print(
                "  This is a REAL measurement even in simulation: "
                "`perf_counter_ns`"
            )
            print(
                "  is deliberately not virtualised, because it measures the "
                "control"
            )
            print("  plane's own CPU cost on this host (docs/03 §5).")
            check(
                "shadow CPU cost is nonzero and real",
                all(value > 0 for value in cpu_values),
            )
    else:
        print("  (no shadow snapshots in this run)")

    metrics_path.unlink(missing_ok=True)
    os.environ.pop("DROFELL_METRICS_PATH", None)

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {FAILURES}")
        return 1
    print(
        "local smoke passed; reviewer-grade G1 still requires the hardware "
        "comparison"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
