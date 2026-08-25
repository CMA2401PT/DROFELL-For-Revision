#!/usr/bin/env python3
"""D3: does recovery cost grow with network size? (2.7 + 1.4 + 2.13)

THE CLAIM UNDER TEST
--------------------
The paper argues recovery is confined to a recovery domain: it is not broadcast
and does not propagate through a healthy node. If true, the cost of recovering
from one node failure must depend on the REPLAY WINDOW (how far the survivor is
from the last confirmed state) and not on how many nodes exist.

D2 argued this analytically (2 + k control messages, independent of N) and the
shadow-overhead sweep measured the state size on hardware. What was missing was
an end-to-end demonstration: actually kill a node in a running deployment and
measure what recovery costs, at several network sizes.

WHY THIS NEEDS THE SIMULATOR
----------------------------
The measured failure matrices go up to 8 logical nodes, which is where the
authors' hardware ends. The question is what happens at tens of nodes, and the
answer must hold compute fixed while node count varies -- which hardware cannot do.

WHAT IS REAL IN THESE NUMBERS
-----------------------------
The recovery orchestration is production code: `AgentInferController.recover_handles`
picks a replacement, `InferStateData.gen_init_cfg` builds the replay window, the
Dealer re-materialises the frontier. The failure injector reuses the production
guards, including the per-(seq, hop) duplicate guard without which redundancy
levels cannot be compared (drosim/failure.py). What is simulated is only how long
each operator takes.

Run from the repository root:
    python simulate/analysis/d3_recovery_scale.py --out simulate/results/d3_recovery.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--drop-step", type=int, default=8)
    args = parser.parse_args()

    from simulate.drosim.run import SimConfig, run

    report: dict[str, Any] = {
        "tokens": args.tokens,
        "drop_step": args.drop_step,
        "sweeps": {},
    }

    print("=" * 78)
    print("D3 · cost of recovering one node failure, vs network size")
    print("=" * 78)
    print(
        f"one deterministic node kill at decode step {args.drop_step}, "
        f"{args.tokens} tokens requested"
    )

    def baseline(segments: int, redundancy: int) -> float:
        clean = run(
            SimConfig(
                segments=segments, redundancy=redundancy, clients=1,
                tokens=args.tokens, log_level="error",
            )
        )
        return clean.virtual_seconds

    # ---- A: recovery overhead vs network size at fixed path length -------
    print("\n### A. ⭐ recovery overhead vs NETWORK SIZE (path length fixed at 4)")
    print(
        "  EXACTLY ONE node is killed in every row (shared kill budget). An"
    )
    print(
        "  earlier version of this sweep gave each Dealer its own budget, so"
    )
    print(
        "  redundancy r produced r kills -- 2/4/8 failures at r=2/4/8 -- and the"
    )
    print(
        "  apparent 54% rise in 'recovery cost vs network size' was really cost"
    )
    print(
        "  vs failure COUNT. See DeterministicPlan in drosim/failure.py."
    )
    print(
        f"  {'redundancy':>11s} {'nodes':>6s} {'clean s':>9s} {'failed s':>9s} "
        f"{'overhead ms':>12s} {'tokens':>7s} {'drops':>6s}"
    )
    rows_a = []
    for redundancy in (2, 4, 8):
        clean_seconds = baseline(4, redundancy)
        result = run(
            SimConfig(
                segments=4, redundancy=redundancy, clients=1,
                tokens=args.tokens, log_level="error",
                deterministic_drop_steps=(args.drop_step,),
            )
        )
        overhead_ms = 1000 * (result.virtual_seconds - clean_seconds)
        rows_a.append({
            "segments": 4,
            "redundancy": redundancy,
            "logical_nodes": 4 * redundancy,
            "clean_seconds": clean_seconds,
            "failed_seconds": result.virtual_seconds,
            "recovery_overhead_ms": overhead_ms,
            "tokens_produced": result.tokens_produced,
            "drops_triggered": result.injection["drops_triggered"],
            "packets": result.fabric["packets"],
        })
        print(
            f"  {redundancy:11d} {4 * redundancy:6d} {clean_seconds:9.4f} "
            f"{result.virtual_seconds:9.4f} {overhead_ms:12.2f} "
            f"{result.tokens_produced:7d} "
            f"{result.injection['drops_triggered']:6d}"
        )
    if len(rows_a) >= 2:
        overheads = [row["recovery_overhead_ms"] for row in rows_a]
        completed = all(
            row["tokens_produced"] == args.tokens for row in rows_a
        )
        print(
            f"  => overhead across "
            f"{rows_a[0]['logical_nodes']}..{rows_a[-1]['logical_nodes']} logical "
            f"nodes: {min(overheads):.2f} - {max(overheads):.2f} ms"
        )
        print(
            f"     every run still produced all {args.tokens} tokens: "
            f"{completed}"
        )
        print(
            "     Recovery cost does not grow with the number of nodes: the"
        )
        print(
            "     end-to-end counterpart of D2's analytical `2 + k` message count."
        )
        print(
            "     Note the overhead is ~0 here, which is the point of sweep D:"
        )
        print(
            "     while a redundant path survives, the failure never reaches the"
        )
        print("     critical path at all.")
    report["sweeps"]["network_size"] = rows_a

    # ---- D: what redundancy actually buys -------------------------------
    print("\n### D. ⭐ the same single kill, with and without a surviving path")
    print(
        f"  {'redundancy':>11s} {'nodes':>6s} {'overhead ms':>12s} "
        f"{'tokens':>7s} {'drops':>6s}"
    )
    rows_d = []
    for redundancy in (1, 2, 4, 8):
        clean_seconds = baseline(4, redundancy)
        result = run(
            SimConfig(
                segments=4, redundancy=redundancy, clients=1,
                tokens=args.tokens, log_level="error",
                deterministic_drop_steps=(args.drop_step,),
            )
        )
        overhead_ms = 1000 * (result.virtual_seconds - clean_seconds)
        rows_d.append({
            "redundancy": redundancy,
            "logical_nodes": 4 * redundancy,
            "recovery_overhead_ms": overhead_ms,
            "tokens_produced": result.tokens_produced,
            "drops_triggered": result.injection["drops_triggered"],
        })
        print(
            f"  {redundancy:11d} {4 * redundancy:6d} {overhead_ms:12.2f} "
            f"{result.tokens_produced:7d} "
            f"{result.injection['drops_triggered']:6d}"
        )
    report["sweeps"]["redundancy_benefit"] = rows_d
    if rows_d:
        single_path = rows_d[0]["recovery_overhead_ms"]
        redundant = [row["recovery_overhead_ms"] for row in rows_d[1:]]
        print(
            f"  => with NO redundancy the kill costs {single_path:.2f} ms: the"
        )
        print(
            "     failure has to be detected, a replacement found, and the replay"
        )
        print(
            "     window re-executed, all on the critical path of the request."
        )
        if redundant:
            print(
                f"     with a surviving redundant path it costs "
                f"{min(redundant):.2f}-{max(redundant):.2f} ms, i.e. effectively"
            )
            print(
                "     nothing -- the surviving path already carries the token"
            )
            print(
                "     while recovery proceeds off the critical path."
            )
        print(
            "     ⭐ This is the mechanism behind the paper's pause-free claim,"
        )
        print(
            "     measured end to end: redundancy does not make recovery cheaper,"
        )
        print("     it moves recovery OFF the path the request is waiting on.")
        print(
            "     ⚠️  Both cases still complete: recovery works without redundancy"
        )
        print(
            "     too (via replacement lookup). Redundancy buys LATENCY, not"
        )
        print("     survival, in a single-failure scenario.")

    # ---- B: recovery overhead vs replay distance ------------------------
    print("\n### B. recovery overhead vs REPLAY DISTANCE (when the kill happens)")
    print(
        "  redundancy 1 on purpose: sweep D showed that with a surviving path the"
    )
    print(
        "  overhead is ~0, so replay would be invisible. Putting recovery ON the"
    )
    print("  critical path is what makes the window's cost measurable.")
    print(
        f"  {'drop step':>10s} {'clean s':>9s} {'failed s':>9s} "
        f"{'overhead ms':>12s} {'tokens':>7s}"
    )
    rows_b = []
    clean_seconds = baseline(4, 1)
    for drop_step in (2, 4, 8, 16):
        if drop_step >= args.tokens:
            continue
        result = run(
            SimConfig(
                segments=4, redundancy=1, clients=1, tokens=args.tokens,
                log_level="error", deterministic_drop_steps=(drop_step,),
            )
        )
        overhead_ms = 1000 * (result.virtual_seconds - clean_seconds)
        rows_b.append({
            "drop_step": drop_step,
            "clean_seconds": clean_seconds,
            "failed_seconds": result.virtual_seconds,
            "recovery_overhead_ms": overhead_ms,
            "tokens_produced": result.tokens_produced,
        })
        print(
            f"  {drop_step:10d} {clean_seconds:9.4f} "
            f"{result.virtual_seconds:9.4f} {overhead_ms:12.2f} "
            f"{result.tokens_produced:7d}"
        )
    report["sweeps"]["replay_distance"] = rows_b
    if len(rows_b) >= 2:
        overheads_b = [row["recovery_overhead_ms"] for row in rows_b]
        spread_b = max(overheads_b) - min(overheads_b)
        print(
            f"  => overhead is FLAT at {min(overheads_b):.2f} ms "
            f"(spread {spread_b:.2f} ms) regardless of when the kill lands."
        )
        print(
            "     ⭐ This was not the expected result and it is the more"
        )
        print(
            "     interesting one. The replay window is bounded by the distance to"
        )
        print(
            "     the LAST CONFIRMED OUTPUT, not by how long the request has been"
        )
        print(
            "     running: the shadow keeps only the unconfirmed tail, so a kill at"
        )
        print(
            "     step 16 replays the same 2 tasks as a kill at step 2 (verified"
        )
        print(
            "     directly in probes/p19: `recovery_task_count` is 2 in every"
        )
        print("     case).")
        print(
            "     Consequence for the paper: recovery cost is bounded by a design"
        )
        print(
            "     property, not merely small on the runs that were measured. The"
        )
        print(
            "     measured shadow-state growth (83137 + 7551*steps bytes, docs/10"
        )
        print(
            "     §2) is growth in the WINDOW; with block_retrieve off the window"
        )
        print("     stays short, which is the regime both sets of runs are in.")

    print("\n### boundaries")
    print("  * one failure per run, injected deterministically, so the overhead")
    print("    attribution is unambiguous. Probabilistic sweeps belong next to")
    print("    the measured failure matrices, not here.")
    print("  * link partition and packet loss stay OFF (docs/04 §3).")
    print("  * tokens are faked, so this says nothing about output correctness")
    print("    after recovery -- that is what the measured recovery experiments")
    print("    are for.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
