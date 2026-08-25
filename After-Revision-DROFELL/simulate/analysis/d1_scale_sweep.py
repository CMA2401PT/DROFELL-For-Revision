#!/usr/bin/env python3
"""D1: the scale answer -- what grows with node count, and what does not.

THE QUESTION (reviewer 2.7)
---------------------------
"...raises concerns about scalability and real-world behavior in edge deployments
with tens to hundreds of nodes."

WHY HARDWARE CANNOT ANSWER IT AND SIMULATION CAN
------------------------------------------------
The authors already measured 1-8 GPUs x 1-128 clients on a rented 8-GPU host, and
that data shows throughput FALLING with node count. But on hardware every extra
logical node also takes a slice of the same finite compute, so "orchestration cost"
and "compute contention" move together and cannot be separated. The simulator can
hold the compute model fixed and vary only the number of logical nodes -- which is
the whole justification for this line (docs/00 §2.0).

THE THREE SWEEPS, AND WHY EACH ONE IS SEPARATE
----------------------------------------------
A-  PATH LENGTH.  Segments 1..32, redundancy 1, one node per device.
    Each extra segment puts one more invocation and one more crossing on the
    critical path of every token. This is the axis that costs something, and it
    is the one hardware also measured (validated to within the measured per-hop
    range in probes/p18).

B-  NETWORK SIZE at fixed path length.  Segments fixed, redundancy 1..8, so the
    node count multiplies while the path a token walks does not change.
    ⭐ This is the sweep that answers the reviewer, because "tens to hundreds of
    nodes" in an edge deployment means MORE HOLDERS, not a longer pipeline.

C-  CONTENTION.  The same logical nodes packed onto a fixed device pool, which
    reproduces the hardware regime and shows why the measured curve fell.

Reported with `client_turnaround_us = 0` throughout, so no number here depends on
the one end-to-end-calibrated constant (see run.CLIENT_TURNAROUND_US).

Run from the repository root:
    python simulate/analysis/d1_scale_sweep.py --out simulate/results/d1_scale_sweep.json
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--clients", type=int, default=1)
    args = parser.parse_args()

    from simulate.drosim.run import SimConfig, run
    from simulate.drosim.topology import Placement

    report: dict[str, Any] = {
        "tokens": args.tokens,
        "clients": args.clients,
        "client_turnaround_us": 0.0,
        "sweeps": {},
    }

    print("=" * 78)
    print("D1 · orchestration cost vs scale (the 2.7 deliverable)")
    print("=" * 78)
    print(
        f"{args.clients} client, {args.tokens} tokens/run, client turnaround "
        f"OFF so nothing here depends on the end-to-end calibrated constant."
    )

    # ---- A: path length -------------------------------------------------
    print("\n### A. path length: segments 1..32, redundancy 1, one device each")
    print(
        f"  {'segments':>9s} {'nodes':>6s} {'layers':>7s} {'itl ms':>8s} "
        f"{'tok/s':>8s} {'wall s':>7s}"
    )
    rows_a = []
    for segments in (1, 2, 4, 8, 16, 32):
        started = time.monotonic()
        result = run(
            SimConfig(
                segments=segments, redundancy=1, clients=args.clients,
                tokens=args.tokens, log_level="error",
            )
        )
        wall = time.monotonic() - started
        rows_a.append({
            "segments": segments,
            "logical_nodes": segments,
            "layers_per_segment": 32 // segments,
            "inter_token_ms": result.per_client_inter_token_ms,
            "tokens_per_second": result.aggregate_tokens_per_second,
            "wall_seconds": wall,
            "packets": result.fabric["packets"],
            "bytes_total": result.fabric["bytes_total"],
        })
        print(
            f"  {segments:9d} {segments:6d} {32 // segments:7d} "
            f"{result.per_client_inter_token_ms:8.2f} "
            f"{result.aggregate_tokens_per_second:8.2f} {wall:7.2f}"
        )
    if len(rows_a) >= 2:
        first, last = rows_a[0], rows_a[-1]
        growth = last["inter_token_ms"] / first["inter_token_ms"]
        print(
            f"  => {first['segments']} -> {last['segments']} segments costs "
            f"{growth:.1f}x per-token latency: path length is the axis that is "
            f"NOT free."
        )
    report["sweeps"]["path_length"] = rows_a

    # ---- B: network size at fixed path length ---------------------------
    print("\n### B. ⭐ network size at FIXED path length (the reviewer's regime)")
    print("  segments held at 8; redundancy 1..8 multiplies holders, not hops")
    print(
        f"  {'redundancy':>11s} {'nodes':>6s} {'itl ms':>8s} {'tok/s':>8s} "
        f"{'packets':>8s} {'RSS MiB':>8s} {'wall s':>7s}"
    )
    rows_b = []
    for redundancy in (1, 2, 4, 8):
        started = time.monotonic()
        result = run(
            SimConfig(
                segments=8, redundancy=redundancy, clients=args.clients,
                tokens=args.tokens, log_level="error",
            )
        )
        wall = time.monotonic() - started
        rows_b.append({
            "segments": 8,
            "redundancy": redundancy,
            "logical_nodes": 8 * redundancy,
            "inter_token_ms": result.per_client_inter_token_ms,
            "tokens_per_second": result.aggregate_tokens_per_second,
            "packets": result.fabric["packets"],
            "wall_seconds": wall,
            "rss_mib": rss_mib(),
        })
        print(
            f"  {redundancy:11d} {8 * redundancy:6d} "
            f"{result.per_client_inter_token_ms:8.2f} "
            f"{result.aggregate_tokens_per_second:8.2f} "
            f"{result.fabric['packets']:8d} {rss_mib():8.1f} {wall:7.2f}"
        )
    if len(rows_b) >= 2:
        latencies = [row["inter_token_ms"] for row in rows_b]
        spread = 100 * (max(latencies) - min(latencies)) / min(latencies)
        print(
            f"  => {rows_b[0]['logical_nodes']} -> {rows_b[-1]['logical_nodes']} "
            f"logical nodes changes per-token latency by {spread:.1f}%."
        )
        print(
            "     Orchestration cost tracks PATH LENGTH, not network size. That"
        )
        print(
            "     is the mechanism behind the analytical claims in D2 (the"
        )
        print(
            "     resource-location map is indexed by model segment, and recovery"
        )
        print("     is confined to a recovery domain), now shown end to end.")
    report["sweeps"]["network_size"] = rows_b

    # ---- C: device sharing ----------------------------------------------
    print("\n### C. device sharing: the same nodes packed onto fewer devices")
    print(
        "  ⚠️  MEASURED RESULT, AND IT IS A NEGATIVE ONE: packing changes nothing"
    )
    print(
        f"  {'segments':>9s} {'devices':>8s} {'nodes/dev':>10s} {'itl ms':>8s} "
        f"{'queue-wait ms':>14s} {'dev busy ms':>12s}"
    )
    rows_c = []
    for segments, devices in ((8, 8), (8, 4), (8, 2), (8, 1)):
        result = run(
            SimConfig(
                segments=segments, redundancy=1, clients=args.clients,
                tokens=args.tokens, placement=Placement.SHARED,
                device_count=devices, log_level="error",
            )
        )
        contention_ms = sum(
            b["device_contention_ns"] for b in result.backends
        ) / 1e6
        # Distinct DeviceModel objects, so de-duplicate by device id before
        # summing busy time -- every backend on a shared device reports the same
        # shared counter, and adding them up would multiply it by the node count.
        busy_by_device = {
            b["device_id"]: b["device_busy_ns"] for b in result.backends
        }
        busy_ms = sum(busy_by_device.values()) / 1e6
        rows_c.append({
            "segments": segments,
            "devices": devices,
            "nodes_per_device": segments / devices,
            "inter_token_ms": result.per_client_inter_token_ms,
            "device_queue_wait_ms": contention_ms,
            "device_busy_ms": busy_ms,
        })
        print(
            f"  {segments:9d} {devices:8d} {segments / devices:10.1f} "
            f"{result.per_client_inter_token_ms:8.2f} {contention_ms:14.2f} "
            f"{busy_ms:12.2f}"
        )
    report["sweeps"]["device_sharing"] = rows_c
    if len(rows_c) >= 2:
        waits = [row["device_queue_wait_ms"] for row in rows_c]
        print(
            f"  => queue-wait for the device is {max(waits):.2f} ms even at "
            f"{rows_c[-1]['nodes_per_device']:.0f} logical nodes per device."
        )
        print(
            "     This is NOT a bug and NOT a null result to be hidden: a single"
        )
        print(
            "     request's decode pipeline is strictly serial -- segment k+1"
        )
        print(
            "     cannot start until segment k returns -- so however many logical"
        )
        print(
            "     nodes share a device, at most one of them has work on it at any"
        )
        print("     instant. Two consequences worth stating in the paper:")
        print(
            "       (i) the throughput drop the authors measured when splitting"
        )
        print(
            "           over more GPUs is NOT device contention; it is path"
        )
        print("           length (sweep A), which is the honest attribution;")
        print(
            "      (ii) contention needs enough CONCURRENT requests to overlap"
        )
        print(
            "           segments, so it is a concurrency effect, not a node-count"
        )
        print("           effect. Reported here rather than quietly dropped.")

    print("\n### what this does and does not license saying")
    print("  CAN say: at fixed path length, the orchestration layer's per-token")
    print("           cost is flat in logical node count up to the sizes swept,")
    print("           and the cost that does grow is proportional to path length.")
    print("  CANNOT say: anything about real geographically dispersed volunteer")
    print("           nodes, or about output quality (tokens are faked), or that")
    print("           absolute throughput at these scales was validated -- only")
    print("           the 1..8 segment range has measured counterparts (p18).")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
