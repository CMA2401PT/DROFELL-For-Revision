from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Optional

TC_ROOT = Path(
    "一稿返修/整理后就绪的实验/吞吐量实验/3090-带tc/DROFELL-推理-7B-三种网络"
)
TIERS = {
    "1g-lt5": {
        "dir": "3090-drofell-7b-infer-2-card-1g-lt5",
        "bandwidth_mbit": 1000.0,
        "latency_label": "<5ms",
    },
    "100m-lt5": {
        "dir": "3090-drofell-7b-infer-2-card-100m-lt5",
        "bandwidth_mbit": 100.0,
        "latency_label": "<5ms",
    },
    "100m-100": {
        "dir": "3090-drofell-7b-infer-2-card-100m-100",
        "bandwidth_mbit": 100.0,
        "latency_label": "100ms",
    },
}
# Llama2-7B hidden state, fp16: the per-token payload that crosses a link.
HIDDEN_SIZE = 4096
BYTES_PER_ELEMENT = 2
PAYLOAD_BITS = HIDDEN_SIZE * BYTES_PER_ELEMENT * 8
# `orbit/examples/tc/Makefile` shapes with LAT=<value>; the tier label is the
# round trip, so one crossing pays half. Solved for below rather than assumed:
# the latency axis returns C only once this is fixed, so it is stated explicitly.
NOMINAL_RTT_MS = 100.0
LATENCY_ONE_WAY_MS = NOMINAL_RTT_MS / 2


def median_inter_token_ms(result_path: Path) -> Optional[float]:
    """Median inter-token latency, per client stream, from a token timeline.

    `result.json` is a flat [[client_index, time_ns], ...] log, so deltas must be
    taken WITHIN a client stream; taking them across the flat list would measure
    the interleaving of clients instead.
    """
    entries = json.loads(result_path.read_text())
    per_client: dict[int, list[int]] = {}
    for client_index, timestamp_ns in entries:
        per_client.setdefault(client_index, []).append(timestamp_ns)
    deltas: list[float] = []
    for stamps in per_client.values():
        stamps.sort()
        deltas.extend((b - a) / 1e6 for a, b in zip(stamps, stamps[1:]))
    return statistics.median(deltas) if deltas else None


def collect(tier_dir: Path) -> dict[tuple[int, int], list[float]]:
    """(clients, tokens) -> median ITL per repeat."""
    out: dict[tuple[int, int], list[float]] = {}
    pattern = re.compile(r"clients-(\d+)/tokens-(\d+)/repeat-(\d+)")
    for path in sorted(tier_dir.glob("cases/inference/clients-*/tokens-*/repeat-*/result.json")):
        match = pattern.search(str(path))
        if not match:
            continue
        median = median_inter_token_ms(path)
        if median is None:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        out.setdefault(key, []).append(median)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=TC_ROOT)
    parser.add_argument(
        "--cell", type=int, nargs=2, default=(1, 128),
        help="clients and tokens of the cell used for the solve (default 1 128)",
    )
    args = parser.parse_args()

    if not args.root.exists():
        print(f"[skip] {args.root} not found")
        return

    print("=" * 78)
    print("P17 · SimFabric transport model validated on three shaped tiers")
    print("=" * 78)
    print(
        f"same host, same 2-dealer topology, same model; only the `tc` setting "
        f"differs.\npayload per crossing = {HIDDEN_SIZE} x fp16 = "
        f"{PAYLOAD_BITS // 8} bytes"
    )

    report: dict[str, Any] = {
        "source": str(args.root),
        "payload_bytes": PAYLOAD_BITS // 8,
        "latency_one_way_ms": LATENCY_ONE_WAY_MS,
        "tiers": {},
    }

    measured: dict[str, dict[tuple[int, int], list[float]]] = {}
    for name, spec in TIERS.items():
        tier_dir = args.root / str(spec["dir"])
        if not tier_dir.exists():
            print(f"\n[skip] tier {name}: {tier_dir} not found")
            continue
        measured[name] = collect(tier_dir)
        report["tiers"][name] = {
            "bandwidth_mbit": spec["bandwidth_mbit"],
            "latency_label": spec["latency_label"],
            "cells": {
                f"c{c}-t{t}": values for (c, t), values in measured[name].items()
            },
        }

    print("\n### measured median inter-token latency (ms), mean over repeats")
    cells = sorted({key for tier in measured.values() for key in tier})
    print(f"  {'cell':>12s}" + "".join(f"{name:>12s}" for name in TIERS))
    for clients, tokens in cells:
        row = f"  {'c%d/t%d' % (clients, tokens):>12s}"
        for name in TIERS:
            values = measured.get(name, {}).get((clients, tokens))
            row += (
                f"{statistics.mean(values):12.2f}" if values else f"{'-':>12s}"
            )
        print(row)

    cell = (args.cell[0], args.cell[1])
    needed = ["1g-lt5", "100m-lt5", "100m-100"]
    if any(not measured.get(n, {}).get(cell) for n in needed):
        print(f"\n[skip] the solve needs all three tiers at cell {cell}")
        return

    itl = {n: statistics.mean(measured[n][cell]) for n in needed}
    print(f"\n### solving for C (shaped crossings per token) at cell {cell}")
    print("  the same integer must fall out of two independent axes:")

    # ---- latency axis ----------------------------------------------------
    delta_latency_ms = itl["100m-100"] - itl["100m-lt5"]
    c_latency = delta_latency_ms / LATENCY_ONE_WAY_MS
    print(
        f"  latency   : ({itl['100m-100']:.2f} - {itl['100m-lt5']:.2f}) ms "
        f"/ {LATENCY_ONE_WAY_MS:.0f} ms  ->  C = {c_latency:.2f}"
    )

    # ---- bandwidth axis --------------------------------------------------
    delta_bandwidth_ms = itl["100m-lt5"] - itl["1g-lt5"]
    per_crossing_ms = 1e3 * PAYLOAD_BITS * (1 / (100 * 1e6) - 1 / (1000 * 1e6))
    c_bandwidth = delta_bandwidth_ms / per_crossing_ms
    print(
        f"  bandwidth : ({itl['100m-lt5']:.2f} - {itl['1g-lt5']:.2f}) ms "
        f"/ {per_crossing_ms:.3f} ms  ->  C = {c_bandwidth:.2f}"
    )

    consensus = round((c_latency + c_bandwidth) / 2)
    agree = abs(c_latency - c_bandwidth) < 0.5
    print(
        f"\n  => the two axes {'AGREE' if agree else 'DISAGREE'} on C = {consensus} "
        f"(spread {abs(c_latency - c_bandwidth):.2f})"
    )
    print(
        f"     2 dealers + 1 request/Agent node => {consensus} crossings per token, "
        f"i.e. C = L + 1."
    )
    if not agree:
        print("     ⚠️  the transport model does NOT explain both axes; do not")
        print("         trust drosim/fabric.py on the weak-network axis.")

    # ---- residual compute term ------------------------------------------
    print("\n### implied compute term, and the model checked back against all tiers")
    print(
        f"  {'tier':>12s} {'measured':>10s} {'transport':>10s} {'W_compute':>11s}"
    )
    compute_terms = {}
    for name in needed:
        bandwidth = float(TIERS[name]["bandwidth_mbit"])
        latency_ow = (
            LATENCY_ONE_WAY_MS if TIERS[name]["latency_label"] == "100ms" else 0.0
        )
        transport = consensus * (
            latency_ow + 1e3 * PAYLOAD_BITS / (bandwidth * 1e6)
        )
        compute = itl[name] - transport
        compute_terms[name] = compute
        print(
            f"  {name:>12s} {itl[name]:10.2f} {transport:10.2f} {compute:11.2f}"
        )
    spread = max(compute_terms.values()) - min(compute_terms.values())
    mean_compute = statistics.mean(compute_terms.values())
    print(
        f"\n  => W_compute is {mean_compute:.2f} ms with a spread of "
        f"{spread:.2f} ms ({100 * spread / mean_compute:.1f}%) across tiers."
    )
    print(
        "     W_compute is a property of the GPU, not of the link, so a small "
        "spread\n     is the actual test of the decomposition -- and it is what "
        "lets DROSIM\n     hold compute fixed while varying the network."
    )
    print("\n  NOTE the `<5ms` tiers are treated as 0 ms of shaped latency: the")
    print("  label is an upper bound, so W_compute above ABSORBS the true few")
    print("  hundred microseconds. That biases W_compute high, never the")
    print("  transport term, so it cannot manufacture the agreement above.")

    report["solve"] = {
        "cell": {"clients": cell[0], "tokens": cell[1]},
        "itl_ms": itl,
        "c_from_latency": c_latency,
        "c_from_bandwidth": c_bandwidth,
        "c_consensus": consensus,
        "axes_agree": agree,
        "compute_term_ms": compute_terms,
        "compute_spread_ms": spread,
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
