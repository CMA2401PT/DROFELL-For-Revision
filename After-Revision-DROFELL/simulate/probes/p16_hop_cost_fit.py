from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

READY = Path("一稿返修/整理后就绪的实验")
SCALE_8 = READY / "推理扩展性(带规模实验)/AutoDL8卡结果/summary.csv"
SCALE_2 = READY / "推理扩展性(带规模实验)/3090节点2卡结果/summary.csv"
# layers per segment for each GPU count, from the network specs
# (network_specs/llama2-7b-infer-scale-Ngpu.json :: model.decoder_group_sizes)
LAYERS_PER_SEGMENT = {1: 32, 2: 16, 4: 8, 8: 4}


def read_matrix(path: Path, deployment: str) -> dict[int, dict[int, float]]:
    table: dict[int, dict[int, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["deployment"] != deployment:
                continue
            table.setdefault(int(row["clients"]), {})[int(row["scale"])] = float(
                row["overall_tokens_per_second"]
            )
    return table


def least_squares(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return (intercept, slope, r2) for y = intercept + slope*x."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = mean_y - slope * mean_x
    sst = sum((y - mean_y) ** 2 for y in ys)
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - sse / sst if sst else float("nan")
    return intercept, slope, r2


def fit_hop_cost(
    matrix: dict[int, dict[int, float]], clients: int
) -> dict[str, Any]:
    row = matrix.get(clients)
    if not row:
        return {}
    scales = sorted(row)
    # aggregate throughput -> per-token latency of one client's stream
    latency_ms = {s: 1000.0 * clients / row[s] for s in scales}
    intercept, slope, r2 = least_squares(
        [float(s) for s in scales], [latency_ms[s] for s in scales]
    )
    return {
        "clients": clients,
        "segments": scales,
        "layers_per_segment": [LAYERS_PER_SEGMENT.get(s) for s in scales],
        "per_token_latency_ms": {str(s): latency_ms[s] for s in scales},
        "fixed_work_ms": intercept,
        "per_hop_ms": slope,
        "r2": r2,
        "predicted_ms": {str(s): intercept + slope * s for s in scales},
        "residual_pct": {
            str(s): 100 * (intercept + slope * s - latency_ms[s]) / latency_ms[s]
            for s in scales
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--deployment", default="combined-request-agent")
    args = parser.parse_args()

    report: dict[str, Any] = {"deployment": args.deployment, "hosts": {}}

    print("=" * 78)
    print("P16 · per-hop cost from the GPU-scale sweep (the segment-count axis)")
    print("=" * 78)

    for label, path in (("autodl-8gpu", SCALE_8), ("rtx3090-2gpu", SCALE_2)):
        if not path.exists():
            print(f"\n[skip] {path} not found")
            continue
        matrix = read_matrix(path, args.deployment)
        if not matrix:
            print(f"\n[skip] {path}: no rows for {args.deployment}")
            continue
        print(f"\n### {label}  ({path})")
        clients_sorted = sorted(matrix)
        scales = sorted({s for row in matrix.values() for s in row})
        print("  measured aggregate throughput (tok/s)")
        print("  " + f"{'clients':>8s}" + "".join(f"{'L=' + str(s):>10s}" for s in scales))
        for clients in clients_sorted:
            cells = "".join(
                f"{matrix[clients].get(s, float('nan')):10.1f}" for s in scales
            )
            print(f"  {clients:8d}{cells}")

        fits = {}
        print("\n  T(L) = W + h*L fitted per client count")
        print(
            "  " + f"{'clients':>8s} {'W (ms)':>9s} {'h (ms/hop)':>11s} "
            f"{'r2':>7s} {'worst resid':>12s}"
        )
        for clients in clients_sorted:
            fit = fit_hop_cost(matrix, clients)
            if not fit:
                continue
            fits[str(clients)] = fit
            worst = max(abs(v) for v in fit["residual_pct"].values())
            print(
                f"  {clients:8d} {fit['fixed_work_ms']:9.2f} "
                f"{fit['per_hop_ms']:11.3f} {fit['r2']:7.4f} {worst:11.1f}%"
            )
        report["hosts"][label] = {"source": str(path), "fits": fits}

        # the low-concurrency fits are the ones that isolate the hop cost:
        # at high concurrency the pipeline overlaps hops and the linear form
        # stops describing the same quantity.
        clean = [fits[str(c)] for c in (1, 2, 4) if str(c) in fits]
        if clean:
            hop = sum(f["per_hop_ms"] for f in clean) / len(clean)
            work = sum(f["fixed_work_ms"] for f in clean) / len(clean)
            worst_r2 = min(f["r2"] for f in clean)
            print(
                f"\n  => serial regime (clients 1-4): h = {hop:.3f} ms/hop, "
                f"W = {work:.2f} ms, worst r2 = {worst_r2:.4f}"
            )
            print(
                f"     per-hop share of the token: {100 * hop / (work + hop):.0f}% at "
                f"L=1 -> {100 * hop * 8 / (work + 8 * hop):.0f}% at L=8"
            )
            report["hosts"][label]["serial_regime"] = {
                "clients_used": [c for c in (1, 2, 4) if str(c) in fits],
                "per_hop_ms": hop,
                "fixed_work_ms": work,
                "worst_r2": worst_r2,
            }

    print("\n### reading")
    print("  * h is measured, not assumed: it is the cost DROSIM must charge per")
    print("    extra logical node on the token critical path, and it is what")
    print("    makes the node-count axis extrapolatable at all (docs/10 §1.1b).")
    print("  * W is L-independent by construction (same 32 decoders throughout),")
    print("    so a good r2 is evidence the decomposition is real rather than a")
    print("    two-parameter curve fitted to four points.")
    print("  * the two hosts are fitted SEPARATELY and never compared in absolute")
    print("    terms: autodl is unshaped loopback, 3090 is tc-shaped (docs/04).")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
