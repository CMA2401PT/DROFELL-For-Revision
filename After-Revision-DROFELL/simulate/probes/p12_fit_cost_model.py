from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path
from typing import Any

READY = Path("一稿返修/整理后就绪的实验/推理整流")
CASES = [
    "c0-m0-d0", "c0-m0-d1", "c0-m1-d0", "c0-m1-d1",
    "c1-m0-d0", "c1-m0-d1", "c1-m1-d0", "c1-m1-d1",
]
# service time = submission + async gap, summed (see finding 1)
SERVICE_PAIRS = {
    "decoder_generating": (
        "exec_decoder->done_decoder", "done_decoder->async_decoder"),
    "infer_head": ("exec_head->done_head", "done_head->async_head"),
    "infer_tail": ("exec_tail->done_tail", "done_tail->async_tail"),
}
NOT_MODELLED = {
    "wait_decoder->exec_decoder": "mailbox queueing (simulator produces it)",
    "wait_head->exec_head": "mailbox queueing (simulator produces it)",
    "wait_tail->exec_tail": "mailbox queueing (simulator produces it)",
    "leave->dealer-recv": "transport (SimFabric)",
    "leave->root": "transport (SimFabric)",
}


def stage_medians(trace_path: Path) -> dict[str, float]:
    traces = json.loads(trace_path.read_text())
    segments: dict[str, list[float]] = collections.defaultdict(list)
    for client in traces:
        for token in client:
            for (name_a, t_a), (name_b, t_b) in zip(token, token[1:]):
                segments[f"{name_a}->{name_b}"].append((t_b - t_a) / 1e3)
    return {k: statistics.median(v) for k, v in segments.items()}


def mean_batches(batch_csv: Path) -> dict[str, dict[str, float]]:
    acc: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    with batch_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            acc[(row["case"], row["operation"])].append(float(row["mean_size"]))
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for (case, op), values in acc.items():
        out[case][op] = statistics.mean(values)
    return out


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return mean_y, 0.0, float("nan")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return intercept, slope, r2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=READY)
    parser.add_argument("--repeat", default="1")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    batches = mean_batches(args.root / "batch_summary.csv")
    throughput: dict[str, list[float]] = collections.defaultdict(list)
    with (args.root / "summary.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            throughput[row["case"]].append(float(row["observed_tok_s"]))

    medians: dict[str, dict[str, float]] = {}
    for case in CASES:
        trace = args.root / case / f"repeat-{args.repeat}" / "traces.json"
        if trace.exists():
            medians[case] = stage_medians(trace)
    if not medians:
        raise SystemExit(f"no traces under {args.root}")

    report: dict[str, Any] = {"repeat": args.repeat, "service_terms": {}}

    # --- finding 1: the split is unstable, the sum is not -------------------
    print("### finding 1: exec->done / done->async split is NOT stable")
    print(f"{'case':10s} {'dual-q':>7s} {'exec->done':>11s} {'done->async':>12s} "
          f"{'SUM':>9s}")
    for case in CASES:
        if case not in medians:
            continue
        submit = medians[case].get("exec_decoder->done_decoder", float("nan"))
        gap = medians[case].get("done_decoder->async_decoder", float("nan"))
        print(f"{case:10s} {'on' if case.endswith('d1') else 'off':>7s} "
              f"{submit:11.1f} {gap:12.1f} {submit + gap:9.1f}")
    print("=> the boundary moves with the dual-queue flag; model the SUM.\n")

    # --- finding 2: service time is flat in batch size ---------------------
    print("### finding 2: service time vs merged batch size")
    for op, (stage_a, stage_b) in SERVICE_PAIRS.items():
        xs: list[float] = []
        ys: list[float] = []
        for case in CASES:
            if case not in medians:
                continue
            size = batches.get(case, {}).get(op)
            if size is None or stage_a not in medians[case]:
                continue
            xs.append(size)
            ys.append(medians[case][stage_a] + medians[case].get(stage_b, 0.0))
        if len(xs) < 2:
            continue
        alpha, beta, r2 = linear_fit(xs, ys)
        ratio = abs(beta) * max(xs) / alpha if alpha else float("nan")
        print(f"  {op:20s} alpha={alpha:8.1f} us  beta={beta:7.1f} us/item  "
              f"|beta|*nmax/alpha={ratio:5.1%}  batch {min(xs):.1f}-{max(xs):.1f}")
        report["service_terms"][op] = {
            "alpha_us": alpha, "beta_us_per_item": beta, "r2": r2,
            "batch_range": [min(xs), max(xs)], "points": len(xs),
        }
    print("=> beta is negligible against alpha: p(n) ~= const over batch 1..8,")
    print("   the strongest form of the paper's §4.2 sub-additivity assumption.\n")

    # --- finding 3: does that predict measured throughput? -----------------
    print("### finding 3: predict measured throughput from (batch / service)")
    print(f"{'case':10s} {'batch':>6s} {'service us':>11s} {'batch/svc':>11s} "
          f"{'measured':>10s}")
    xs = []
    ys = []
    for case in CASES:
        if case not in medians or case not in throughput:
            continue
        size = batches[case]["decoder_generating"]
        service = (medians[case]["exec_decoder->done_decoder"]
                   + medians[case]["done_decoder->async_decoder"])
        rate = size / (service / 1e6)
        measured = statistics.mean(throughput[case])
        xs.append(rate)
        ys.append(measured)
        print(f"{case:10s} {size:6.2f} {service:11.1f} {rate:11.1f} "
              f"{measured:10.1f}")
    intercept, slope, r2 = linear_fit(xs, ys)
    print(f"\nmeasured = {slope:.3f} * (batch/service) + {intercept:.1f}"
          f"   r2 = {r2:.3f}   n = {len(xs)}")
    print(f"=> ONE utilisation constant eta={slope:.3f} reproduces all "
          f"{len(xs)} cases spanning "
          f"{max(ys) / min(ys):.1f}x throughput.")
    report["throughput_model"] = {
        "eta": slope, "intercept_tok_s": intercept, "r2": r2, "points": len(xs),
    }

    print("\n### not modelled by the cost model (by design)")
    first = next(iter(medians.values()))
    for stage, why in NOT_MODELLED.items():
        if stage in first:
            print(f"  {stage:30s} {first[stage]:8.1f} us   <- {why}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
