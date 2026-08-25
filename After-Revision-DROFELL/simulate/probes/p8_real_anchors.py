from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_SUMMARY = (
    "一稿返修/整理后就绪的实验/不可靠节点(带规模实验)/3090节点2卡结果"
    "/DROFELL-不可靠节点/final/results/3090-final-strict-r8-20260818/summary.csv"
)


def spread_pct(low: str, high: str, mean: str) -> float:
    try:
        low_f, high_f, mean_f = float(low), float(high), float(mean)
    except (TypeError, ValueError):
        return float("nan")
    return 100 * (high_f - low_f) / mean_f if mean_f else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", default=DEFAULT_SUMMARY, type=Path)
    parser.add_argument("--deployment", default="combined-request-agent")
    args = parser.parse_args()

    if not args.summary.exists():
        raise SystemExit(f"summary not found: {args.summary}")

    print(f"source: {args.summary}")
    print(f"deployment={args.deployment}")
    header = (
        f"{'topology':9s} {'p':>7s} {'rep':>3s} {'tp128':>9s} {'spread%':>8s} "
        f"{'tp1024':>9s} {'spread%':>8s}"
    )
    print(header)
    worst = 0.0
    cells = 0
    with args.summary.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("deployment") != args.deployment:
                continue
            cells += 1
            s128 = spread_pct(
                row["throughput_128_min"],
                row["throughput_128_max"],
                row["throughput_128_mean"],
            )
            s1024 = spread_pct(
                row["throughput_1024_min"],
                row["throughput_1024_max"],
                row["throughput_1024_mean"],
            )
            worst = max(worst, s128, s1024)
            print(
                f"{row['topology']:9s} {row['probability']:>7s} "
                f"{row['repeat_count']:>3s} {float(row['throughput_128_mean']):9.2f} "
                f"{s128:8.1f} {float(row['throughput_1024_mean']):9.2f} {s1024:8.1f}"
            )

    print(f"\ncells={cells}  worst repeat-to-repeat spread: {worst:.1f}%")
    print(
        "=> calibration verdict should be 'does the simulated value fall inside the\n"
        "   measured min-max interval', not a fixed error percentage (docs/03 §3.2)."
    )


if __name__ == "__main__":
    main()
