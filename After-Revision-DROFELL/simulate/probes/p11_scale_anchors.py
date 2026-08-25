from __future__ import annotations

import argparse
import csv
from pathlib import Path

READY = Path("一稿返修/整理后就绪的实验")
UNRELIABLE = READY / "不可靠节点(带规模实验)/AutoDL8卡结果/DRODELL-不可靠节点/summary.csv"
SCALE_8 = READY / "推理扩展性(带规模实验)/AutoDL8卡结果/summary.csv"
DEPLOYMENT = "combined-request-agent"


def spread_pct(low: str, high: str, mean: str) -> float:
    try:
        low_f, high_f, mean_f = float(low), float(high), float(mean)
    except (TypeError, ValueError):
        return float("nan")
    return 100 * (high_f - low_f) / mean_f if mean_f else 0.0


def show_unreliable(path: Path) -> None:
    if not path.exists():
        print(f"[skip] {path} not found")
        return
    print("=== unreliable_node (8-GPU host): topology x failure rate ===")
    header = (
        f"{'topology':9s} {'p':>7s} {'rep':>3s} {'tp128':>9s} {'spread%':>8s} "
        f"{'tp1024':>9s} {'spread%':>8s}"
    )
    print(header)
    worst = 0.0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["deployment"] != DEPLOYMENT:
                continue
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
    print(f"\nworst repeat-to-repeat spread across cells: {worst:.1f}%")
    print(
        "=> simulation accuracy demands must be read against this spread, "
        "not against an absolute target (docs/03 §3)."
    )


def show_scale(path: Path) -> None:
    if not path.exists():
        print(f"[skip] {path} not found")
        return
    print("\n=== infer_scale (8-GPU host): GPU scale x clients -> tok/s ===")
    table: dict[str, dict[str, float]] = {}
    clients_seen: list[str] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["deployment"] != DEPLOYMENT:
                continue
            table.setdefault(row["scale"], {})[row["clients"]] = float(
                row["overall_tokens_per_second"]
            )
            if row["clients"] not in clients_seen:
                clients_seen.append(row["clients"])
    clients_sorted = sorted(clients_seen, key=int)
    print("scale " + "".join(f"{'c' + c:>10s}" for c in clients_sorted))
    for scale in sorted(table, key=int):
        cells = "".join(
            f"{table[scale].get(c, float('nan')):10.1f}" for c in clients_sorted
        )
        print(f"{scale:>5s} {cells}")
    print(
        "\n=> real data already spans 1..8 GPUs and 1..128 clients; simulation only "
        "has to extrapolate the logical-node axis beyond it."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unreliable", type=Path, default=UNRELIABLE)
    parser.add_argument("--scale", type=Path, default=SCALE_8)
    args = parser.parse_args()
    show_unreliable(args.unreliable)
    show_scale(args.scale)


if __name__ == "__main__":
    main()
