#!/usr/bin/env python3
"""P14: validate the shadow-memory model against the measured sweep.

Source: 一稿返修/整理后就绪的实验/影子节点开销/ (7B/70B x shard {8,16,20,32,40} x window)

This is the G4 gate in docs/03 §3.2. Shadow footprint and shadow CPU time are
computed by the SAME Python code in simulation as on hardware, so a mismatch
means the simulator is wired up wrong -- it must NOT be excused as model error.

Findings recorded in docs/06 P14:
  * every snapshot has checkpoint_block_count == 0 (block_retrieve off), so the
    replay window is the entire footprint in this sweep
  * generating_bytes = 83137 + 7551 * replay_task_count, r2 = 0.315
    -> the per-step constant is measured; the scatter is real and reported
  * shadow CPU time stays around 1 ms per recovery regardless of model size

Run from the repository root.
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics
from pathlib import Path

DEFAULT = Path(
    "一稿返修/整理后就绪的实验/影子节点开销"
    "/paper-shadow-model-shard-window-20260816/snapshots.csv"
)


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return mean_y, 0.0, float("nan")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return intercept, slope, (1 - ss_res / ss_tot if ss_tot else float("nan"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", nargs="?", default=DEFAULT, type=Path)
    args = parser.parse_args()

    if not args.snapshots.exists():
        raise SystemExit(f"not found: {args.snapshots}")
    rows = list(csv.DictReader(args.snapshots.open(newline="")))
    print(f"source: {args.snapshots}")
    print(f"snapshots: {len(rows)}")
    print()

    print("### per-case medians")
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = (
        collections.defaultdict(list)
    )
    for row in rows:
        grouped[(row["case"], row["model_layers"], row["shard_layers"])].append(row)
    print(f"{'case':12s} {'layers':>6s} {'shard':>5s} {'n':>3s} {'meta KiB':>9s} "
          f"{'replay KiB':>11s} {'cpu ms':>7s} {'steps':>6s}")
    for key, group in sorted(grouped.items()):
        def med(field: str) -> float:
            return statistics.median(float(item[field]) for item in group)
        print(f"{key[0]:12s} {key[1]:>6s} {key[2]:>5s} {len(group):3d} "
              f"{med('metadata_bytes') / 1024:9.1f} "
              f"{med('generating_bytes') / 1024:11.1f} "
              f"{med('update_cpu_ns') / 1e6:7.2f} "
              f"{med('replay_task_count'):6.0f}")
    print()

    print("### model: generating_bytes = base + per_step * replay_task_count")
    xs = [float(row["replay_task_count"]) for row in rows]
    ys = [float(row["generating_bytes"]) for row in rows]
    base, per_step, r2 = linear_fit(xs, ys)
    print(f"  base = {base:.0f} bytes,  per_step = {per_step:.0f} bytes,  "
          f"r2 = {r2:.3f},  n = {len(xs)}")
    print("  => per-step constant is MEASURED; the modest r2 is reported, not hidden")
    print("     (per-step payload varies with prompt and decode content).")
    print()

    print("### mode check: was KV checkpointing active in this sweep?")
    blocks = collections.Counter(row["checkpoint_block_count"] for row in rows)
    kv_values = sorted({row["kv_bytes"] for row in rows})
    print(f"  checkpoint_block_count: {dict(blocks)}")
    print(f"  kv_bytes values: {kv_values}")
    if set(blocks) == {"0"}:
        print("  => block_retrieve OFF in every snapshot: the replay window is the")
        print("     entire shadow footprint here. KV-checkpoint mode is a separate")
        print("     regime and must be reported separately.")
    print()

    print("### shadow CPU cost (this is a REAL measurement even in simulation)")
    cpu_ms = sorted(float(row["update_cpu_ns"]) / 1e6 for row in rows)
    print(f"  p50 = {statistics.median(cpu_ms):.2f} ms   "
          f"p90 = {cpu_ms[int(0.9 * len(cpu_ms))]:.2f} ms   "
          f"max = {cpu_ms[-1]:.2f} ms")
    print("  => sub-millisecond-to-millisecond per recovery, and it does not grow")
    print("     with model size: 70B shard40 is the same order as 7B shard8.")
    print()
    print("### G4 gate (docs/03 §3.2)")
    print("  Simulated metadata_bytes / kv_bytes / update_cpu_ns must land within")
    print("  2x of these medians. They are produced by the same code, so a larger")
    print("  gap indicates a wiring bug, NOT cost-model error.")


if __name__ == "__main__":
    main()
