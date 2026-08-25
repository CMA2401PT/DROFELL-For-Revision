#!/usr/bin/env python3
"""Aggregate consolidation-window sensitivity runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics
from typing import Any

import matplotlib.pyplot as plt


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pattern = re.compile(
        rf"^{re.escape(args.run_prefix)}-(?P<window>[0-9]+(?:p[0-9]+)?)ms-"
        r"d(?P<dual>[01])$"
    )
    rows = []
    for run_dir in sorted(args.results_root.iterdir()):
        match = pattern.match(run_dir.name)
        if match is None or not (run_dir / "summary.json").exists():
            continue
        window_ms = float(match.group("window").replace("p", "."))
        dual = int(match.group("dual"))
        case_id = f"c1-m1-d{dual}"
        case = read_json(run_dir / "summary.json")["cases"][case_id]
        rates = [float(item["observed_tok_s"]) for item in case["repeats"]]
        batch = case["batch_operations"]["decoder_generating"]
        marks = case["marks_repeat_medians_ms"]
        latency = case["request_latency_repeat_medians_ms"]
        rows.append({
            "run_id": run_dir.name,
            "window_ms": window_ms,
            "dual_queues": dual,
            "repeat_count": len(rates),
            "throughput_median_tok_s": statistics.median(rates),
            "throughput_sd_tok_s": statistics.stdev(rates),
            "decode_wall_median_tok_s": case["decode_wall_tok_s"]["median"],
            "decoder_batch_mean_median": batch[
                "repeat_mean_batch_size"
            ]["median"],
            "decoder_fraction_ge_7_median": batch[
                "repeat_fraction_ge_7"
            ]["median"],
            "cumulative_hold_per_task_median_ms": marks[
                "consolidation_cumulative_hold_per_task_ms"
            ]["median"],
            "window_hold_median_ms": marks[
                "consolidation_window_hold_ms"
            ]["median"],
            "decode_start_to_completion_median_ms": latency[
                "decode_start_to_completion_ms"
            ]["median"],
            "inter_token_median_ms": latency["inter_token_ms"]["median"],
            "repeat_throughputs_tok_s": rates,
        })
    if not rows:
        raise SystemExit("no matching sensitivity runs")
    rows.sort(key=lambda row: (row["dual_queues"], row["window_ms"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "window_sensitivity.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_rows = [
        {key: value for key, value in row.items()
         if key != "repeat_throughputs_tok_s"}
        for row in rows
    ]
    with (args.output_dir / "window_sensitivity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=180)
    for dual, label, color in (
        (1, "dual queues on", "#00ADB5"),
        (0, "dual queues off", "#FF2E63"),
    ):
        selected = [row for row in rows if row["dual_queues"] == dual]
        windows = [row["window_ms"] for row in selected]
        axes[0].errorbar(
            windows,
            [row["throughput_median_tok_s"] for row in selected],
            yerr=[row["throughput_sd_tok_s"] for row in selected],
            marker="o", capsize=3, label=label, color=color,
        )
        axes[1].plot(
            windows,
            [row["decoder_batch_mean_median"] for row in selected],
            marker="o", label=label, color=color,
        )
        axes[2].plot(
            windows,
            [row["cumulative_hold_per_task_median_ms"] for row in selected],
            marker="o", label=label, color=color,
        )
    axes[0].set_ylabel("Observed throughput (token/s)")
    axes[1].set_ylabel("Mean decoder batch size")
    axes[2].set_ylabel("Cumulative hold/task (ms)")
    for axis in axes:
        axis.set_xlabel("Configured maximum window (ms)")
        axis.set_xscale("log")
        axis.set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
        axis.grid(True, alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Consolidation cap-release window sensitivity")
    figure.tight_layout()
    figure.savefig(args.output_dir / "window_sensitivity.png")
    plt.close(figure)


if __name__ == "__main__":
    main()
