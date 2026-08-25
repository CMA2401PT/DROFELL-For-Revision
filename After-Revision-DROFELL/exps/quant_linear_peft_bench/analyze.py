"""Summarize quantized-linear PEFT benchmark JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "our_no_cache": "ExLlama no-cache",
    "our_cache": "ExLlama cache",
    "marlin": "Marlin direct",
    "marlin_dense_no_cache": "Marlin dense no-cache",
    "marlin_dense_cache": "Marlin dense cache",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def plot_metric(
    rows: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["seq_len"],
        )
        values = [row.get(metric) for row in selected]
        if not selected or any(value is None for value in values):
            continue
        axis.plot(
            [row["seq_len"] for row in selected],
            values,
            marker="o",
            label=METHOD_LABELS.get(method, method),
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Flattened tokens (M)")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", linewidth=0.5, alpha=0.35)
    axis.legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def observed_crossover(rows: list[dict[str, Any]]) -> int | None:
    by_key = {(row["method"], row["seq_len"]): row for row in rows}
    lengths = sorted({row["seq_len"] for row in rows})
    for index, seq_len in enumerate(lengths):
        remaining = lengths[index:]
        if all(
            ("marlin", length) in by_key
            and ("marlin_dense_no_cache", length) in by_key
            and by_key[("marlin_dense_no_cache", length)]["forward_mean_ms"]
            <= by_key[("marlin", length)]["forward_mean_ms"]
            for length in remaining
        ):
            return seq_len
    return None


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    raw_dir = run_dir / "raw"
    inputs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(raw_dir.glob("*.json"))
    ]
    if not inputs:
        raise FileNotFoundError(f"no benchmark JSON files under {raw_dir}")

    rows: list[dict[str, Any]] = []
    for result in inputs:
        for record in result["records"]:
            memory = record.get("memory", {})
            row = {
                "method": result["method"],
                "projection": result["projection"],
                "dtype": result["dtype"],
                "seq_len": record["seq_len"],
                "forward_mean_ms": record["forward"]["mean_ms"],
                "forward_stdev_ms": record["forward"]["stdev_ms"],
                "reconstruction_mean_ms": record.get("reconstruction", {}).get(
                    "mean_ms"
                ),
                "dense_forward_only_mean_ms": record.get(
                    "dense_forward_only", {}
                ).get("mean_ms"),
                "peft_forward_mean_ms": record.get("peft_forward", {}).get(
                    "mean_ms"
                ),
                "backward_mean_ms": record.get("backward", {}).get("mean_ms"),
                "backward_stdev_ms": record.get("backward", {}).get(
                    "stdev_ms"
                ),
                "peft_total_mean_ms": record.get("peft_total", {}).get(
                    "mean_ms"
                ),
                "input_grad_checksum": record.get("input_grad_checksum"),
                "retained_after_forward_mib": memory.get(
                    "retained_after_forward_bytes", 0
                )
                / 2**20,
                "forward_peak_delta_mib": memory.get(
                    "forward_peak_delta_bytes", 0
                )
                / 2**20,
                "backward_peak_delta_from_forward_mib": memory.get(
                    "backward_peak_delta_from_forward_bytes", 0
                )
                / 2**20,
                "total_peak_delta_mib": memory.get("total_peak_delta_bytes", 0)
                / 2**20,
            }
            rows.append(row)

    by_key = {(row["method"], row["seq_len"]): row for row in rows}
    comparisons: list[dict[str, Any]] = []
    for seq_len in sorted({row["seq_len"] for row in rows}):
        no_cache = by_key.get(("our_no_cache", seq_len))
        cache = by_key.get(("our_cache", seq_len))
        marlin = by_key.get(("marlin", seq_len))
        comparison: dict[str, Any] = {"seq_len": seq_len}
        if no_cache and cache:
            comparison["exllama_cache_peft_speedup"] = (
                no_cache["peft_total_mean_ms"] / cache["peft_total_mean_ms"]
            )
            comparison["exllama_cache_peft_time_reduction_percent"] = 100 * (
                1
                - cache["peft_total_mean_ms"]
                / no_cache["peft_total_mean_ms"]
            )
            comparison["exllama_cache_backward_speedup"] = (
                no_cache["backward_mean_ms"] / cache["backward_mean_ms"]
            )
            comparison["exllama_cache_vs_no_cache_grad_checksum_delta"] = abs(
                cache["input_grad_checksum"] - no_cache["input_grad_checksum"]
            )
        if no_cache and marlin:
            comparison["marlin_vs_exllama_no_cache_peft_time_ratio"] = (
                marlin["peft_total_mean_ms"] / no_cache["peft_total_mean_ms"]
            )
            comparison["marlin_vs_exllama_no_cache_backward_time_ratio"] = (
                marlin["backward_mean_ms"] / no_cache["backward_mean_ms"]
            )
            comparison["marlin_vs_exllama_no_cache_peft_time_increase_percent"] = (
                100
                * (
                    marlin["peft_total_mean_ms"]
                    / no_cache["peft_total_mean_ms"]
                    - 1
                )
            )
            comparison["marlin_vs_exllama_no_cache_grad_checksum_relative_delta"] = (
                abs(
                    marlin["input_grad_checksum"]
                    - no_cache["input_grad_checksum"]
                )
                / max(abs(no_cache["input_grad_checksum"]), 1e-12)
            )
        if cache and marlin:
            comparison["marlin_vs_exllama_cache_peft_time_ratio"] = (
                marlin["peft_total_mean_ms"] / cache["peft_total_mean_ms"]
            )
        comparisons.append(comparison)

    summary = {
        "schema_version": 2,
        "run_dir": str(run_dir),
        "methods": sorted({result["method"] for result in inputs}),
        "rows": rows,
        "comparisons": comparisons,
        "observed_marlin_forward_crossover_tokens": observed_crossover(rows),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = list(rows[0])
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    plot_metric(
        rows,
        "forward_mean_ms",
        "Inference forward time (ms)",
        run_dir / "marlin-forward-dispatch.png",
    )
    plot_metric(
        rows,
        "peft_total_mean_ms",
        "PEFT forward + backward time (ms)",
        run_dir / "peft-total.png",
    )
    print(json.dumps(comparisons, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
