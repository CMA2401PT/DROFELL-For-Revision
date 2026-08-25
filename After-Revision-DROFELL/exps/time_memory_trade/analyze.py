"""Analyze decoder checkpoint/swap benchmark records."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

from operators.helpers.cuda_utils.memory import MemoryDeltaStatus


PAPER = {
    "each_decoder": {
        "forward_ms": 82.17,
        "swap_ms": 33.40,
        "backward_ms": 185.31,
        "forward_peak_mib": 385.01,
        "forward_residual_mib": 160.01,
        "backward_peak_mib": 707.27,
    },
    "first.2pass": {
        "forward_ms": 83.53,
        "swap_ms": 2.62,
        "backward_ms": 274.07,
        "forward_peak_mib": 289.01,
        "forward_residual_mib": 16.01,
        "backward_peak_mib": 705.27,
    },
    "first.grad": {
        "forward_ms": 82.60,
        "swap_ms": 2.62,
        "backward_ms": 191.38,
        "forward_peak_mib": 289.01,
        "forward_residual_mib": 16.01,
        "backward_peak_mib": 2925.13,
    },
    "grad": {
        "forward_ms": 81.72,
        "swap_ms": 519.16,
        "backward_ms": 103.74,
        "forward_peak_mib": 2909.13,
        "forward_residual_mib": 2706.13,
        "backward_peak_mib": 3482.25,
    },
}
PAPER_NAMES = {
    "each_decoder": "each",
    "first.2pass": "2pass",
    "first.grad": "ckpt",
    "grad": "grad",
}
STRATEGY_ORDER = {name: index for index, name in enumerate(PAPER)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-dpi", type=int, default=240)
    args = parser.parse_args()
    if args.plot_dpi <= 0:
        parser.error("--plot-dpi must be positive")
    return args


def device_trace(snapshot: dict[str, Any]) -> list[Any]:
    traces = snapshot["device_traces"]
    if len(traces) != 1:
        raise ValueError(f"expected one visible CUDA device, got {len(traces)}")
    return traces[0]


def percent_delta(value: float, reference: float) -> float:
    return 100 * (value / reference - 1)


def analyze_record(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        record = pickle.load(stream)
    metadata = record["metadata"]
    params = metadata["parameters"]
    timing = metadata["timing"]

    forward = MemoryDeltaStatus(device_trace(record["forward_memory"]))
    swap_out = MemoryDeltaStatus(device_trace(record["swap_out_memory"]))
    swap_in = MemoryDeltaStatus(device_trace(record["swap_in_memory"]))
    backward = MemoryDeltaStatus(device_trace(record["backward_memory"]))
    if forward.free_before != 0:
        raise ValueError(f"unexpected pre-history free in {path}: {forward.free_before}")

    row: dict[str, Any] = {
        "backend": params["backend"],
        "strategy": params["strategy"],
        "paper_name": PAPER_NAMES[params["strategy"]],
        "reuse_reconstruct_cache": params[
            "reuse_reconstruct_cache_when_grad"
        ],
        "forward_ms": timing["forward"]["mean_ms"],
        "forward_stdev_ms": timing["forward"]["stdev_ms"],
        "swap_ms": timing["swap"]["mean_ms"],
        "swap_stdev_ms": timing["swap"]["stdev_ms"],
        "backward_ms": timing["backward"]["mean_ms"],
        "backward_stdev_ms": timing["backward"]["stdev_ms"],
        "forward_peak_mib": forward.peak_memory / 2**20,
        "forward_residual_mib": forward.preserved_new_alloc / 2**20,
        "after_swap_out_mib": swap_out.preserved_new_alloc / 2**20,
        "swap_in_peak_mib": swap_in.peak_memory / 2**20,
        "backward_peak_mib": backward.peak_memory / 2**20,
        "backward_residual_mib": backward.preserved_new_alloc / 2**20,
        "record_path": str(path),
    }
    if params["backend"] == "exllama_fuse":
        paper = PAPER[params["strategy"]]
        for metric, reference in paper.items():
            row[f"paper_{metric}"] = reference
            row[f"paper_delta_{metric}_percent"] = percent_delta(
                row[metric], reference
            )
    else:
        for metric in next(iter(PAPER.values())):
            row[f"paper_{metric}"] = None
            row[f"paper_delta_{metric}_percent"] = None
    return row


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    record_paths = sorted((run_dir / "raw").glob("*/*/record.pkl"))
    if not record_paths:
        raise FileNotFoundError(f"no record.pkl files under {run_dir / 'raw'}")
    rows = [analyze_record(path) for path in record_paths]
    rows.sort(key=lambda row: (row["backend"], STRATEGY_ORDER[row["strategy"]]))

    by_key = {(row["backend"], row["strategy"]): row for row in rows}
    backend_comparisons: list[dict[str, Any]] = []
    for strategy in PAPER:
        exllama = by_key.get(("exllama_fuse", strategy))
        marlin = by_key.get(("marlin", strategy))
        if exllama is None or marlin is None:
            continue
        backend_comparisons.append(
            {
                "strategy": strategy,
                "paper_name": PAPER_NAMES[strategy],
                "marlin_vs_exllama_forward_time_ratio": (
                    marlin["forward_ms"] / exllama["forward_ms"]
                ),
                "marlin_vs_exllama_swap_time_ratio": (
                    marlin["swap_ms"] / exllama["swap_ms"]
                ),
                "marlin_vs_exllama_backward_time_ratio": (
                    marlin["backward_ms"] / exllama["backward_ms"]
                ),
                "marlin_minus_exllama_forward_peak_mib": (
                    marlin["forward_peak_mib"] - exllama["forward_peak_mib"]
                ),
                "marlin_minus_exllama_forward_residual_mib": (
                    marlin["forward_residual_mib"]
                    - exllama["forward_residual_mib"]
                ),
                "marlin_minus_exllama_backward_peak_mib": (
                    marlin["backward_peak_mib"] - exllama["backward_peak_mib"]
                ),
            }
        )

    summary = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "paper": PAPER,
        "rows": rows,
        "backend_comparisons": backend_comparisons,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not args.no_plots:
        # Keep tabular analysis usable in minimal environments that do not
        # install Matplotlib.
        from scripts.module.decoder.plot import plot_record, plot_run_comparisons

        figures_dir = run_dir / "figures"
        for path in record_paths:
            with path.open("rb") as stream:
                record = pickle.load(stream)
            params = record["metadata"]["parameters"]
            row = by_key[(params["backend"], params["strategy"])]
            backend = row["backend"]
            strategy = row["strategy"]
            backend_label = "ExLlama" if backend == "exllama_fuse" else "Marlin"
            plot_record(
                record,
                figures_dir / "cases" / backend / f"{strategy}.png",
                title=f"{backend_label} / {row['paper_name']}",
                dpi=args.plot_dpi,
            )
        plot_run_comparisons(rows, figures_dir, dpi=args.plot_dpi)
    print(json.dumps(backend_comparisons, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
