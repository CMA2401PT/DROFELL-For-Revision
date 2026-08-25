#!/usr/bin/env python3
"""Validate and plot PEFT correctness experiment results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CONDITIONS = ("reference", "offline", "online-static", "online-recovery")
ADAPTER_LABELS = {"ptuning": "P-Tuning v2", "lora": "LoRA"}
COLORS = {
    "reference": "#111111",
    "offline": "#377eb8",
    "online-static": "#4daf4a",
    "online-recovery": "#e41a1c",
}
LINESTYLES = {
    "reference": "-",
    "offline": "--",
    "online-static": "-.",
    "online-recovery": ":",
}
SMOOTHING_WINDOW = 32
RAW_LINE_ALPHA = 0.30


class AnalysisError(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise AnalysisError(f"missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def record_for(run_root: Path, adapter: str, condition: str) -> dict[str, Any]:
    adapter_root = run_root / adapter
    if condition in {"reference", "offline"}:
        return read_json(adapter_root / condition / "repeat-0/raw/record.json")
    online = adapter_root / "online"
    case = "normal" if condition == "online-static" else "recover-latency-0ms"
    value = read_json(online / case / "repeat-0/raw/results.json")
    if not isinstance(value, list) or len(value) < 2:
        raise AnalysisError(f"unexpected online results structure for {adapter}/{case}")
    return value[0]


def minibatch_rows(adapter: str, condition: str, record: dict[str, Any], expected: int) -> list[dict[str, Any]]:
    required = ("history", "ids", "tokens")
    if any(key not in record for key in required):
        raise AnalysisError(f"{adapter}/{condition} lacks record keys {required}")
    histories = record["history"]
    ids = record["ids"]
    tokens = record["tokens"]
    if not (len(histories) == len(ids) == len(tokens) == expected):
        raise AnalysisError(
            f"{adapter}/{condition}: expected {expected} minibatches, got "
            f"loss={len(histories)}, ids={len(ids)}, tokens={len(tokens)}"
        )
    rows = []
    seen: set[str] = set()
    for loss, minibatch_id, token_count in zip(histories, ids, tokens):
        minibatch_id = str(minibatch_id)
        if minibatch_id in seen:
            raise AnalysisError(f"duplicate minibatch {minibatch_id} in {adapter}/{condition}")
        seen.add(minibatch_id)
        try:
            version_text, index_text = minibatch_id.split("-", 1)
            version, index = int(version_text), int(index_text)
            loss_value = float(loss)
        except (TypeError, ValueError) as error:
            raise AnalysisError(f"invalid minibatch row in {adapter}/{condition}") from error
        if not math.isfinite(loss_value):
            raise AnalysisError(f"non-finite loss in {adapter}/{condition}/{minibatch_id}")
        rows.append({
            "adapter": adapter,
            "condition": condition,
            "minibatch_id": minibatch_id,
            "minibatch_index": index,
            "optimizer_update": version,
            "loss": loss_value,
            "tokens": int(token_count),
        })
    rows.sort(key=lambda row: row["minibatch_index"])
    if [row["minibatch_index"] for row in rows] != list(range(expected)):
        raise AnalysisError(f"minibatch indices are not exactly 0..{expected - 1} in {adapter}/{condition}")
    return rows


def update_losses(rows: list[dict[str, Any]], expected_updates: int) -> list[float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["optimizer_update"]), []).append(float(row["loss"]))
    if sorted(grouped) != list(range(expected_updates)):
        raise AnalysisError("optimizer update IDs are incomplete")
    return [statistics.mean(grouped[index]) for index in range(expected_updates)]


def moving_average(values: list[float], window: int) -> list[float]:
    if window < 1:
        raise ValueError("moving-average window must be positive")
    result: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def failure_versions(run_root: Path, adapter: str) -> list[int]:
    path = run_root / adapter / "online/recover-latency-0ms/repeat-0/events.jsonl"
    if not path.is_file():
        return []
    versions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "failure_injected" or event.get("name") == "failure_injected":
            version = event.get("request_version")
            if type(version) is int and version >= 0:
                versions.append(version)
    return sorted(set(versions))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(run_root: Path) -> None:
    manifest = read_json(run_root / "manifest.json")
    parameters = manifest["parameters"]
    adapters = list(parameters["adapters"])
    expected_minibatches = int(parameters["minibatches"])
    expected_updates = int(parameters["optimizer_updates"])
    submission_window_s = float(parameters["planned_submission_window_s"])
    all_rows: list[dict[str, Any]] = []
    minibatch_data: dict[str, dict[str, list[dict[str, Any]]]] = {}
    update_data: dict[str, dict[str, list[float]]] = {}
    summary_rows: list[dict[str, Any]] = []

    for adapter in adapters:
        minibatch_data[adapter] = {}
        update_data[adapter] = {}
        for condition in CONDITIONS:
            rows = minibatch_rows(
                adapter, condition,
                record_for(run_root, adapter, condition),
                expected_minibatches,
            )
            all_rows.extend(rows)
            minibatch_data[adapter][condition] = rows
            values = update_losses(rows, expected_updates)
            update_data[adapter][condition] = values
            final = values[-min(50, len(values)):]
            first_below_two = next((index for index, value in enumerate(values) if value < 2.0), None)
            summary_rows.append({
                "adapter": adapter,
                "condition": condition,
                "optimizer_updates": len(values),
                "initial_20_update_mean_loss": statistics.mean(values[:min(20, len(values))]),
                "final_50_update_mean_loss": statistics.mean(final),
                "final_50_update_std_loss": statistics.pstdev(final),
                "minimum_update_mean_loss": min(values),
                "first_update_mean_below_2": first_below_two,
            })

    tables = run_root / "tables"
    figures = run_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "minibatches.csv", all_rows)
    write_csv(tables / "summary.csv", summary_rows)

    adapter_summaries: dict[str, Any] = {}
    for adapter in adapters:
        fig, axis = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
        for condition in CONDITIONS:
            rows = minibatch_data[adapter][condition]
            xs = [int(row["minibatch_index"]) for row in rows]
            values = [float(row["loss"]) for row in rows]
            axis.plot(
                xs,
                values,
                color=COLORS[condition],
                linestyle=LINESTYLES[condition],
                alpha=RAW_LINE_ALPHA,
                linewidth=0.65,
                zorder=1,
            )
            axis.plot(
                xs,
                moving_average(values, SMOOTHING_WINDOW),
                color=COLORS[condition],
                linestyle=LINESTYLES[condition],
                linewidth=1.6,
                label=condition,
                zorder=3,
            )
        for version in failure_versions(run_root, adapter):
            recovery_rows = minibatch_data[adapter]["online-recovery"]
            failure_x = next(
                (int(row["minibatch_index"]) for row in recovery_rows
                 if int(row["optimizer_update"]) == version),
                None,
            )
            if failure_x is not None:
                axis.axvline(
                    failure_x,
                    color=COLORS["online-recovery"],
                    linestyle=":",
                    linewidth=0.9,
                    zorder=4,
                )
                axis.annotate(
                    "recovery",
                    xy=(failure_x, 0.62),
                    xycoords=("data", "axes fraction"),
                    xytext=(3, 0),
                    textcoords="offset points",
                    color=COLORS["online-recovery"],
                    fontsize=7.5,
                    rotation=90,
                    ha="left",
                    va="top",
                    zorder=5,
                )
        axis.axhline(2.0, color="#777777", linestyle="--", linewidth=0.8, zorder=2)
        axis.set_xlabel("Minibatch index")
        axis.set_ylabel("Training loss")
        axis.set_title(
            f"{ADAPTER_LABELS.get(adapter, adapter)} — "
            f"minibatch loss with MA({SMOOTHING_WINDOW}) "
            f"({submission_window_s:g} s submission window)"
        )
        axis.grid(alpha=0.2)
        condition_legend = axis.legend(
            frameon=True,
            framealpha=0.9,
            ncol=1,
            loc="upper right",
            title="Condition",
        )
        axis.add_artist(condition_legend)
        layer_handles = [
            Line2D(
                [0], [0], color="#666666", linewidth=0.65,
                alpha=RAW_LINE_ALPHA, label="raw loss (background)",
            ),
            Line2D(
                [0], [0], color="#666666", linewidth=1.6,
                label=f"MA({SMOOTHING_WINDOW}) (foreground)",
            ),
        ]
        axis.legend(
            handles=layer_handles,
            frameon=True,
            framealpha=0.9,
            loc="upper center",
            title="Line layer",
        )
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"{adapter}_loss.{suffix}", dpi=220)
        plt.close(fig)

        if adapter == "ptuning":
            fig, axis = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
            rows = minibatch_data[adapter]["offline"]
            xs = [int(row["minibatch_index"]) for row in rows]
            values = [float(row["loss"]) for row in rows]
            axis.plot(
                xs,
                values,
                color=COLORS["offline"],
                alpha=RAW_LINE_ALPHA,
                linewidth=0.65,
                label="raw minibatch loss (background)",
                zorder=1,
            )
            axis.plot(
                xs,
                moving_average(values, SMOOTHING_WINDOW),
                color=COLORS["offline"],
                linewidth=1.8,
                label=f"moving average (w={SMOOTHING_WINDOW}, foreground)",
                zorder=3,
            )
            axis.axhline(2.0, color="#777777", linestyle="--", linewidth=0.8, zorder=2)
            axis.set_xlabel("Minibatch index")
            axis.set_ylabel("Training loss")
            axis.set_title(
                f"P-Tuning v2 — offline minibatch loss with MA({SMOOTHING_WINDOW}) "
                f"({submission_window_s:g} s submission window)"
            )
            axis.grid(alpha=0.2)
            axis.legend(frameon=True, framealpha=0.9, loc="upper right")
            for suffix in ("png", "pdf"):
                fig.savefig(figures / f"{adapter}_offline_loss.{suffix}", dpi=220)
            plt.close(fig)

        selected = [row for row in summary_rows if row["adapter"] == adapter]
        final_means = [float(row["final_50_update_mean_loss"]) for row in selected]
        adapter_summaries[adapter] = {
            "all_conditions_completed": True,
            "all_final_window_means_below_2": all(value < 2.0 for value in final_means),
            "final_window_mean_range": max(final_means) - min(final_means),
            "conditions": {row["condition"]: row for row in selected},
            "failure_request_versions": failure_versions(run_root, adapter),
        }

    conclusion = {
        "scope": (
            "Basic convergence comparison only. Independent initialization means the result "
            "does not establish stepwise gradient or parameter equivalence."
        ),
        "adapters": adapter_summaries,
    }
    write_json = lambda path, value: path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_json(run_root / "summary.json", {
        "schema_version": 1,
        "validated": True,
        "expected_minibatches_per_condition": expected_minibatches,
        "expected_optimizer_updates_per_condition": expected_updates,
        "conclusion": conclusion,
    })
    print(json.dumps(conclusion, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args().run_root.resolve())
