#!/usr/bin/env python3
"""Render the three figures the simulation line can defend.

STYLE follows the paper's existing figures (seaborn whitegrid, Times New Roman,
dpi 300), so these drop next to `返修稿/figs/` without looking foreign.

WHY THESE THREE AND NOT MORE
----------------------------
Each one answers a distinct half of reviewer 2.7, and each is backed by a
validation gate rather than by the simulator's own word:

  fig 1  VALIDATION.  Simulated vs measured across the four segment counts the
         authors actually ran, then extrapolated past them. Establishes that the
         extrapolation is entitled to be believed -- and marks where measurement
         stops, because that boundary is the honest part.

  fig 2  THE ANSWER.  Two axes side by side: path length costs 9.5x, network size
         at fixed path length costs nothing. "Tens to hundreds of nodes" in an
         edge deployment means more holders, not a longer pipeline, so this is the
         figure that actually addresses the concern.

  fig 3  RECOVERY.  Overhead of one node failure against network size, plus the
         replay-window finding. Supports the resilience claim at scale.

DELIBERATELY NOT PLOTTED
------------------------
* Absolute throughput at scales with no measured counterpart, as a headline. The
  extrapolated region is drawn but visually separated, never presented as if it
  were measured.
* Anything about output quality. Simulated tokens are faked (docs/04 §2).

Run from the repository root, after the result JSONs exist:
    python simulate/analysis/make_figures.py --out simulate/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS = Path("simulate/results")
MEASURED_SCALE = Path(
    "一稿返修/整理后就绪的实验/推理扩展性(带规模实验)/AutoDL8卡结果/summary.csv"
)
DEPLOYMENT = "combined-request-agent"
# The measured matrix stops at 8 GPUs. Everything past that is extrapolation and
# is drawn as such.
MEASURED_LIMIT = 8


def style() -> None:
    sns.set_style("whitegrid")
    sns.set_theme(font="Times New Roman", style="whitegrid")


def load(name: str) -> Optional[dict[str, Any]]:
    path = RESULTS / name
    if not path.exists():
        print(f"[skip] {path} not found -- run the analysis scripts first")
        return None
    return json.loads(path.read_text())


def measured_curve(clients: int = 1) -> dict[int, float]:
    """Per-token latency (ms) of one client stream, per GPU/segment count."""
    if not MEASURED_SCALE.exists():
        return {}
    out: dict[int, float] = {}
    with MEASURED_SCALE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["deployment"] != DEPLOYMENT:
                continue
            if int(row["clients"]) != clients:
                continue
            out[int(row["scale"])] = (
                1000.0 * clients / float(row["overall_tokens_per_second"])
            )
    return out


# ---------------------------------------------------------------------------
# Figure 1: validation, then extrapolation
# ---------------------------------------------------------------------------
def figure_validation(out_dir: Path) -> Optional[Path]:
    report = load("sim_vs_measured.json")
    sweep = load("d1_scale_sweep.json")
    measured = measured_curve()
    if report is None or not measured:
        return None

    sim_points = {
        row["segments"]: row["sim_with_turnaround_ms"] for row in report["rows"]
    }
    extrapolated: dict[int, float] = {}
    if sweep is not None:
        # The D1 sweep runs with client turnaround OFF, so it is not directly
        # comparable in absolute terms. Shift it by the same constant used in the
        # validated range rather than mixing two conventions on one axis.
        offsets = [
            row["sim_with_turnaround_ms"] - row["sim_bare_ms"]
            for row in report["rows"]
        ]
        offset = sum(offsets) / len(offsets) if offsets else 0.0
        for row in sweep["sweeps"]["path_length"]:
            if row["segments"] > MEASURED_LIMIT:
                extrapolated[row["segments"]] = row["inter_token_ms"] + offset

    style()
    figure, axes = plt.subplots(figsize=(7.5, 5))

    measured_x = sorted(measured)
    axes.plot(
        measured_x, [measured[x] for x in measured_x],
        marker="o", markersize=9, linewidth=2.2, color="#1f4e79",
        label="Measured (8-GPU host)", zorder=4,
    )
    sim_x = sorted(sim_points)
    axes.plot(
        sim_x, [sim_points[x] for x in sim_x],
        marker="s", markersize=8, linewidth=2.2, color="#c0392b",
        linestyle="--", label="DROSIM (validated range)", zorder=3,
    )
    if extrapolated:
        bridge_x = [max(sim_x)] + sorted(extrapolated)
        bridge_y = [sim_points[max(sim_x)]] + [
            extrapolated[x] for x in sorted(extrapolated)
        ]
        axes.plot(
            bridge_x, bridge_y,
            marker="^", markersize=8, linewidth=2.0, color="#c0392b",
            linestyle=":", alpha=0.75,
            label="DROSIM (extrapolated, no measured counterpart)", zorder=2,
        )
        axes.axvspan(
            MEASURED_LIMIT, max(extrapolated), color="#c0392b", alpha=0.05,
            zorder=0,
        )
        axes.axvline(
            MEASURED_LIMIT, color="#666666", linewidth=1.2, linestyle="-.",
            zorder=1,
        )
        axes.annotate(
            "hardware stops here",
            xy=(MEASURED_LIMIT, max(bridge_y) * 0.97),
            xytext=(MEASURED_LIMIT * 1.15, max(bridge_y) * 0.97),
            fontsize=11, color="#444444", va="top",
        )

    worst = report.get("worst_error_pct")
    fits = report.get("fits", {})
    hop_sim = fits.get("sim_bare", {}).get("h_ms_per_hop")
    hop_meas = fits.get("measured", {}).get("h_ms_per_hop")
    subtitle = []
    if worst is not None:
        subtitle.append(f"worst error {worst:.1f}% over 1-8 segments")
    if hop_sim and hop_meas:
        subtitle.append(
            f"per-hop cost {hop_sim:.2f} vs {hop_meas:.2f} ms (not fitted)"
        )

    axes.set_xscale("log", base=2)
    axes.set_xticks(sorted(set(measured_x) | set(sim_x) | set(extrapolated)))
    axes.get_xaxis().set_major_formatter(
        matplotlib.ticker.ScalarFormatter()
    )
    axes.set_xlabel("Model segments (= logical nodes on the token path)", fontsize=13)
    axes.set_ylabel("Per-token latency (ms)", fontsize=13)
    axes.set_title(
        "Simulator validated against hardware, then extrapolated\n"
        + " · ".join(subtitle),
        fontsize=13,
    )
    axes.legend(fontsize=11, loc="upper left")
    figure.tight_layout()
    path = out_dir / "sim-validation.pdf"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Figure 2: the two axes
# ---------------------------------------------------------------------------
def figure_two_axes(out_dir: Path) -> Optional[Path]:
    sweep = load("d1_scale_sweep.json")
    if sweep is None:
        return None
    path_rows = sweep["sweeps"]["path_length"]
    size_rows = sweep["sweeps"]["network_size"]

    style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))

    xs = [row["logical_nodes"] for row in path_rows]
    ys = [row["inter_token_ms"] for row in path_rows]
    left.plot(
        xs, ys, marker="o", markersize=9, linewidth=2.4, color="#c0392b",
    )
    left.set_xscale("log", base=2)
    left.set_xticks(xs)
    left.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    left.set_xlabel("Logical nodes (each adds one hop)", fontsize=12)
    left.set_ylabel("Per-token latency (ms)", fontsize=12)
    growth = ys[-1] / ys[0]
    left.set_title(
        f"(a) Longer path: {growth:.1f}x cost\nnode count grows the token path",
        fontsize=12,
    )
    for x, y in zip(xs, ys):
        left.annotate(
            f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 9),
            ha="center", fontsize=10, color="#7b241c",
        )

    xs2 = [row["logical_nodes"] for row in size_rows]
    ys2 = [row["inter_token_ms"] for row in size_rows]
    right.plot(
        xs2, ys2, marker="s", markersize=9, linewidth=2.4, color="#1f4e79",
    )
    right.set_xscale("log", base=2)
    right.set_xticks(xs2)
    right.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    right.set_xlabel(
        "Logical nodes (path length fixed at 8 hops)", fontsize=12
    )
    right.set_ylabel("Per-token latency (ms)", fontsize=12)
    spread = 100 * (max(ys2) - min(ys2)) / min(ys2)
    right.set_title(
        f"(b) More holders: {spread:.1f}% change\nnode count grows redundancy",
        fontsize=12,
    )
    # A flat line needs a scale that shows it is flat, not a scale that magnifies
    # numerical noise into an apparent trend.
    centre = sum(ys2) / len(ys2)
    right.set_ylim(centre * 0.5, centre * 1.5)
    for x, y in zip(xs2, ys2):
        right.annotate(
            f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 11),
            ha="center", fontsize=10, color="#154360",
        )

    figure.suptitle(
        "Orchestration cost tracks path length, not network size",
        fontsize=14,
    )
    figure.tight_layout()
    path = out_dir / "sim-two-axes.pdf"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# Figure 3: recovery
# ---------------------------------------------------------------------------
def figure_recovery(out_dir: Path) -> Optional[Path]:
    report = load("d3_recovery.json")
    if report is None:
        return None
    benefit_rows = report["sweeps"].get("redundancy_benefit", [])
    replay_rows = report["sweeps"].get("replay_distance", [])
    if not benefit_rows:
        return None

    style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))

    # (a) what redundancy buys, and that it stays bought as the network grows
    labels = [
        f"r={row['redundancy']}\n({row['logical_nodes']} nodes)"
        for row in benefit_rows
    ]
    values = [row["recovery_overhead_ms"] for row in benefit_rows]
    colours = ["#c0392b"] + ["#1f4e79"] * (len(values) - 1)
    left.bar(labels, values, color=colours, width=0.6)
    left.set_ylabel("Recovery overhead on the critical path (ms)", fontsize=12)
    left.set_xlabel("Redundancy (and resulting logical node count)", fontsize=12)
    left.set_ylim(0, max(max(values) * 1.5, 0.5))
    left.set_title(
        "(a) One node killed: what redundancy buys\n"
        "cost stays ~0 from 8 to 32 nodes",
        fontsize=12,
    )
    for index, value in enumerate(values):
        left.annotate(
            f"{value:.2f} ms" if value >= 0.01 else "~0",
            (index, value), textcoords="offset points", xytext=(0, 6),
            ha="center", fontsize=10,
            color="#7b241c" if index == 0 else "#154360",
        )
    if all(
        row["tokens_produced"] == benefit_rows[0]["tokens_produced"]
        for row in benefit_rows
    ):
        left.annotate(
            "every run completed all requested tokens",
            xy=(0.5, 0.93), xycoords="axes fraction", ha="center",
            fontsize=10, color="#444444",
        )

    # (b) the replay window is bounded by design, measured where it is visible
    if replay_rows:
        xs = [row["drop_step"] for row in replay_rows]
        ys = [row["recovery_overhead_ms"] for row in replay_rows]
        right.plot(
            xs, ys, marker="D", markersize=8, linewidth=2.4, color="#c0392b",
        )
        right.set_xlabel(
            "Decode step at which the node was killed", fontsize=12
        )
        right.set_ylabel("Recovery overhead (ms)", fontsize=12)
        centre = sum(ys) / len(ys)
        right.set_ylim(centre * 0.5, centre * 1.5)
        right.xaxis.set_major_locator(MaxNLocator(integer=True))
        right.set_title(
            "(b) Killing later costs no more (no redundancy)\n"
            "the shadow keeps only the unconfirmed tail",
            fontsize=12,
        )
        for x, y in zip(xs, ys):
            right.annotate(
                f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=10, color="#7b241c",
            )

    figure.suptitle(
        "Redundancy moves recovery off the critical path; "
        "the replay window bounds its cost",
        fontsize=14,
    )
    figure.tight_layout()
    path = out_dir / "sim-recovery.pdf"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("simulate/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("rendering simulation figures")
    print("=" * 70)
    for builder in (figure_validation, figure_two_axes, figure_recovery):
        path = builder(args.out)
        if path is not None:
            print(f"  wrote {path} (+ .png)")
    print(
        "\nNOTE every figure that shows an extrapolated region marks it. Do not"
    )
    print("crop that marking off when placing these in the manuscript.")


if __name__ == "__main__":
    main()
