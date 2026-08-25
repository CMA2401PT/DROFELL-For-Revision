import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns


CaseKey = tuple[int, int, str]
Record = list[list[int]]
SpeedPoint = tuple[int, float]


LEGACY_RESULTS: tuple[tuple[int, int, str], ...] = (
    (1, 1, "7B-1-36K"),
    (1, 4, "7B-4-9K"),
    (1, 8, "7B-8-4500"),
    (1, 16, "7B-16-2250"),
    (2, 1, "7B-2-1-80K"),
    (2, 16, "7B-2-16-2500"),
    (2, 32, "7B-2-32-2500"),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def client_anchored_speeds(
    record: Record,
    *,
    anchor_client: int = 0,
) -> list[SpeedPoint]:
    """Return all-client instantaneous throughput per anchor-client token.

    The first output token is returned by prefill and is absent from
    ``result.json``.  Positions here are one-based within the timed decode
    trace; this distinction is at most one token and is recorded explicitly in
    the experiment README.
    """
    if len(record) < 2:
        return []
    anchor_indices = [
        index for index, (client, _) in enumerate(record)
        if int(client) == anchor_client
    ]
    points: list[SpeedPoint] = []
    for occurrence, (left, right) in enumerate(
        zip(anchor_indices, anchor_indices[1:]), start=2
    ):
        elapsed_ns = int(record[right][1]) - int(record[left][1])
        if elapsed_ns <= 0:
            raise ValueError("timestamps must increase between anchor tokens")
        completed_tokens = right - left
        points.append((occurrence, completed_tokens / (elapsed_ns / 1e9)))
    return points


def centered_average(
    points: Iterable[SpeedPoint],
    *,
    window: int = 3,
) -> list[SpeedPoint]:
    """Apply an odd centered moving average while retaining every center."""
    values = list(points)
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    radius = window // 2
    if len(values) < window:
        return []
    return [
        (
            values[index][0],
            sum(value for _, value in values[index-radius:index+radius+1])
            / window,
        )
        for index in range(radius, len(values) - radius)
    ]


def paper_speed_points(record: Record) -> list[SpeedPoint]:
    return centered_average(client_anchored_speeds(record), window=3)


def discover_records(root: Path) -> dict[CaseKey, Record]:
    """Load the new experiment layout or the historical seven directories."""
    records: dict[CaseKey, Record] = {}
    for result_path in sorted(root.glob("scale-*/clients-*/*/result.json")):
        deployment = result_path.parent.name
        if deployment not in {"combined-request-agent", "persistent-agent"}:
            continue
        scale = int(result_path.parents[2].name.removeprefix("scale-"))
        clients = int(result_path.parents[1].name.removeprefix("clients-"))
        records[(scale, clients, deployment)] = read_json(result_path)
    if records:
        return records
    for scale, clients, directory in LEGACY_RESULTS:
        result_path = root / directory / "result.json"
        if result_path.is_file():
            records[(scale, clients, "legacy")] = read_json(result_path)
    if not records:
        raise FileNotFoundError(
            f"no infer-scale result.json files found below {root}"
        )
    return records


def records_to_frame(
    records: Mapping[CaseKey, Record],
    *,
    deployment: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scale, clients, case_deployment), record in records.items():
        if case_deployment != deployment:
            continue
        for position, speed in paper_speed_points(record):
            rows.append({
                "Generated Token Position": position,
                "Tokens Per Second": speed,
                "clients": str(clients),
                "scale": str(scale),
                "deployment": deployment,
            })
    if not rows:
        raise ValueError(f"no records for deployment {deployment!r}")
    return pd.DataFrame.from_records(rows)


def _plot_ranges(data: pd.DataFrame) -> list[tuple[float, float]]:
    maxima = {
        (int(scale), int(clients)): int(group["Generated Token Position"].max())
        for (scale, clients), group in data.groupby(["scale", "clients"])
    }
    # Keep the concurrency-focused part readable.  Lower-concurrency traces
    # are much longer because every case uses the same total token budget, so
    # their endpoints belong in compact broken-axis tail panels.
    main_values = [
        maximum for (_, clients), maximum in maxima.items() if clients >= 8
    ]
    if not main_values:
        main_values = [
            maximum for (_, clients), maximum in maxima.items()
            if clients != 1
        ]
    main_end = max(main_values, default=min(maxima.values()))
    main_end = max(256, int(main_end * 1.03))
    ranges: list[tuple[float, float]] = [(0, main_end)]
    tail_ends = sorted({
        maximum for maximum in maxima.values() if maximum > main_end * 1.2
    })
    clusters: list[list[int]] = []
    cluster_gap = main_end * 0.5
    for maximum in tail_ends:
        if not clusters or maximum - clusters[-1][-1] > cluster_gap:
            clusters.append([maximum])
        else:
            clusters[-1].append(maximum)
    for cluster in clusters:
        lower, upper = cluster[0], cluster[-1]
        half_width = min(750, max(200, upper * 0.015))
        ranges.append((lower - half_width, upper + half_width * 0.25))
    return ranges


def draw_break_marks(
    ax: Any,
    *,
    side: str,
    size: float = 0.012,
) -> None:
    figure = ax.figure
    figure.canvas.draw()
    bbox = ax.get_position()
    figure_width, figure_height = figure.get_size_inches()
    axis_width = bbox.width * figure_width
    axis_height = bbox.height * figure_height
    dy = size
    dx = size * (axis_height / axis_width)
    x = 1.0 if side == "right" else 0.0
    kwargs = {
        "transform": ax.transAxes,
        "color": "black",
        "clip_on": False,
        "linewidth": 1.0,
    }
    ax.plot((x - dx, x + dx), (-dy, +dy), **kwargs)
    ax.plot((x - dx, x + dx), (1 - dy, 1 + dy), **kwargs)


def plot_deployment(
    records: Mapping[CaseKey, Record],
    *,
    deployment: str,
    output: Path,
) -> None:
    data = records_to_frame(records, deployment=deployment)
    ranges = _plot_ranges(data)
    raw_widths = [right - left for left, right in ranges]
    # Long-context panels cover a small numerical range but still need enough
    # physical width for readable tick labels and the client=1 tail.
    minimum_panel_width = raw_widths[0] * 0.22
    widths = [raw_widths[0]] + [
        max(width, minimum_panel_width) for width in raw_widths[1:]
    ]
    client_order = [str(value) for value in sorted({
        int(value) for value in data["clients"].unique()
    })]
    sns.set_style("whitegrid")
    sns.set_theme(font="Times New Roman")
    figure, axes = plt.subplots(
        ncols=len(ranges), nrows=1, sharey=True,
        figsize=(max(9, 7 + 1.2 * (len(ranges) - 1)), 5),
        gridspec_kw={"wspace": 0.03, "width_ratios": widths},
    )
    if len(ranges) == 1:
        axes = [axes]
    for index, (axis, x_range) in enumerate(zip(axes, ranges)):
        sns.lineplot(
            data=data,
            x="Generated Token Position",
            y="Tokens Per Second",
            hue="clients",
            hue_order=client_order,
            style="scale",
            estimator=None,
            sort=False,
            ax=axis,
        )
        axis.set_xlim(*x_range)
        leftmost = index == 0
        rightmost = index == len(axes) - 1
        if leftmost:
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
        else:
            midpoint = round(((x_range[0] + x_range[1]) / 2) / 100) * 100
            axis.set_xticks([midpoint])
        if not rightmost:
            axis.spines["right"].set_visible(False)
            if axis.legend_ is not None:
                axis.legend_.remove()
            draw_break_marks(axis, side="right")
        if not leftmost:
            axis.spines["left"].set_visible(False)
            axis.tick_params(axis="y", left=False, labelleft=False)
            axis.set_ylabel("")
            draw_break_marks(axis, side="left")
        else:
            axis.set_ylabel("Tokens Per Second (All Clients, Current)")
        axis.set_xlabel("")
    figure.text(0.5, 0.04, "Generated Token Position", ha="center")
    figure.suptitle(deployment, y=0.95)
    figure.subplots_adjust(bottom=0.14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    records = discover_records(args.results)
    output_dir = args.output_dir or args.results / "figures"
    deployments = sorted({deployment for _, _, deployment in records})
    for deployment in deployments:
        output = output_dir / f"infer-scale-{deployment}.png"
        plot_deployment(records, deployment=deployment, output=output)
        print(output)


if __name__ == "__main__":
    main()
