import argparse
import json
import os
from typing import Any, Dict, List, Tuple

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from analysis_infer import Analyzer, extract_speed_base, smooth_speed
import matplotlib.pyplot as plt
import seaborn as sns


results: List[Tuple[str, str, str]] = [
    ("1", "1", "7B-1-36K"),
    ("1", "4", "7B-4-9K"),
    ("1", "8", "7B-8-4500"),
    ("1", "16", "7B-16-2250"),
    ("2", "1", "7B-2-1-80K"),
    ("2", "16", "7B-2-16-2500"),
    ("2", "32", "7B-2-32-2500"),
]

parser = argparse.ArgumentParser()
parser.add_argument("output", type=str)
args, _ = parser.parse_known_args()
output_dir = args.output
analyzer = Analyzer(limit=128*1024)
for node_info, clients, dir_path in results:
    result_dir = os.path.join(output_dir, dir_path)
    with open(os.path.join(result_dir, "result.json"), "r") as f:
        detail = json.load(f)
        analyzer.add_record(f"{node_info}#{clients}", detail)
plt.figure(figsize=(10, 6), dpi=330)
# analyzer.plot(lambda name: {"scale": name.split("#")[0], "clients": name.split("#")[1]}, hue="clients", style="scale", limit=65535, window=3)
limit = 128*1024
window = 3


def name_to_kwargs(name: str):
    return {"scale": name.split("#")[0], "clients": name.split("#")[1]}


hue = "clients"
style = "scale"
pd_records: List[Dict[str, Any]] = []
for rec_name, record in analyzer.records.items():
    speed = extract_speed_base(record)
    cur_limit = limit
    if cur_limit == -1:
        cur_limit = analyzer.limit
    if cur_limit != -1:
        speed = speed[:cur_limit]
    speed = smooth_speed(speed=speed, window=window)
    extra_args = name_to_kwargs(rec_name)
    for i, s in enumerate(speed):
        r: Dict[str, Any] = {"Seq Length": i*window, "Parallel": rec_name, "Tokens Per Second": s}
        r.update(extra_args)
        pd_records.append(r)
data = pd.DataFrame.from_records(pd_records)

sns.set_style("whitegrid")
sns.set_theme(font="Times New Roman")

ranges = [(0, 10000), (35000, 36500), (79000, 80500)]

width_ratios = [r[1] - r[0] for r in ranges]

fig, axs = plt.subplots(
    ncols=len(ranges),
    nrows=1,
    sharey=True,
    figsize=(8, 5),
    gridspec_kw={
        "wspace": 0.03,
        "width_ratios": width_ratios
    }
)

if len(ranges) == 1:
    axs = [axs]

common_kwargs: Dict[str, str] = dict(
    data=data,
    x="Seq Length",
    y="Tokens Per Second",
    hue=hue,
    style=style,
)

d = 0.04


def draw_break_marks(ax: Any, side: str = "right", size: float = 0.015, **line_kwargs: Any):
    fig = ax.figure
    fig.canvas.draw()

    bbox = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    ax_w = bbox.width * fig_w
    ax_h = bbox.height * fig_h
    dy = size
    dx = size * (ax_h / ax_w)

    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1.0)
    kwargs.update(line_kwargs)

    if side == "right":
        x0 = 1.0
        ax.plot((x0 - dx, x0 + dx), (-dy, +dy), **kwargs)
        ax.plot((x0 - dx, x0 + dx), (1 - dy, 1 + dy), **kwargs)

    elif side == "left":
        x0 = 0.0
        ax.plot((x0 - dx, x0 + dx), (-dy, +dy), **kwargs)
        ax.plot((x0 - dx, x0 + dx), (1 - dy, 1 + dy), **kwargs)
    else:
        raise ValueError("side must be 'left' or 'right'")


for axi, ax in enumerate(axs):
    sns.lineplot(ax=ax, legend=True, **common_kwargs)
    ax.set_xlim(*ranges[axi])
    is_left_most = axi == 0
    is_right_most = axi == len(axs) - 1
    if not is_right_most:
        ax.spines["right"].set_visible(False)
    if not is_left_most:
        ax.spines["left"].set_visible(False)
    if is_left_most:
        ax.set_ylabel("Tokens Per Second (All Clients, Current)")
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_ylabel("")
    if not is_right_most and ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_xlabel("")

    if not is_left_most:
        draw_break_marks(ax, side="left", size=0.012)

    if not is_right_most:
        draw_break_marks(ax, side="right", size=0.012)
    ticks = ax.get_xticks()
    ticks = [t for t in ticks if ranges[axi][0] <= t <= ranges[axi][1]]
    if not is_right_most and len(ticks) > 0:
        ticks = ticks[:-1]
    ax.set_xticks(ticks)
label_fs = plt.rcParams.get("axes.labelsize", 12)
plt.subplots_adjust(bottom=0.14)

fig.text(
    0.5, 0.06, "Seq Length",
    ha="center", va="center",
    fontsize=label_fs
)

plt.savefig(
    os.path.join(output_dir, "infer-scale.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)
analyzer.speed_info()

plt.figure(figsize=(6, 4), dpi=330)
analyzer.plot(
    name_to_kwargs=lambda name: {"scale": name.split("#")[0], "clients": name.split("#")[1]},
    hue="clients",
    style="scale",
    limit=65535,
    window=3
)
plt.legend().remove()
plt.xlabel("")
plt.xlim(500, 2000)
plt.ylim(50, 900)
plt.savefig(
    os.path.join(output_dir, "infer-scale-detail.png"),
    dpi=300,
    bbox_inches="tight"
)
