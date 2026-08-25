# %%
import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Dict, Tuple
import numpy as np

# %%
record_files: Dict[Tuple[str, int], str] = {
    ("results/inference/7B_1_retrieve_1e2/result.json", 0): "drop_retrieve",
    ("results/inference/7B_1_no_retrieve_1e2/result.json", 0): "drop_no_retrieve",
}

records: List[Dict[str, Any]] = []
acc_max, delta_max = 0, 0
for (record_file, trace), name in record_files.items():
    with open(record_file, "r")as f:
        step_time = json.load(f)
        step_time = [
            t
            for trace_i, t in step_time
            if trace_i == trace
        ]
        for i, t in enumerate(step_time[1:]):
            acc_max = max((t-step_time[0])/(10**9), acc_max)
            delta_max = max((t-step_time[i])/(10**9), delta_max)
            records.append({"step": i, "time": (t-step_time[0])/(10**9), "Case": name, "Data": "Accumulate"})
            records.append({"step": i, "time": (t-step_time[i])/(10**9)*100, "Case": name, "Data": "Delta"})


sns.set_theme(
    style="darkgrid",
    font="Times New Roman",
    rc={
        "figure.dpi": 330,
        "savefig.dpi": 330
    }
)

data = pd.DataFrame.from_records(records)

fig, ax = plt.subplots(figsize=(10, 6), dpi=330)

left_data = data[data["Data"] == "Accumulate"]

sns.lineplot(
    data=left_data,
    x="step",
    y="time",
    hue="Case",
    palette="Set2",
    ax=ax,
    linewidth=2.0
)

ax.set_xlabel("Step")
ax.set_ylabel("Accumulate Time (Million Seconds)")
ax.legend().remove()

ax.grid(True)
ax.set_axisbelow(True)
ax2 = ax.twinx()
ax2.grid(False)

right_data = data[data["Data"] == "Delta"]

style_vals = right_data["Data"].unique()
dash_map = {v: (4, 2) for v in style_vals}

sns.lineplot(
    data=right_data,
    x="step",
    y="time",
    hue="Case",
    style="Data",
    palette="Set2",
    dashes=dash_map,
    ax=ax2,
    linewidth=2.0
)

ax2.set_ylabel("Delta Time (Million Seconds)")


def autoscale_from_series(ax_obj, series, margin=0.12):
    if len(series) == 0:
        return
    ymin = np.nanmin(series)
    ymax = np.nanmax(series)
    if ymin == ymax:
        delta = abs(ymax) * 0.05 if ymax != 0 else 1.0
        ymin -= delta
        ymax += delta
    span = ymax - ymin
    ax_obj.set_ylim(ymin - span * margin, ymax + span * margin)


autoscale_from_series(ax, left_data["time"])
autoscale_from_series(ax2, right_data["time"])

plt.tight_layout()
plt.savefig("analysis_infer_recover.png", dpi=330)

# %%
