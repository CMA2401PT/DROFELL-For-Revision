# %%
from analysis_utils import blocking_percentage, parallel_amount
import matplotlib.pyplot as plt
import json
from typing import Tuple, List
import os

from scripts.online.speed.analysis_utils import Analizer, gen_detail_plot, plot_timeline, TimeTracer

# %%
Record = Tuple[str, int]
StepRecords = List[Record]
TaskStepsRecords = List[List[Record]]
MultiTasksStepsRecords = List[TaskStepsRecords]

# %%
analyzer = Analizer(limit=20000)


def reduce_data_to_step_time(rec: List[List[List[Tuple[str, int]]]]):
    step_time: List[Tuple[int, int]] = []
    for task_i, task in enumerate(rec):
        for step in task:
            t = step[0][1]
            step_time.append((task_i, t))
    step_time = sorted(step_time, key=lambda x: x[1])
    return step_time


# %%
with open(os.path.join(os.path.dirname(__file__), "out.json"), "r") as f:
    data = json.load(f)["detail"]
    start = data[0][0][0][1]
    data = [[[(name, (stamp-start)) for (name, stamp) in step] for step in task] for task in data]
print(len(data))
analyzer.add_record(f"Test Parallel={len(data)}", reduce_data_to_step_time(data))


# %%
plt.figure(figsize=(10, 6), dpi=330)


def name_split(combine_name: str):
    method, parallel = combine_name.split(" Parallel=")
    return {"Method": method, "Parallel": parallel}


analyzer.plot(name_split, hue="Method", style="Parallel", window=1)
plt.plot()

# %%
analyzer.speed_info()

# %%
step_max = 128
fig = plot_timeline(
    data=[d[:min(len(d), step_max)] for d in data],
    fig=None,
    trim_events=(0, 0.2),
    trim_time=(0, 1.0),
    style_override=True,
    opacity_override=False,
    hidden_override=False,
    group_range_ms=0.1,
)
fig.show()

# %%
plt.figure(figsize=(15, 8), dpi=200)
gen_detail_plot(data[-1][:32])
plt.show()

# %%
blocking_percentage(data)

# %%
parallel_amount(data)

# %%
print(TimeTracer.records_to_detail_str(data[0]))

# %%
