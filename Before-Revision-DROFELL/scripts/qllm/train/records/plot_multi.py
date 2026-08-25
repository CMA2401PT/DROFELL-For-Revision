# %%
import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Dict

# %%


def extract_records_with_name(record_path: str, name: str):
    with open(record_path, "r")as f:
        raw_data = json.load(f)
        history_loss = raw_data["history"]
        step_time = raw_data["times"]
        print(f"avg speed: {len(step_time)/(step_time[-1]-step_time[0])}")

        samples_in_each_step = raw_data["samples_in_each_step"]
        records: List[Dict[str, Any]] = []
        for i, h in enumerate(history_loss):
            records.append({"step": (i//64*64), "Data": "Loss", "value": h, "Method": name})
        print(f"min loss:{min(history_loss)}")
        speed: List[float] = []
        for i in range(len(step_time)-4):
            d = step_time[i+4]-step_time[i]
            speed.append((4/d)*samples_in_each_step)
        for i, s in enumerate(speed):
            records.append({"step": (i//64*64), "Data": "Speed (minibatch/s)", "value": s, "Method": name})
        return records


# %%
records =\
    extract_records_with_name("err-eps/record.json", "Float16") +\
    extract_records_with_name("bfloat16_start/record.json", "Bloat16+EPS") +\
    extract_records_with_name("bfloat16/record.json", "Bloat16")
data = pd.DataFrame.from_records(records)
fig = plt.figure(figsize=(10, 6), dpi=330)
sns.set_style("darkgrid")
sns.set_theme(font="Times New Roman")

# plt.figure(figsize=(8,5))
g = sns.lineplot(data, x="step", y="value", hue="Method", style="Data")
# sns.lineplot(data[data["data"]=="speed"],x="step",y="value")
plt.ylabel("Loss & Speed (Samples per Second)")
sns.move_legend(g, "lower right")
# plt.ylim(1, 9)
plt.xlabel("Step")
# ax2 = plt.twinx()
# plt.ylim(4.3, 5.1)
# sns.set_style("white")
# sns.set_theme(font="Times New Roman")
# sns.lineplot(data[data["data"] == "memoryGiB"], x="step", y="value", color="green")
# plt.ylabel("Memory (GiB)")
# plt.xlim(-200, 12500)
