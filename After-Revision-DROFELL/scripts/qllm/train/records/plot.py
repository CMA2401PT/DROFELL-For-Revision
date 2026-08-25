# %%
import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Dict

# %%


def extract_data(record_path: str):
    with open(record_path, "r")as f:
        raw_data = json.load(f)
        history_loss = raw_data["history"]
        step_time = raw_data["times"]
        events = raw_data["events"]
        samples_in_each_step = raw_data["samples_in_each_step"]
        records: List[Dict[str, Any]] = []
        for i, h in enumerate(history_loss):
            records.append({"step": (i//64*64), "data": "loss", "value": h})
        print(f"min loss:{min(history_loss)}")
        speed: List[float] = []
        for i in range(len(step_time)-4):
            d = step_time[i+4]-step_time[i]
            speed.append((4/d)*samples_in_each_step)
        for i, s in enumerate(speed):
            records.append({"step": (i//64*64), "data": "speed", "value": s})
        for (step, name, val) in events:
            if name == "memory":
                records.append({"step": (step//64*64), "data": "memoryGiB", "value": val["0"][0]/(2**30)})
                records.append({"step": (step//64*64), "data": "memory", "value": val["0"][0]})
        data = pd.DataFrame.from_records(records)
        return data


# %%
def plot_record(data: pd.DataFrame):
    plt.figure(figsize=(10, 6), dpi=330)
    sns.set_style("darkgrid")
    sns.set_theme(font="Times New Roman")
    # plt.figure(figsize=(8,5))
    sns.lineplot(data[data["data"] != "memoryGiB"], x="step", y="value", hue="data")
    # sns.lineplot(data[data["data"]=="speed"],x="step",y="value")
    plt.ylabel("Loss & Speed (Samples per Second)")
    plt.ylim(1, 9)
    plt.xlabel("Step")
    ax2 = plt.twinx()
    plt.ylim(4.3, 5.1)
    sns.set_style("white")
    sns.set_theme(font="Times New Roman")
    sns.lineplot(data[data["data"] == "memoryGiB"], x="step", y="value", color="green")
    plt.ylabel("Memory (GiB)")
    plt.xlim(-200, 12500)


# %%
data = extract_data("err-eps/record.json")
plot_record(data)
plt.show()

# %%
