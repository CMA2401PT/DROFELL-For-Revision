# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json

# %%
import numpy as np
from typing import Any, Dict, List, Tuple
from numpy import argmax
from typing import Any, Dict, List, Set
import numpy as np


class TraceRecord:
    def __init__(self, traces: List[Any]) -> None:
        self.free_before, self.current, self.history =\
            self.analyzed_trace(traces)
        self.data_block = self.gen_block_history(traces)
        self.peak_memory = max(self.history)
        self.peak_step = int(argmax(self.history))

    @staticmethod
    def analyzed_trace(traces: List[Any]) -> Tuple[int, int, List[int]]:
        current: Dict[int, int] = {}
        free_before: Dict[int, int] = {}
        history: List[int] = []
        for trace in traces:
            action = trace["action"]
            addr = trace["addr"]
            size = trace["size"]
            if action == 'alloc':
                current[addr] = size
                history.append(sum(current.values()))
            if action == "free_requested":
                continue
            if action == "free_completed":
                if not addr in current:
                    free_before[addr] = size
                else:
                    del current[addr]
                    history.append(sum(current.values()))
            if action == "snapshot":
                continue
            assert "frames" in trace, trace
            assert len(trace["frames"])
        return sum(free_before.values()), sum(current.values()), history

    @staticmethod
    def gen_block_history(traces: List[Any]):
        occoured: Set[int] = set()
        steps = 0
        for trace in traces:
            addr = trace["addr"]
            action = trace["action"]
            if action == 'alloc':
                occoured.add(addr)
                steps += 1
            elif action == "free_completed":
                steps += 1

        addrs = [a for a in occoured]
        addrs = sorted(addrs)
        addr_mapping = {a: i for i, a in enumerate(addrs)}
        data_block = np.zeros((len(addrs), steps), dtype=np.int64)

        current: Dict[int, int] = {}
        step_i = 0
        for (trace) in traces:
            action = trace["action"]
            addr = trace["addr"]
            size = trace["size"]
            if action not in ['alloc', 'free_completed']:
                continue

            if action == 'alloc':
                current[addr] = size
                for addr, size in current.items():
                    data_block[addr_mapping[addr], step_i] = size
            elif action == "free_completed":
                if addr in current:
                    del current[addr]

            for addr, size in current.items():
                data_block[addr_mapping[addr], step_i] = size/(2**20)
            step_i += 1
        return data_block

    def plot(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        _, steps = self.data_block.shape
        # set seaborn style
        sns.set_theme(context='paper', style='whitegrid', font="Times New Roman")
        plt.rcParams["patch.force_edgecolor"] = False
        x = range(steps)
        y = self.data_block/(2**10)
        plt.ylabel("Memory (KiB)")
        plt.xlabel("Operate")
        # Plot
        plt.stackplot(x, y)
        plt.xlim(-10, steps+10)


# %%
with open("single.json", "r") as f:
    single_record = json.load(f)
single_record.keys()

# %%
memory_record = TraceRecord(single_record['memory'])
plt.figure(figsize=(10, 6), dpi=330)
memory_record.plot()
plt.show()

# %%
with open("multi.json", "r") as f:
    multi_record = json.load(f)
multi_record.keys()

# %%
memory_record = TraceRecord(multi_record['memory'])
plt.figure(figsize=(10, 6), dpi=330)
memory_record.plot()
plt.show()

# %%
pd_records = []
for k, v in single_record.items():
    if k == "memory":
        continue
    scale, parallel = k.split(" ")
    print("single", scale, parallel)
    if parallel == "2":
        continue
    for ts, dt in v:
        pd_records.append({"Scale (User+Prompt)": scale, "method": f"Single", "TPS": ts/dt})
for k, v in multi_record.items():
    if k == "memory":
        continue
    scale, parallel = k.split(" ")
    print("multi", scale, parallel)
    if parallel == "2":
        continue
    for ts, dt in v:
        pd_records.append({"Scale (User+Prompt)": scale, "method": f"Multi Stream", "TPS": ts/dt})
data = pd.DataFrame.from_records(pd_records)
plt.figure(figsize=(6, 4), dpi=330)
plt.ylim(8200, 9500)
sns.barplot(data=data, x="Scale (User+Prompt)", y="TPS", hue="method")
plt.show()

# %%
data = pd.DataFrame.from_records(pd_records)
print(data)

# %%
sns.barplot(data=data, x="Scale (User+Prompt)", y="TPS", hue="method")

# %%
