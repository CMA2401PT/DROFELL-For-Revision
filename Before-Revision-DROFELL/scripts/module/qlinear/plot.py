# %%
import numpy as np
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

# %%
print(os.getcwd())

# %%
with open("result.pkl", "rb") as f:
    records = pickle.load(f)
len(records)

# %%


data: DataFrame = pd.DataFrame.from_records(records)

# %%
plt.figure(figsize=(10, 5), dpi=330)
sns.set_theme(font="Times New Roman")
sns.lineplot(x="seq_len", y="time",
             style="kind", markers=True,
             hue="method",
             data=data[(data['dtype'] == "float16") & (data["kind"] == "forward")])
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Sequence Length")
plt.ylabel("Time (nano second)")

# %%
plt.figure(figsize=(10, 5), dpi=330)
sns.set_theme(font="Times New Roman")
sns.lineplot(x="seq_len", y="time",
             style="dtype",
             hue="method",
             markers=True,
             data=data[data["kind"] == "forward"])
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Sequence Length")
plt.ylabel("Time (nano second)")

# %%
plt.figure(figsize=(8.5, 3.1), dpi=330)
sns.set_theme(font="Times New Roman")
data['dtype_method'] = data['dtype'] + "_" + data['method']
select_data = data[data["kind"] == "forward+backward"]
select_data = select_data[select_data["seq_len"] > 16]
sns.barplot(x="seq_len", y="time",
            hue="dtype_method",
            data=select_data)
plt.yscale('log')
plt.xlabel("Sequence Length")
plt.ylabel("Time (nano second)")

# %%


grouped_record = {}
for rec in records:
    seq_len = rec["seq_len"]
    method = rec["method"]
    kind = rec["kind"]
    dtype = rec["dtype"]
    time = rec["time"]
    mark = f"{method},{kind},{dtype}"
    if not mark in grouped_record:
        grouped_record[mark] = {}
    if not seq_len in grouped_record[mark]:
        grouped_record[mark][seq_len] = []
    grouped_record[mark][seq_len].append(time)
for mark, records_of_seqs in grouped_record.items():
    print(mark)
    for seq_len, records in records_of_seqs.items():
        mean = np.mean(records)
        std = np.std(records)
        print(f"\t{seq_len} {mean/(10**3):.2f}({std/(10**3):.2f})µs")

# %%
