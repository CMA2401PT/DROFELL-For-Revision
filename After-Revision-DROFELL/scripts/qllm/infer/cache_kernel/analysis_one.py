# %%
import matplotlib.pyplot as plt
from analysis_utils import *
import os
import sys
import pickle as pkl
sys.path.append(os.path.abspath(".."))

# %%
target_file = "kernel_vllm_2k.pkl"
with open(target_file, "rb") as f:
    report = pkl.load(f)
# %%

plt.figure(figsize=(10, 6), dpi=330)
plot_speed({
    (k, "Report"): v.generating_time_record
    for k, v in report.items() if k != "memory"
})
plt.plot()

# %%
report["memory"].memory_record

# %%
if ("memory" in report) and (report["memory"].memory_record is not None):
    print(len(report["memory"].memory_record))
    trace = TraceRecord(report["memory"].memory_record[:12000])
    plt.figure(figsize=(10, 6), dpi=330)
    trace.plot()

# %%
print(target_file)
for k, v in report.items():
    if k == "memory":
        continue
    records = v.generating_time_record
    sp = compute_avg_speed(records, [128, 1024, 2048, -1])
    s = '\t'.join([f'{s:.2f}TPS' for s in sp])
    print(f"{k}: {s}")

# %%
