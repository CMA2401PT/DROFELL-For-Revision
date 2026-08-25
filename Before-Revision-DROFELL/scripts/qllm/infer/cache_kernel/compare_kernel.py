# %%
import matplotlib.pyplot as plt
from analysis_utils import *
import os
import sys
import pickle as pkl
import json
sys.path.append(os.path.abspath(".."))
# %%
with open("kernel_vllm.pkl", "rb") as f:
    orig_report = pkl.load(f)
with open("kernel_dual_table.pkl", "rb") as f:
    our_report = pkl.load(f)
with open("kernel_dual_table_memory.pkl", "rb") as f:
    memory_report = pkl.load(f)

# %%
plt.figure(figsize=(10, 6), dpi=330)
plot_speed({
    ("1", "Origin"): orig_report['1x'].generating_time_record,
    ("2", "Origin"): orig_report['2x'].generating_time_record,
    ("4", "Origin"): orig_report['4x'].generating_time_record,
    ("8", "Origin"): orig_report['8x'].generating_time_record,
    ("1", "Our"): our_report['1x'].generating_time_record,
    ("2", "Our"): our_report['2x'].generating_time_record,
    ("4", "Our"): our_report['4x'].generating_time_record,
    ("8", "Our"): our_report['8x'].generating_time_record,
})
plt.plot()

# %%
print(len(memory_report["memory"].memory_record))
trace = TraceRecord(memory_report["memory"].memory_record[:12000])
plt.figure(figsize=(10, 6), dpi=330)
trace.plot()

# %%
print("orig")
for parallel in ["1x", "2x", "4x", "8x"]:
    records = orig_report[parallel].generating_time_record
    sp = compute_avg_speed(records, [100, 1000, 2000, -1])
    s = '\t'.join([f'{s:.2f}TPS' for s in sp])
    print(s)
print("="*80)
print("our")
for parallel in ["1x", "2x", "4x", "8x"]:
    records = our_report[parallel].generating_time_record
    sp = compute_avg_speed(records, [100, 1000, 2000, -1])
    s = '\t'.join([f'{s:.2f}TPS' for s in sp])
    print(s)

# %%
baseline = {
    '1x': orig_report['1x'].generating_time_record,
    '2x': orig_report['2x'].generating_time_record,
    '4x': orig_report['4x'].generating_time_record,
    '8x': orig_report['8x'].generating_time_record
}
with open("baseline.json", "w") as f:
    json.dump(baseline, f)
