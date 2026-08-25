# %%
from typing import List
import numpy as np
from dataclasses import dataclass
import pandas as pd
import seaborn as sns
from typing import Any, Dict, List, Optional
import os
import pickle
import numpy as np
from typing import Any, Dict, List, Tuple

from matplotlib import pyplot as plt
import seaborn as sns

from operators.helpers.cuda_utils.memory import MemoryDeltaStatus
sns.set_theme(font="Times New Roman")

# %%


@dataclass
class AnalysisRet:
    forward: MemoryDeltaStatus
    backward: MemoryDeltaStatus
    swapout: Optional[MemoryDeltaStatus]
    swapin: Optional[MemoryDeltaStatus]
    forward_avg: float
    forward_std: float
    swap_avg: float
    swap_std: float
    backward_avg: float
    backward_std: float

    def dump_record(self) -> List[Dict[str, Any]]:
        forward: List[Dict[str, Any]] = [
            {
                "stage": "forward",
                "step": i+1,
                "memory": v/(2**20)
            }
            for i, v in enumerate(self.forward.history)
        ]
        start = len(self.forward.history)
        swapout: List[Dict[str, Any]] = []
        swapin: List[Dict[str, Any]] = []
        if self.swapout is not None:
            swapout = [
                {
                    "stage": "swap out",
                    "step": i+1,
                    "memory": v/(2**20)
                }
                for i, v in enumerate(self.swapout.history) if i > start
            ]
            start = len(self.swapout.history)-1
        if self.swapin is not None:
            swapin = [
                {
                    "stage": "swap in",
                    "step": i+1,
                    "memory": v/(2**20)
                }
                for i, v in enumerate(self.swapin.history) if i > start
            ]
            start = len(self.swapin.history)-2
        backward: List[Dict[str, Any]] = [
            {
                "stage": "backward",
                "step": i,
                "memory": v/(2**20)
            }
            for i, v in enumerate(self.backward.history) if i > start
        ]
        return [{"stage": "forward", "step": 0, "memory": 0}]+forward+swapout+swapin+backward

    def plot(self, adjust: Dict[str, Any]):
        def add_mark(x: int, y: int, text: str, above: bool, color: str = "blue"):
            plt.axhline(
                y/(2**20)+(1 if above else -1),
                (x/len(self.backward.history))-0.05, (x/len(self.backward.history))+0.05,
                color=color
            )
            plt.text((x-200) if above else (x+200), y/(2**20)+(10 if above else -60), f"{text}\n{y/(2**20):.2f}MiB", fontsize=9)
        forward_steps = len(self.forward.history)

        forward_memory_peak = self.forward.peak_memory
        forward_perisist_memory = self.forward.preserved_new_alloc
        middle = (forward_memory_peak+forward_perisist_memory)/2
        forward_memory_peak_step = self.forward.peak_step

        backward_memory_peak = self.backward.peak_memory
        backward_memoey_peak_step = self.backward.peak_step

        data = pd.DataFrame.from_records(self.dump_record())
        plt.axvline(forward_steps, color="grey", linestyle=":")
        plt.axvline(len(self.swapin.history), color="grey", linestyle=":")
        sns.lineplot(data=data, x="step", y="memory", hue="stage")

        # plt.text(x=100, y=middle/(2**20)+80, s=f"Forward {self.forward_avg:.2f}ms", fontsize=10, color="blue")
        # if self.swapout is not None:
        #     plt.text(x=len(self.swapin.history)+20,
        #          y=80, s=f"Swap In+Out {self.swap_avg:.2f}ms", fontsize=10, color="green")
        # plt.text(x=len(self.backward.history)/2,
        #          y=middle/(2**20)-180, s=f"Backward {self.backward_avg:.2f}ms", fontsize=10, color="red")
        # plt.axhline(middle/(2**20), color="grey", linestyle=":")

        plt.xlim(-10, len(self.backward.history)+10)
        plt.ylim(0, max(self.backward.history)/(2**20)+100)

        # add_mark(forward_memory_peak_step, forward_memory_peak, "Forward Peak", True)
        # add_mark(forward_steps, forward_perisist_memory, "Forward Persist", False)
        # add_mark(backward_memoey_peak_step, backward_memory_peak, "Backward Peak", True, color="red")

        plt.xlabel("Operate")
        plt.ylabel("Memory (MiB)")


def analysis_forward_backward_memory(
    forward_trace: List[Any], backward_trace: List[Any],
    swap_out_trace: Optional[List[Any]],
    swap_in_trace: Optional[List[Any]],
    forward_time: List[int], swap_time: List[int], backward_time: List[int]
):
    forward_tr = MemoryDeltaStatus(forward_trace)
    assert forward_tr.free_before == 0
    swapout_tr = MemoryDeltaStatus(swap_out_trace) if swap_out_trace is not None else None
    swapin_tr = MemoryDeltaStatus(swap_in_trace) if swap_in_trace is not None else None
    forward_memory_peak = forward_tr.peak_memory
    forward_memoey_peak_step = forward_tr.peak_step
    forward_steps = len(forward_tr.history)
    forward_perisist_memory = forward_tr.preserved_new_alloc
    backward_tr = MemoryDeltaStatus(backward_trace)
    # assert backward_tr.current == 0, backward_tr.current
    backward_memory_peak = backward_tr.peak_memory
    backward_memoey_peak_step = backward_tr.peak_step

    def to_memory(x): return x/(2**20)
    print(f"forward peak: {to_memory(forward_memory_peak)} MiB @ {forward_memoey_peak_step}")
    if swapout_tr is not None:
        print(f"after swap out: {to_memory(swapout_tr.preserved_new_alloc)} MiB")
    print(f"forward residual: {to_memory(forward_perisist_memory)} MiB @ {forward_steps}")
    print(f"backward peak: {to_memory(backward_memory_peak)} MiB @ {backward_memoey_peak_step}")
    print(f"backward: {to_memory(backward_tr.preserved_new_alloc)} MiB")
    return AnalysisRet(
        forward=forward_tr,
        swapout=swapout_tr,
        swapin=swapin_tr,
        backward=backward_tr,
        forward_avg=float(np.mean(forward_time)/(10**6)),
        forward_std=float(np.std(forward_time)/(10**6)),
        swap_avg=float(np.mean(swap_time)/(10**6)),
        swap_std=float(np.std(swap_time)/(10**6)),
        backward_avg=float(np.mean(backward_time)/(10**6)),
        backward_std=float(np.std(backward_time)/(10**6)),
    )


# %%
def config_renamer(config_name: str):
    alt_name = ""
    if "user_grad" in config_name:
        alt_name = "user:grad"
    if "user_seed" in config_name:
        alt_name = "user:seed"
    if "base_each" in config_name:
        alt_name += "+base:each"
    if "base_grad" in config_name:
        alt_name += "+base:grad"
    if "base_first.grad" in config_name:
        alt_name += "+base:first-grad"
    if "base_first.2pass" in config_name:
        alt_name += "+base:first-2pass"
    return alt_name


for config_name in sorted(os.listdir("results")):
    alt_name = config_renamer(config_name)
    print(alt_name, end="\t")
    with open(f"results/{config_name}/record.pkl", "rb") as f:
        snapshot = pickle.load(f)
        forward_time = snapshot["forward_time"]
        swap_time = snapshot["swap_time"]
        backward_time = snapshot["backward_time"]
        forward_avg = float(np.mean(forward_time)/(10**6))
        forward_std = float(np.std(forward_time)/(10**6))
        swap_avg = float(np.mean(swap_time)/(10**6))
        swap_std = float(np.std(swap_time)/(10**6))
        backward_avg = float(np.mean(backward_time)/(10**6))
        backward_std = float(np.std(backward_time)/(10**6))
        print(f"{forward_avg:.2f}±{forward_std:.2f}ms", end="\t")
        print(f"{swap_avg:.2f}±{swap_std:.2f}ms", end="\t")
        print(f"{backward_avg:.2f}±{backward_std:.2f}ms")

for config_name in sorted(os.listdir("results")):
    alt_name = config_renamer(config_name)
    print(alt_name, end="\t")
    with open(f"results/{config_name}/record.pkl", "rb") as f:
        snapshot = pickle.load(f)
        forward_trace = snapshot["forward_memory"]['device_traces'][0]
        swap_out_trace = snapshot["swap_out_memory"]['device_traces'][0] if snapshot["swap_out_memory"] is not None else None
        swap_in_trace = snapshot["swap_in_memory"]['device_traces'][0] if snapshot["swap_in_memory"] is not None else None
        backward_trace = snapshot["backward_memory"]['device_traces'][0]
        forward_tr = MemoryDeltaStatus(forward_trace)
        assert forward_tr.free_before == 0
        swapout_tr = MemoryDeltaStatus(swap_out_trace) if swap_out_trace is not None else None
        swapin_tr = MemoryDeltaStatus(swap_in_trace) if swap_in_trace is not None else None
        forward_memory_peak = forward_tr.peak_memory
        forward_memoey_peak_step = forward_tr.peak_step
        forward_steps = len(forward_tr.history)
        forward_perisist_memory = forward_tr.preserved_new_alloc
        backward_tr = MemoryDeltaStatus(backward_trace)
        # assert backward_tr.current == 0, backward_tr.current
        backward_memory_peak = backward_tr.peak_memory
        backward_memoey_peak_step = backward_tr.peak_step

        def to_memory(x): return x/(2**20)
        print(f"{to_memory(forward_memory_peak):.2f}MiB", end="\t")
        if swapout_tr is not None:
            print(f"{to_memory(swapout_tr.preserved_new_alloc):.2f}MiB", end="\t")
        print(f"{to_memory(forward_perisist_memory):.2f}MiB", end="\t")
        print(f"{to_memory(backward_memory_peak):.2f}MiB")
        # print(f"{to_memory(backward_tr.current)}MiB")


# %%
for config_name in os.listdir("results"):
    print(config_name)
    with open(f"results/{config_name}/record.pkl", "rb") as f:
        snapshot = pickle.load(f)
        ret = analysis_forward_backward_memory(
            forward_trace=snapshot["forward_memory"]['device_traces'][0],
            swap_out_trace=snapshot["swap_out_memory"]['device_traces'][0] if snapshot["swap_out_memory"] is not None else None,
            swap_in_trace=snapshot["swap_in_memory"]['device_traces'][0] if snapshot["swap_in_memory"] is not None else None,
            backward_trace=snapshot["backward_memory"]['device_traces'][0],
            forward_time=snapshot["forward_time"],
            swap_time=snapshot["swap_time"],
            backward_time=snapshot["backward_time"],
        )
        sns.set_theme(font="Times New Roman")
        plt.figure(figsize=(8.5, 3.1), dpi=330)
        ret.plot({})
        plt.tight_layout()
        os.makedirs("./analysis_out", exist_ok=True)
        plt.legend(loc="upper left")
        plt.savefig(f"./analysis_out/{config_name}_no_hint.png")
        sns.set_theme(font="Times New Roman")
        plt.figure(figsize=(8.5, 3.1), dpi=330)
        plt.legend(loc="upper left")
        ret.backward.plot()
        plt.savefig(f"./analysis_out/{config_name}_memory_no_hint.png")

    print()
with open("results/quant_user_seed_base_first.2pass/record.pkl", "rb") as f:
    snapshot = pickle.load(f)

# %%
current: Dict[int, Tuple[int, List[Any]]] = {}
free_before: Dict[int, int] = {}
for trace in snapshot["swap_out_memory"]['device_traces'][0]:
    action = trace["action"]
    addr = trace["addr"]
    size = trace["size"]
    if action == 'alloc':
        current[addr] = (size, trace["frames"])
    if action == "free_requested":
        continue
    if action == "free_completed":
        if not addr in current:
            free_before[addr] = size
        else:
            del current[addr]
    if action == "snapshot":
        continue
    assert "frames" in trace, trace
    assert len(trace["frames"])

# %%
for s, stack in current.values():
    print(s)
    print(stack)

# %%

record = MemoryDeltaStatus(snapshot["backward_memory"]['device_traces'][0])
data = record.gen_block_history(snapshot["backward_memory"]['device_traces'][0])

# %%
record.plot()
plt.show()

# %%
sns.reset_orig()
orig = plt.rcParams.copy()
sns.set_theme(context='paper', style='white')
for k, v in plt.rcParams.items():
    if v != orig[k]:
        print(f"plt.rcParams[\"{k}\"]={orig[k]}")

# %%


# %%
data.shape

# %%
data[0].shape

# %%
