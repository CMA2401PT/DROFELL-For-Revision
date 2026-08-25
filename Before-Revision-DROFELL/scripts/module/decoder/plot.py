import numpy as np
from typing import Any, Dict, List
from typing import Any, Dict, List, Optional
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from dataclasses import dataclass
import numpy as np
from dataclasses import dataclass
from operators.helpers.cuda_utils.memory import MemoryDeltaStatus


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
            plt.text(x-200, y/(2**20)+(10 if above else -100), f"{text}\n{y/(2**20):.2f}MiB", fontsize=9)
        forward_steps = len(self.forward.history)

        forward_memory_peak = self.forward.peak_memory
        forward_perisist_memory = self.forward.preserved_new_alloc
        middle = (forward_memory_peak+forward_perisist_memory)/2
        forward_memory_peak_step = self.forward.peak_step

        backward_memory_peak = self.backward.peak_memory
        backward_memoey_peak_step = self.backward.peak_step

        data = pd.DataFrame.from_records(self.dump_record())
        plt.axvline(forward_steps, color="grey", linestyle=":")
        if self.swapin is not None:
            plt.axvline(len(self.swapin.history), color="grey", linestyle=":")
        sns.lineplot(data=data, x="step", y="memory", hue="stage")

        plt.text(x=100, y=middle/(2**20)+80, s=f"Forward {self.forward_avg:.2f}ms", fontsize=10, color="blue")
        if self.swapout is not None:
            assert self.swapin is not None
            plt.text(x=len(self.swapin.history),
                     y=20, s=f"Swap In+Out {self.swap_avg:.2f}ms", fontsize=10, color="green")
        plt.text(x=len(self.backward.history)/2,
                 y=middle/(2**20)-180, s=f"Backward {self.backward_avg:.2f}ms", fontsize=10, color="red")
        # plt.axhline(middle/(2**20), color="grey", linestyle=":")

        plt.xlim(-10, len(self.backward.history)+10)
        plt.ylim(0, max(self.backward.history)/(2**20)+100)

        add_mark(forward_memory_peak_step, forward_memory_peak, "Forward Peak", True)
        add_mark(forward_steps, forward_perisist_memory, "Forward Persist", False)
        add_mark(backward_memoey_peak_step, backward_memory_peak, "Backward Peak", True, color="red")

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
    assert backward_tr.preserved_new_alloc == 0, f"memory remaining: {backward_tr.preserved_new_alloc/(2**20)}MiB, {backward_tr.preserved_alloc_frames}"
    backward_memory_peak = backward_tr.peak_memory
    backward_memoey_peak_step = backward_tr.peak_step

    def to_memory(x: int): return x/(2**20)
    print(f"forward peak: {to_memory(forward_memory_peak)} MiB @ {forward_memoey_peak_step}")
    print(f"forward residual: {to_memory(forward_perisist_memory)} MiB @ {forward_steps}")
    print(f"backward peak: {to_memory(backward_memory_peak)} MiB @ {backward_memoey_peak_step}")
    return AnalysisRet(
        forward=forward_tr,
        swapout=swapout_tr,
        swapin=swapin_tr,
        backward=backward_tr,
        forward_avg=float(np.mean(forward_time)/(10**6)),
        forward_std=float(np.std(forward_time)/(10**6)),
        swap_avg=float(np.mean(swap_time)/(10**6)) if (swap_time is not None and len(swap_time) > 0) else 0,
        swap_std=float(np.std(swap_time)/(10**6)) if (swap_time is not None and len(swap_time) > 0) else 0,
        backward_avg=float(np.mean(backward_time)/(10**6)),
        backward_std=float(np.std(backward_time)/(10**6)),
    )
