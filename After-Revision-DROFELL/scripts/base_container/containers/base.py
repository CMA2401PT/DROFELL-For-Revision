from collections import deque
import json
import time
from typing import Any, Callable, Dict, List, Literal, Tuple
import numpy as np
import torch
from operators.container import RetType, ContainerMode, get_thread_context
import extend_asyncio as asyncio


class Operate:
    def __init__(self,
                 step_name: str, step: Callable[..., RetType],
                 mode: Literal["inference", "no_grad", "normal"],
                 kwargs: Dict[str, Any]) -> None:
        self.step_name = step_name
        self.step = step
        self.kwargs = kwargs
        self.fut = asyncio.Future()
        self.mode: Literal["inference", "no_grad", "normal"] = mode
        self.ret = None

    def run(self):
        ret = self.step(**self.kwargs)
        self.ret = ret
        self.fut.set_result(ret)


class BaseContainer:
    def __init__(self, stream: torch.cuda.Stream) -> None:
        self.records: List[Tuple[str, int, int]] = []
        self.blockings: deque[Operate] = deque()
        self.stream = stream
        self.cuda_stream = stream.cuda_stream
        self._current_mode: Literal["inference", "no_grad", "normal"] = "normal"
        self._run_env = get_thread_context().make_env(stream)
        self.last_time = 0

    @property
    def device(self) -> torch.device:
        return self._run_env.device

    def reset(self):
        self.records: List[Tuple[str, int, int]] = []
        self.blockings: deque[Operate] = deque()
        self.last_time = 0

    def set_mode(self, mode: Literal["inference", "no_grad", "normal"]):
        self._current_mode = mode

    def mode(self, mode: Literal["inference", "no_grad", "normal"]) -> ContainerMode:
        orig_mode = self._current_mode

        def on_set():
            self._current_mode = mode

        def on_exit():
            self._current_mode = orig_mode
        return ContainerMode(on_set, on_exit)

    def __bool__(self) -> bool:
        return len(self.blockings) != 0

    def __len__(self) -> int:
        return len(self.blockings)

    def run_once(self) -> None:
        operate = self.blockings.popleft()
        start_time = time.perf_counter_ns()
        if self.last_time != 0:
            self.records.append(("other", start_time-self.last_time, 0))
        self._run_env(operate.mode, operate.run)
        ctime = time.perf_counter_ns()
        self.records.append((operate.step_name, ctime-start_time, 0))
        self.last_time = ctime
        operate.fut.set_result(None)

    def compute_exp_base_waiting_metrics(self):
        seq: List[Tuple[float, float]] = []
        first = set()
        for type_name, cpu_time, cuda_time in self.records:
            if not type_name in first:
                first.add(type_name)
                continue
            seq.append((cpu_time/1e6, cuda_time/1e6))
        total_time = sum(w + i for w, i in seq)
        if total_time == 0:
            return 0, 0, 0  # avoid division by zero

        # Average: weighted average over busy periods
        weighted_sum = sum((w * (w / 2)) for w, _ in seq)  # average wait in each busy segment
        average_wait = weighted_sum / total_time

        # Worst-case: maximum duration of any busy period
        worst_wait = np.percentile([w for w, _ in seq], 99.85)
        busy_percentage = (sum(w for w, _ in seq)/total_time)
        return average_wait, worst_wait, busy_percentage

    def dump_profile(self, profile_file: str):
        average_wait, worst_wait, busy_percentage = self.compute_exp_base_waiting_metrics()
        print(f"\tAverage Actual wait: {average_wait:.4f}ms")
        print(f"\tWorst-case Actual wait: {worst_wait:.4f}ms")
        print(f"\tBusy Percentage: {busy_percentage:.2%}")
        with open(profile_file, "w") as f:
            json.dump(self.records, f)
