from collections import deque
import json
import time

import numpy as np
from extend_asyncio.common import BlockingScheduler
from operators.helpers import cuda_async
from .container import RetType, ContainerMode, get_thread_context, Container, Modes, RunContext
import extend_asyncio as asyncio
from typing import Any, Awaitable, Callable, Dict, Generic, List, Set, Tuple
import torch


class Operate(Generic[RetType]):
    def __init__(self,
                 step_name: str, step: Callable[..., RetType],
                 mode: Modes,
                 kwargs: Dict[str, Any]) -> None:
        self.step_name = step_name
        self.step = step
        self.kwargs = kwargs
        self.fut: asyncio.Future[RetType] = asyncio.Future()
        self.mode: Modes = mode
        self.ret = None

    def run(self):
        ret = self.step(**self.kwargs)
        self.ret = ret
        self.fut.set_result(ret)


_CPU = torch.device("cpu")


class _BaseContainer(Container):
    def __init__(self, stream: torch.cuda.Stream, sync: bool = False) -> None:
        super().__init__()
        self._current_mode: Modes = "normal"
        self._run_env: RunContext = get_thread_context().make_env(stream)
        self._cuda_stream_i = stream.cuda_stream
        # for statis
        self.records: List[Tuple[str, int, int]] = []
        self.record_enable: bool = False
        self.last_time = 0
        self.sync = sync

        self.blockings: deque[Callable[[], None]]

    @property
    def device(self) -> torch.device:
        return self._run_env.device

    @property
    def stream(self) -> torch.cuda.Stream:
        return self._run_env.stream

    def set_mode(self, mode: Modes) -> None:
        self._current_mode = mode

    def mode(self, mode: Modes) -> ContainerMode:
        orig_mode = self._current_mode

        def on_set():
            self._current_mode = mode

        def on_exit():
            self._current_mode = orig_mode
        return ContainerMode(on_set, on_exit)

    def compute_exp_base_waiting_metrics(self):
        seq: List[Tuple[float, float]] = []
        first: Set[str] = set()
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

    async def step(self, step_name: str, step: Callable[..., RetType], **kwargs: Any) -> RetType:
        step_name = self.stage.step_name(step=step_name)
        mode = self._current_mode
        fut: asyncio.Future[RetType] = asyncio.Future()

        def run():
            if not self.record_enable:
                ret = self._run_env(mode, step, **kwargs)
                fut.set_result(ret)
                return
            else:
                start_time = time.perf_counter_ns()
                if self.last_time != 0:
                    self.records.append(("other", start_time-self.last_time, 0))
                ret = self._run_env(mode, step, **kwargs)
                ctime = time.perf_counter_ns()
                self.records.append((step_name, ctime-start_time, 0))
                self.last_time = ctime
                fut.set_result(ret)
                return
        self.blockings.append(run)
        return await fut

    def sync_point(self) -> Awaitable[None]:
        if not self.sync:
            ev = cuda_async.CudaEventSyncer(self._cuda_stream_i)
            fut: Awaitable[None] = asyncio.get_event_loop().delegate_c_blocking_closure(ev)
            # fut = asyncio.Future()
            # fut.set_result(None)
            return fut
        else:
            torch.cuda.synchronize(self.device)
            fut_placeholder: asyncio.Future[None] = asyncio.Future()
            fut_placeholder.set_result(None)
            return fut_placeholder

    async def data_device_to_cpu(self, data: torch.Tensor, wait: bool) -> torch.Tensor:
        with torch.cuda.stream(self.stream):
            cpu_data = data.to(device=_CPU, non_blocking=True)
        if wait:
            await self.sync_point()
        return cpu_data

    async def data_cpu_to_device(self, data: torch.Tensor) -> torch.Tensor:
        '''
        move data from cpu to device without waiting
        need to all sync point to sync
        '''
        with torch.cuda.stream(self.stream):
            device_data = data.to(device=self.device, non_blocking=True)
        return device_data

    async def slow_operate(self, step_name: str, step: Callable[..., RetType], **kwargs: Any) -> RetType:
        ret = step(**kwargs)
        return ret

    async def notify(self, info: str) -> None:
        pass


class _BaseBlockingScheduler(BlockingScheduler):
    def __init__(self) -> None:
        self.blockings: deque[Callable[[], None]] = deque()

    def reset(self):
        assert len(self.blockings) == 0
        self.records: List[Tuple[str, int, int]] = []
        self.last_time = 0

    def __bool__(self) -> bool:
        return len(self.blockings) != 0

    def __len__(self) -> int:
        return len(self.blockings)

    def run_once(self) -> None:
        operate = self.blockings.popleft()
        operate()


class BaseContainer(_BaseContainer, _BaseBlockingScheduler, BlockingScheduler):
    def __init__(self, stream: torch.cuda.Stream) -> None:
        _BaseBlockingScheduler.__init__(self)
        _BaseContainer.__init__(self, stream)
        asyncio.get_running_loop().set_blocking_scheduler(scheduler=self)


class SharedBaseContainer(_BaseContainer):
    def __init__(self, stream: torch.cuda.Stream) -> None:
        blocking_scheduler = asyncio.get_running_loop().blocking_scheduler
        if blocking_scheduler is None:
            blocking_scheduler = _BaseBlockingScheduler()
            asyncio.get_running_loop().set_blocking_scheduler(scheduler=blocking_scheduler)
        else:
            assert isinstance(blocking_scheduler, _BaseBlockingScheduler)
        _BaseContainer.__init__(self, stream)
        self.blockings = blocking_scheduler.blockings
