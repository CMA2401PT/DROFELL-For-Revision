import time
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple, cast

import torch
from extend_asyncio.common import BlockingScheduler

import extend_asyncio as asyncio
from operators.helpers.cuda_async import CudaEventSyncer
from operators.container import Container, RetType
from .base import Operate, BaseContainer


class PressureModelling:
    def __init__(self, operate_load: Dict[str, Tuple[float, float]]) -> None:
        self.operate_load = self._compute_load(operate_load)
        self.reset_time = time.perf_counter_ns()
        self.progress = 0
        self.estimated_load = 0

    def _compute_load(self, operate_load: Dict[str, Tuple[float, float]]) -> Dict[str, int]:
        alpha = 0.5327
        beta = 0.0191
        tight = 0.9
        operate_estim = {
            k: int(((10**6) * ((mean_cpu*alpha+mean_cuda)+beta))*tight)
            for k, (mean_cpu, mean_cuda) in operate_load.items()
        }
        return operate_estim

    def reset(self):
        self.reset_time = time.perf_counter_ns()
        self.progress = 0
        self.estimated_load = 0

    def update_progress(self):
        self.progress = time.perf_counter_ns()-self.reset_time
        self.estimated_load = max(self.estimated_load, self.progress)

    def update_load(self, step: str):
        self.estimated_load += self.operate_load[step]

    def caliberate_esitmate(self, load: int):
        assert load >= 0
        self.estimated_load += load

    @property
    def estimate_finish_time(self) -> int:
        return self.estimated_load+self.reset_time

    @property
    def pressure(self) -> int:
        return self.estimated_load-self.progress


_MS = 1e6


class PressureReleaseContainer(BaseContainer, Container, BlockingScheduler):
    def __init__(self, stream: torch.cuda.Stream, operate_load: Dict[str, Tuple[float, float]]) -> None:
        BaseContainer.__init__(self, stream)
        Container.__init__(self)
        BlockingScheduler.__init__(self)
        self.loop = asyncio.get_event_loop()
        self.pressure_modelling = PressureModelling(operate_load)
        self.event_sync_period = int(0.5*_MS)
        self.event_sync_overhead = int(0.03*_MS)
        self.sleep_threas = int(0.1*_MS)
        self.sleep_error = int(0.5*_MS)

        self.sync_event: Optional[Tuple[int, CudaEventSyncer]] = None
        self.last_sync_time = 0

        # self.device_context = torch.device(self.device)
        # torch.cuda.set_stream(self.stream)
        # self.device_context.__enter__()

    def _reset(self):
        ct = time.perf_counter_ns()
        self.sync_event = None
        self.last_sync_time = ct
        self.pressure_modelling.reset()

    def _insert_ev(self):
        st = time.perf_counter_ns()
        self.sync_event = (self.pressure_modelling.estimate_finish_time, CudaEventSyncer(self.cuda_stream))
        et = time.perf_counter_ns()
        self.records.append(("sync", et-st, 0))

    async def _sync_ev(self, sync_event: Optional[Tuple[int, CudaEventSyncer]] = None):
        if sync_event is None:
            assert self.sync_event is not None
            sync_event = self.sync_event
            self.sync_event = None
        estimate_finish_time, ev = sync_event
        st = time.perf_counter_ns()
        await self.loop.delegate_c_poll_closure(ev)
        et = time.perf_counter_ns()
        self.records.append(("sync", 0, et-st))
        estimate_error = max(0, et-estimate_finish_time-self.event_sync_overhead)
        self.pressure_modelling.caliberate_esitmate(estimate_error)
        self.pressure_modelling.update_progress()
        self.last_sync_time = time.perf_counter_ns()

    async def _submit(self, step_name: str, step: Callable[..., RetType], kwargs: Dict[str, Any]) -> RetType:
        operate = Operate(step_name, step, self._current_mode, kwargs)
        self.blockings.append(operate)
        await operate.fut
        return cast(RetType, operate.ret)

    async def _sync(self):
        if self.sync_event is not None:
            event_estimate_finish = self.sync_event[0]+self.event_sync_overhead
            if event_estimate_finish < self.pressure_modelling.estimate_finish_time:
                await self._sync_ev()

        # insert ev?
        if self.sync_event is None:
            sync_delta = time.perf_counter_ns()-self.last_sync_time
            if sync_delta >= self.event_sync_period:
                self._insert_ev()

    async def _yield_ctl(self):
        self.pressure_modelling.update_progress()
        if self.pressure_modelling.pressure > (self.sleep_threas+self.sleep_error):
            st = time.perf_counter_ns()
            await asyncio.sleep((self.pressure_modelling.pressure-self.sleep_error)/1e9)
            et = time.perf_counter_ns()
            self.records.append(("sleep", 0, et-st))
            self.pressure_modelling.update_progress()

    async def step(self, step_name: str, step: Callable[..., RetType], **kwargs: Any) -> RetType:
        step_name = self.stage.step_name(step_name)
        ret = await self._submit(step_name, step, kwargs)
        self.pressure_modelling.update_load(step_name)
        self.pressure_modelling.update_progress()
        if self.sync_event is None:
            await self._yield_ctl()
        await self._sync()
        return ret

    async def sync_point(self) -> Coroutine[Any, Any, None]:
        st = time.perf_counter_ns()
        sync_event = (self.pressure_modelling.estimate_finish_time, CudaEventSyncer(self.cuda_stream))
        et = time.perf_counter_ns()
        self.records.append(("sync_point", et-st, 0))

        async def wait():
            await self._sync_ev(sync_event)
        return wait()

    async def data_device_to_cpu(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    async def data_cpu_to_device(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    async def slow_operate(self, step_name: str, step: Callable[..., RetType], **kwargs: Any) -> RetType:
        raise NotImplementedError

    async def notify(self, info: str) -> None:
        pass
