import time
from typing import Any, Callable, Coroutine, Dict, List, Tuple, cast
import torch
from extend_asyncio.common import BlockingScheduler
import extend_asyncio as asyncio
from operators.container import Container, RetType
from .base import Operate, BaseContainer


class MeasureContainer(BaseContainer, Container, BlockingScheduler):
    def __init__(self, stream: torch.cuda.Stream) -> None:
        BaseContainer.__init__(self, stream)
        Container.__init__(self)
        BlockingScheduler.__init__(self)
        self.loop = asyncio.get_event_loop()

    async def step(self, step_name: str, step: Callable[..., RetType], **kwargs: Any) -> RetType:
        operate = Operate(self.stage.step_name(step_name), step, self._current_mode, kwargs)
        self.blockings.append(operate)
        ret = cast(RetType, await operate.fut)
        return ret

    async def sync_point(self) -> Coroutine[Any, Any, None]:
        ev = torch.cuda.Event(blocking=True)
        ev.record(torch.cuda.current_stream())

        async def wait():
            ev.synchronize()
        return wait()

    async def data_device_to_cpu(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    async def data_cpu_to_device(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    async def slow_operate(self, step_name: str, step: Callable[..., RetType], **kwargs: Any) -> RetType:
        raise NotImplementedError

    def compute_stats(self):
        from statistics import mean, stdev
        from collections import defaultdict
        grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"cpu": [], "cuda": [], "total": []})
        seq: List[Tuple[float, float]] = []
        for type_name, cpu_time, cuda_time in self.records:
            grouped[type_name]["cpu"].append(cpu_time/(10**6))
            grouped[type_name]["cuda"].append(cuda_time/(10**6))
            grouped[type_name]["total"].append((cpu_time+cuda_time)/(10**6))
            seq.append((cpu_time/(10**6), cuda_time/(10**6)))
        result: Dict[str, Tuple[float, float]] = {}
        for type_name, ops in grouped.items():
            cpu_list = ops["cpu"]
            cuda_list = ops["cuda"]
            # total_list = ops["total"]
            cpu_mean = mean(cpu_list)
            cpu_stdev = stdev(cpu_list) if len(cpu_list) > 1 else 0.0
            cuda_mean = mean(cuda_list)
            cuda_stdev = stdev(cuda_list) if len(cuda_list) > 1 else 0.0
            # total_mean = mean(total_list)
            # total_stdev = stdev(total_list) if len(total_list) > 1 else 0.0
            result[type_name] = (cpu_mean, cuda_mean)
        return result

    def run_once(self) -> None:
        operate = self.blockings.popleft()
        st = time.perf_counter_ns()
        self._run_env(operate.mode)
        operate.run()
        ct = time.perf_counter_ns()
        cpu_time = ct-st
        st = ct
        self.stream.synchronize()
        extra_time = time.perf_counter_ns()-st
        self.records.append((operate.step_name, cpu_time, extra_time))

    async def notify(self, info: str) -> None:
        pass
