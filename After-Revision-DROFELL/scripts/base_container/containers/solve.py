import time
from typing import Any, Callable, Coroutine, Dict, List, Tuple, cast
import numpy as np
import torch
from extend_asyncio.common import BlockingScheduler
from operators.container import Container, RetType
from .base import Operate, BaseContainer


class SolveContainer(BaseContainer, Container, BlockingScheduler):
    def __init__(self, stream: torch.cuda.Stream, operate_load: Dict[str, Tuple[float, float]]) -> None:
        BaseContainer.__init__(self, stream)
        Container.__init__(self)
        BlockingScheduler.__init__(self)
        self.operate_load = operate_load

        self.estimate_start: int = 0
        self.estimated_cpu_load = 0
        self.estimated_cuda_load = 0
        self.estimate_count = 0

        self.estimate_alpha: float = 0.5327
        self.estimate_beta: float = 0.0191

        self.pressure_records: List[Tuple[int, int, int, int]] = []
        self.estimate_interval_ms = [(i*5)+10 for i in range(60)]
        self.warm_up = True

    def _reset_pressure(self):
        self.estimate_start = time.time_ns()
        self.estimated_cpu_load = 0
        self.estimated_cuda_load = 0
        self.estimate_count = 0

    def estimate_model(self):
        data = [(
            (float(y)/(10**6)), (float(cpu)/(10**6)), (float(cuda)/(10**6)), float(n)
        )for (y, cpu, cuda, n) in self.pressure_records]
        y_prime = []
        A = []
        for y, cpu, cuda, n in data:
            y_prime.append(y-cuda)
            A.append([cpu, n])
        w, residual, rank, s = np.linalg.lstsq(np.array(A), np.array(y_prime), rcond=None)
        alpha, beta = w
        return alpha, beta
        # self.estimate_alpha = alpha
        # self.estimate_beta = beta

    @property
    def estimate_load(self) -> int:
        return int(self.estimate_alpha*self.estimated_cpu_load) +\
            self.estimated_cuda_load +\
            self.estimate_count*int(self.estimate_beta*(10**6))

    def _actual_load(self, ctime: int = 0) -> int:
        if ctime == 0:
            ctime = time.time_ns()
        return ctime-self.estimate_start

    def _before_new_operate(self, step: str):
        mean_cpu, mean_cuda = self.operate_load[step]
        self.estimate_count += 1
        self.estimated_cpu_load += int(mean_cpu*(10**6))
        self.estimated_cuda_load += int(mean_cuda*(10**6))

    def _after_operate(self):
        ctime = time.time_ns()
        if ctime-self.estimate_start < self.estimate_interval_ms[0]*(10**6):
            return
        self.estimate_interval_ms.append(self.estimate_interval_ms[0])
        self.estimate_interval_ms = self.estimate_interval_ms[1:]
        torch.cuda.synchronize()
        actual_load = self._actual_load(0)
        cpu_load = self.estimated_cpu_load
        cuda_load = self.estimated_cuda_load
        estimate_count = self.estimate_count
        self.pressure_records.append(
            (actual_load, cpu_load, cuda_load, estimate_count))
        print(f"actual/estimate load: {actual_load/(10**6):.2f}/{self.estimate_load/(10**6):.2f}ms ({(actual_load/self.estimate_load)-1:.2%})")
        self._reset_pressure()

    async def step(self, step_name: str, step: Callable[..., RetType], **kwargs: Any) -> RetType:
        operate = Operate(self.stage.step_name(step_name), step, self._current_mode, kwargs)
        self.blockings.append(operate)
        return cast(RetType, await operate.fut)

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

    async def notify(self, info: str) -> None:
        if info == "train_step_finish":
            if self.warm_up == True:
                self.warm_up = False
                self._reset_pressure()
        return

    def run_once(self) -> None:
        operate = self.blockings.popleft()
        self._run_env(operate.mode)
        operate.run()
        if self.warm_up:
            return
        self._after_operate()
