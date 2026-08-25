from typing import Any, Callable, Coroutine, cast
import torch
from extend_asyncio.common import BlockingScheduler
from operators.container import Container, RetType
from .base import Operate, BaseContainer


class FastContainer(BaseContainer, Container, BlockingScheduler):
    def __init__(self, stream: torch.cuda.Stream) -> None:
        BaseContainer.__init__(self, stream)
        Container.__init__(self)
        BlockingScheduler.__init__(self)

    def with_grad(self, with_grad: bool) -> None:
        self._with_grad = with_grad

    async def step(self, step_name: str, step: Callable[..., RetType], **kwargs: Any) -> RetType:
        operate = Operate(self.stage.step_name(step_name), step, self._current_mode,  kwargs)
        self.blockings.append(operate)
        return cast(RetType, await operate.fut)

    async def sync_point(self) -> Coroutine[Any, Any, None]:
        ev = torch.cuda.Event(blocking=True)
        ev.record(torch.cuda.current_stream())

        async def wait():
            ev.synchronize()
        return wait()

    async def data_device_to_cpu(self, data: torch.Tensor) -> torch.Tensor:
        cpu_data = data.to(device="cpu", non_blocking=True)
        return cpu_data

    async def data_cpu_to_device(self, data: torch.Tensor) -> torch.Tensor:
        device_data = data.to(device=self.device, non_blocking=True)
        return device_data

    async def slow_operate(self, step_name: str, step: Callable[..., RetType], **kwargs: Any) -> RetType:
        ret = step(**kwargs)
        return ret

    async def notify(self, info: str) -> None:
        pass
