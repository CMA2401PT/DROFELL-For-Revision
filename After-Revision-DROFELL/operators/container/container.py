from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, List, Optional
import torch

from operators.container.run_context import RunContext
from .run_context import RetType, Modes, get_thread_context
from torch.profiler import profile, ProfilerActivity
import extend_asyncio as asyncio


class Stage:
    def __init__(self) -> None:
        self._stage_list: List[str] = []
        self._stage: str = "/"

    def __str__(self) -> str:
        return self._stage

    @property
    def full(self) -> str:
        return self._stage

    def push(self, stage_name: str) -> None:
        self._stage_list.append(stage_name)
        self._stage = "/"+"/".join(self._stage_list)

    def pop(self):
        self._stage_list.pop()
        self._stage = "/"+"/".join(self._stage_list)

    def step_name(self, step: str) -> str:
        if self._stage[-1] == "/":
            return self._stage+step
        return self._stage + "/"+step

    def __call__(self, stage_name: str):
        self.push(stage_name=stage_name)
        return self

    def __enter__(self):
        return None

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        self.pop()


class ContainerMode:
    def __init__(self, enter_mode: Callable[[], None], exit_mode: Callable[[], None]) -> None:
        self.enter_mode = enter_mode
        self.exit_mode = exit_mode

    def __enter__(self):
        self.enter_mode()

    def __exit__(self, exc_type, exc_val, exc_tb):  # type:ignore
        self.exit_mode()


class Container(ABC):
    def __init__(self) -> None:
        self.stage = Stage()

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    @abstractmethod
    def stream(self) -> torch.cuda.Stream:
        raise NotImplementedError

    @abstractmethod
    def set_mode(self, mode: Modes) -> None:
        raise NotImplementedError

    @abstractmethod
    def mode(self, mode: Modes) -> ContainerMode:
        raise NotImplementedError

    @abstractmethod
    async def step(self, step_name: str, step: Callable[..., RetType], **kwargs: Any) -> RetType:
        raise NotImplementedError

    @abstractmethod
    def sync_point(self) -> Awaitable[None]:
        raise NotImplementedError

    @abstractmethod
    async def data_device_to_cpu(self, data: torch.Tensor, wait: bool) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    async def data_cpu_to_device(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    async def slow_operate(self, step_name: str, step: Callable[..., RetType], **kwargs: Any) -> RetType:
        raise NotImplementedError

    @abstractmethod
    async def notify(self, info: str) -> None:
        raise NotImplementedError


class NonAsyncContainer(Container):
    def __init__(self, stream: torch.cuda.Stream) -> None:
        super().__init__()
        self._current_mode: Modes = "normal"
        self._run_env: RunContext = get_thread_context().make_env(stream)

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

    async def step(self, step_name: str, step: Callable[..., RetType], **kwargs: Any) -> RetType:
        return self._run_env(self._current_mode, step, **kwargs)

    def sync_point(self) -> Awaitable[None]:
        torch.cuda.synchronize(self.device)
        fut: asyncio.Future[None] = asyncio.Future()
        fut.set_result(None)
        return fut

    async def data_device_to_cpu(self, data: torch.Tensor, wait: bool) -> torch.Tensor:
        with torch.cuda.stream(self.stream):
            cpu_data = data.to(device="cpu", non_blocking=True)
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


class LeakDetectContainer(Container):
    def __init__(self, stream: torch.cuda.Stream) -> None:
        super().__init__()
        self._stream = stream
        self._current_mode: Modes = "normal"
        self._run_env = get_thread_context().make_env(stream)
        self.last_operate = ""
        self.prof: Optional[profile] = profile(activities=[ProfilerActivity.CUDA])

    @property
    def device(self) -> torch.device:
        return self._run_env.device

    @property
    def stream(self) -> torch.cuda.Stream:
        return self._stream

    def set_mode(self, mode: Modes) -> None:
        self._current_mode = mode

    def mode(self, mode: Modes) -> ContainerMode:
        orig_mode = self._current_mode

        def on_set():
            self._current_mode = mode

        def on_exit():
            self._current_mode = orig_mode
        return ContainerMode(on_set, on_exit)

    def _on_new_step(self, step_name: str):
        if self.last_operate == "":
            return
        assert self.prof is not None
        pos = f"{self.last_operate}->{step_name}"
        self.prof.stop()
        activities = self.prof.key_averages()
        activities = [e for e in activities if e.key != 'cudaDeviceSynchronize']
        if len(activities) == 0:
            return
        print(f"detect leak @ {pos}")
        for e in activities:
            print(f"\t{e.key}")
        raise ValueError(pos)

    def _on_step_finish(self, step_name: str):
        torch.cuda.synchronize()
        self.last_operate = step_name
        self.prof = profile(activities=[ProfilerActivity.CUDA])
        self.prof.start()

    async def step(self, step_name: str, step: Callable[..., RetType], **kwargs: Any) -> RetType:
        self._on_new_step(step_name)
        ret = self._run_env(self._current_mode, step, **kwargs)
        self._on_step_finish(step_name)
        return ret

    def sync_point(self) -> Awaitable[None]:
        ev = torch.cuda.Event(blocking=True)
        ev.record(self.stream)
        ev.synchronize()
        fut: asyncio.Future[None] = asyncio.Future()
        return fut

    async def data_device_to_cpu(self, data: torch.Tensor, wait: bool) -> torch.Tensor:
        self._on_new_step("data_device_to_cpu")
        with torch.cuda.stream(self.stream):
            cpu_data = data.to(device="cpu", non_blocking=True)
        self._on_step_finish("data_device_to_cpu")
        if wait:
            await self.sync_point()
        return cpu_data

    async def data_cpu_to_device(self, data: torch.Tensor) -> torch.Tensor:
        self._on_new_step("data_cpu_to_device")
        with torch.cuda.stream(self.stream):
            device_data = data.to(device=self.device, non_blocking=True)
        self._on_step_finish("data_cpu_to_device")
        return device_data

    async def slow_operate(self, step_name: str, step: Callable[..., RetType], **kwargs: Any) -> RetType:
        ret = step(**kwargs)
        return ret

    async def notify(self, info: str) -> None:
        pass


if __name__ == "__main__":
    import extend_asyncio as asyncio

    def some_cuda_operate():
        a = torch.zeros((1,), device="cuda")

    async def amain():
        loop = asyncio.get_event_loop()
        conatainer = LeakDetectContainer(torch.cuda.current_stream())
        await conatainer.step("operate1", some_cuda_operate)
        some_cuda_operate()
        await conatainer.step("operate2", some_cuda_operate)

        await conatainer.step("operate3", some_cuda_operate)
        await asyncio.Future()

    asyncio.config_impl("golang", True)
    asyncio.run(amain())
