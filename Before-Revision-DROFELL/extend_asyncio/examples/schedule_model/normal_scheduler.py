from abc import ABC, abstractmethod
import asyncio
import collections
import heapq
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar

from concurrent.futures import ThreadPoolExecutor
from orbit import *
import extend_asyncio as asyncio


Ret = TypeVar("Ret")


class BlockingSubmitter(ABC):
    @abstractmethod
    async def submit(self, call: Callable[..., Ret], *args: Any, task_name: str) -> Ret: raise NotImplementedError


class NoneScheduler(asyncio.BlockingScheduler, BlockingSubmitter):
    def __init__(self) -> None:
        self.loop = asyncio.get_event_loop()

    async def submit(self, call: Callable[..., Ret], *args: Any, task_name: str) -> Ret:
        fut = asyncio.Future()

        def enclosure():
            ret = call(*args)
            fut.set_result(ret)
        self.loop.call_soon(enclosure)
        return await fut  # type:ignore

    def __bool__(self) -> bool: return False
    def __len__(self) -> int: return 0
    def run_once(self) -> None: raise NotImplementedError


class SeqFIFOBlockingScheduler(asyncio.BlockingScheduler, BlockingSubmitter):
    def __init__(self) -> None:
        self.blockings: collections.deque[Callable[[], None]] = collections.deque()

    async def submit(self, call: Callable[..., Ret], *args: Any, task_name: str) -> Ret:
        fut = asyncio.Future()

        def enclosure():
            ret = call(*args)
            fut.set_result(ret)
        self.blockings.append(enclosure)
        return await fut  # type:ignore

    def __bool__(self) -> bool:
        return len(self.blockings) != 0

    def __len__(self) -> int:
        return len(self.blockings)

    def run_once(self) -> None:
        task = self.blockings.popleft()
        task()


class MultiThreadsBlockingScheduler(asyncio.BlockingScheduler, BlockingSubmitter):
    def __init__(self, priority_control: bool = False, pool_size: int = -1) -> None:
        # 处理 -1 表示无限的情况
        if pool_size == -1:
            self.executor = ThreadPoolExecutor()
        else:
            self.executor = ThreadPoolExecutor(max_workers=pool_size)
        self.loop = asyncio.get_event_loop()
        self.priority_control = priority_control
        self.high_priority_waitor: Optional[asyncio.Event] = None
        # self.low_priority

    async def submit(self, call: Callable[..., Ret], *args: Any, task_name: str) -> Ret:
        if self.high_priority_waitor is not None and task_name == "low":
            await self.high_priority_waitor.wait()
        if task_name == "high" and self.priority_control:
            self.high_priority_waitor = asyncio.Event()
            ret = await self.loop.run_in_executor(self.executor, call, *args)
            self.high_priority_waitor.set()
            self.high_priority_waitor = None
            return ret
        return await self.loop.run_in_executor(self.executor, call, *args)

    def __bool__(self) -> bool: return False
    def __len__(self) -> int: return 0
    def run_once(self) -> None: raise NotImplementedError


class SeqPriorityBlockingScheduler(asyncio.BlockingScheduler, BlockingSubmitter):
    def __init__(self) -> None:
        self.blockings: List[Tuple[int, Callable[[], None]]] = []
        self._priority_names: Dict[str, int] = {}

    async def submit(self, call: Callable[..., Ret], *args: Any, task_name: str) -> Ret:
        fut = asyncio.Future()
        priority = 0
        if task_name == "low":
            priority = 1

        def enclosure():
            ret = call(*args)
            fut.set_result(ret)
        heapq.heappush(self.blockings, (priority, enclosure))
        return await fut  # type:ignore

    def __bool__(self) -> bool:
        return len(self.blockings) != 0

    def __len__(self) -> int:
        return len(self.blockings)

    def run_once(self) -> None:
        _, task = heapq.heappop(self.blockings)
        task()


def get_scheduler(implement: Literal["normal", "seq-fifo", "seq-priority", "multi-threading"]) -> BlockingSubmitter:
    loop = asyncio.get_event_loop()
    if implement == "normal":
        submitter = NoneScheduler()
        loop.set_blocking_scheduler(submitter)
    elif implement == "seq-fifo":
        submitter = SeqFIFOBlockingScheduler()
        loop.set_blocking_scheduler(submitter)
    elif implement == "seq-priority":
        submitter = SeqPriorityBlockingScheduler()
        loop.set_blocking_scheduler(submitter)
    elif implement == "multi-threading":
        submitter = MultiThreadsBlockingScheduler()
        loop.set_blocking_scheduler(submitter)
    elif implement == "multi-threading-priority":
        submitter = MultiThreadsBlockingScheduler(priority_control=True)
        loop.set_blocking_scheduler(submitter)
    else:
        raise ValueError(submitter)
    return submitter
