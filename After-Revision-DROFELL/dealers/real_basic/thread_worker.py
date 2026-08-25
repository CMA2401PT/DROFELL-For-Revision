import threading
from typing import Any, Awaitable, Callable, Coroutine, Generic, List, Optional, cast

import torch
from ..common.brain import OperateType
import extend_asyncio as asyncio
from queue import Queue

from operators.container.base_container import BaseContainer
from operators.container import Container


class ThreadOperate(Generic[OperateType]):
    def __init__(self,
                 op: OperateType,
                 loop: asyncio.Loop,
                 handle_fn: Callable[[OperateType, Container], Coroutine[Any, Any, None]],
                 on_finish_cb: Callable[[], None],
                 thread_fut: Optional[asyncio.Future[None]] = None
                 ) -> None:
        self.base_op = op
        self.loop = loop
        self.sync: Optional[Awaitable[None]]
        self.on_finish_cb = on_finish_cb
        self.handle_fn = handle_fn
        self.thread_fut = thread_fut

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_op, name)

    def set_result(self, ret: Any, sp: Optional[Awaitable[None]] = None):
        def _move_thread_and_set(ret: Any):
            self.base_op.set_result(ret, None)
            self.on_finish_cb()
        if sp is None:
            self.loop.call_soon_threadsafe(_move_thread_and_set, ret)

            return
        else:
            async def sync_and_set():
                await sp
                self.loop.call_soon_threadsafe(_move_thread_and_set, ret)
            self.sync = sync_and_set()
            return

    async def run(self, container: Container):
        await self.handle_fn(cast(OperateType, self), container)
        if self.sync is not None:
            await self.sync

        if self.thread_fut is not None:
            self.loop.call_soon_threadsafe(self.thread_fut.set_result, None)


def _new_stream() -> torch.cuda.Stream:
    return torch.cuda.Stream()  # type:ignore


class ThreadTaskRunner:
    def __init__(self,
                 max_workers: int = 4,
                 get_stream: Callable[[], torch.cuda.Stream] = _new_stream
                 ) -> None:
        self._ctrl_loop = asyncio.get_running_loop()
        self.queue: Queue[ThreadOperate[Any]] = Queue(-1)
        self.threads: List[threading.Thread] = []
        self.get_stream = get_stream
        self.max_workers = max_workers
        self.idle_workers = 0

    def _new_worker(self):
        stream = self.get_stream()
        t = threading.Thread(target=lambda: asyncio.run(self._working_thread_coro(stream)))
        self.threads.append(t)
        self.idle_workers += 1
        t.start()

    def _lazy_new_worker(self):
        if self.idle_workers > 0:
            return
        if len(self.threads) < self.max_workers:
            self._new_worker()

    async def _working_thread_coro(self, stream: torch.cuda.Stream):
        container: Container = BaseContainer(stream)
        while True:
            op = self.queue.get()
            self.idle_workers -= 1
            await op.run(container)
            self.idle_workers += 1

    def run_no_wait(self,
                    op: OperateType,
                    handle_fn: Callable[[OperateType, Container], Coroutine[Any, Any, None]],
                    on_finish_cb: Callable[[], None],
                    ) -> Any:
        top = ThreadOperate(op, self._ctrl_loop, handle_fn, on_finish_cb)
        self._lazy_new_worker()
        self.queue.put_nowait(top)

    async def run_wait(self,
                       op: OperateType,
                       handle_fn: Callable[[OperateType, Container], Coroutine[Any, Any, None]],
                       on_finish_cb: Callable[[], None],
                       ):
        fut: asyncio.Future[None] = asyncio.Future()
        top = ThreadOperate(op, self._ctrl_loop, handle_fn, on_finish_cb, fut)
        self._lazy_new_worker()
        self.queue.put_nowait(top)
        await fut


class AsyncOperate(Generic[OperateType]):
    def __init__(self,
                 op: OperateType,
                 handle_fn: Callable[[OperateType, Container], Coroutine[Any, Any, None]],
                 on_finish_cb: Callable[[], None],
                 ) -> None:
        self.base_op = op
        self.handle_fn = handle_fn
        self.on_finish_cb = on_finish_cb

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_op, name)

    async def run(self, container: Container):
        await self.handle_fn(cast(OperateType, self), container)
        self.on_finish_cb()


class AsyncTaskRunner:
    def __init__(self, container: Container) -> None:
        self.queue: asyncio.Queue[AsyncOperate[Any]] = asyncio.Queue()
        self.task = asyncio.create_task(self.loop())
        self.container = container

    async def loop(self):
        while True:
            op = await self.queue.get()
            await op.run(self.container)

    def run_no_wait(
        self,
        op: OperateType,
        handle_fn: Callable[[OperateType, Container], Coroutine[Any, Any, None]],
        on_finish_cb: Callable[[], None],
    ) -> Any:
        self.queue.put_nowait(AsyncOperate(op, handle_fn, on_finish_cb))

    async def run_wait(
        self,
        op: OperateType,
        handle_fn: Callable[[OperateType, Container], Coroutine[Any, Any, None]],
        on_finish_cb: Callable[[], None],
    ):
        await handle_fn(op, self.container)
        on_finish_cb()
