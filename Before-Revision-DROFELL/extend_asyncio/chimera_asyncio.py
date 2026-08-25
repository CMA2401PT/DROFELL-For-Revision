import asyncio
import collections
import heapq
import os
import time
import traceback
from typing import Dict, List, Optional, cast
from .common import *
from .profiler import MixtureProfiler
from . import golib
Handle = asyncio.events.Handle


class ChimeraEventLoop(asyncio.base_events.BaseEventLoop, Loop):  # type:ignore
    def __init__(self, busy_wait: bool) -> None:
        super().__init__()
        self.scheduled: List[Handle] = self._scheduled  # type:ignore
        self.ready: collections.deque[Handle] = self._ready  # type:ignore
        self._clock_resolution = time.get_clock_info('monotonic').resolution
        self.blocking_profiler = MixtureProfiler()
        self.blocking_profiler.enable(enable=False)
        self.nonblocking_profiler = MixtureProfiler()
        self.nonblocking_profiler.enable(enable=False)
        self.go_selector = golib.Selector(busy_wait)
        self._blocking_scheduler: Optional[BlockingScheduler] = None

        def exit_on_except(loop: Loop, e: Any):
            if hasattr(e, "__traceback__"):
                print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            print(e)
            os._exit(-1)

        self.on_except(exit_on_except)

    def set_blocking_scheduler(self, scheduler: BlockingScheduler) -> None:
        assert self._blocking_scheduler is None
        self._blocking_scheduler = scheduler

    @property
    def blocking_scheduler(self) -> Optional[BlockingScheduler]:
        return self._blocking_scheduler

    # async def run_blocking_soon(self, call: Callable[..., RetType], *call_args: Any, **schedule_kwargs: Any) -> RetType:
    #     fut: asyncio.Future[RetType] = asyncio.Future()

    #     def enclosure():
    #         ret = call(*call_args)
    #         fut.set_result(ret)
    #     if self.blocking_scheduler is None:
    #         self.call_soon(enclosure)
    #     else:
    #         self.blocking_scheduler.add_blocking_call(enclosure, **schedule_kwargs)
    #     return await fut

    # def call_blocking_soon(self, call: Callable[..., RetType], *call_args: Any, **schedule_kwargs: Any) -> None:
    #     if self.blocking_scheduler is None:
    #         self.call_soon(call, call_args)
    #         return

    #     def enclosure():
    #         call(*call_args)
    #     self.blocking_scheduler.add_blocking_call(enclosure, **schedule_kwargs)

    def on_except(self, callback: Callable[[Loop, Any], None]) -> None:
        def exception_handler(loop: asyncio.AbstractEventLoop, context: Dict[str, Any]):
            callback(self, context['exception'])
        self.set_exception_handler(exception_handler)

    def enable_profile(self, non_blocking: bool, blocking: bool) -> None:
        self.blocking_profiler.enable(blocking)
        self.nonblocking_profiler.enable(non_blocking)

    @property
    def blocking_calltime_ms(self) -> float:
        return self.blocking_profiler.blocking_time_ms

    @property
    def non_blocking_calltime_ms(self) -> float:
        return self.nonblocking_profiler.blocking_time_ms

    def clear_profile(self) -> None:
        self.blocking_profiler.clear_profile()
        self.nonblocking_profiler.clear_profile()

    def dump_profile(self, path: Optional[str] = None):
        if path is None:
            path = "dump_profile"
        os.makedirs(path, exist_ok=True)
        print(f"avg blocking call time: {self.blocking_profiler.blocking_time_ms:.4f}ms")
        self.blocking_profiler.dump_profile(os.path.join(path, "blocking"))
        print(f"avg non blocking call time: {self.nonblocking_profiler.blocking_time_ms:.4f}ms")
        self.nonblocking_profiler.dump_profile(os.path.join(path, "nonblocking"))

    def _run_once(self):
        # Remove delayed calls that were cancelled from head of queue.
        while self.scheduled and self.scheduled[0]._cancelled:
            self._timer_cancelled_count -= 1
            handle = heapq.heappop(self.scheduled)
            handle._scheduled = False  # type:ignore

        # get wait timeout for io
        wait_until = 0
        if self.ready or self._blocking_scheduler or cast(bool, self._stopping):  # type:ignore
            wait_until = 0
        elif self.scheduled:
            # Compute the desired timeout.
            wait_until = self.scheduled[0]._when  # type:ignore
        else:
            wait_until = -1
        self.go_selector.select_by_until(wait_until)

        end_time = self.time() + self._clock_resolution
        while self.scheduled:
            handle = self.scheduled[0]
            if handle._when >= end_time:  # type:ignore
                break
            handle = heapq.heappop(self.scheduled)
            handle._scheduled = False  # type:ignore
            self.ready.append(handle)

        ntodo = len(self.ready)
        if ntodo != 0:
            for _ in range(ntodo):
                handle = self.ready.popleft()
                if handle._cancelled:
                    continue
                else:
                    with self.nonblocking_profiler:
                        handle._run()
        elif self._blocking_scheduler:
            with self.blocking_profiler:
                self._blocking_scheduler.run_once()
        handle = None

    async def tcp_connect(self, addr: str) -> FrameConn:
        return cast(FrameConn, await self.go_selector.tcp_connect(addr))

    async def tcp_server(self, addr: str, on_conn: Callable[[FrameConn], Coroutine[Any, Any, None]]):
        return await self.go_selector.tcp_server(self, addr, on_conn)

    def pipe_reader(self, read_conn: MultiProcessConn) -> FrameConnReader:
        return cast(FrameConn, self.go_selector.pipe_reader(read_conn))

    def pipe_writer(self, write_conn: MultiProcessConn) -> FrameConnWriter:
        return cast(FrameConn, self.go_selector.pipe_writer(write_conn))

    def delegate_c_blocking_closure(self, load: BlockingLoad) -> Awaitable[Any]:
        return self.selector.delegate_c_blocking_closure(load)  # type:ignore

    def delegate_c_poll_closure(self, load: PollBlockingLoad) -> Awaitable[Any]:
        return self.selector.delegate_c_poll_closure(load)  # type:ignore

    # async def delegate_py_blocking_closure(self, load: BlockingLoad, with_gil: bool) -> Any:
    #     return await self.go_selector.delegate_py_blocking_closure(load, with_gil)

# def new_event_loop() -> GoSelectorEventLoop:
#     return GoSelectorEventLoop()


# _thread_local = threading.local()


# def set_event_loop(loop: GoSelectorEventLoop):
#     eloop = getattr(_thread_local, 'loop', None)
#     if eloop is not None:
#         raise ValueError("loop already set")
#     _thread_local.loop = loop


# def get_running_loop() -> GoSelectorEventLoop:
#     loop = getattr(_thread_local, 'loop', None)
#     if loop is None:
#         raise ValueError("no current loop")
#     return loop


# def get_event_loop() -> GoSelectorEventLoop:
#     loop = getattr(_thread_local, 'loop', None)
#     if loop is None:
#         loop = new_event_loop()
#         set_event_loop(loop)
#     return loop


# gather = asyncio.gather
# Queue = asyncio.Queue
# Future = asyncio.Future
# sleep = asyncio.sleep
golib.Future = asyncio.Future
golib.Queue = asyncio.Queue


# async def sleep_until(wake_time: float):
#     return await asyncio.sleep(wake_time-time.time())
