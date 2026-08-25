import asyncio
from asyncio import events
import collections
import contextvars
import heapq
import os
import queue
import time
from multiprocessing.connection import Connection as MultiProcessConn
from typing import Any, Callable, List, Coroutine, Optional, Tuple, cast

from extend_asyncio import std_asyncio
from extend_asyncio.common import PollBlockingLoad
from .profiler import MixtureProfiler
from . import golib
from .common import *
import traceback


class Future:
    def __init__(self) -> None:
        self._done = False
        self._result = None
        self._callbacks: List[Callable[[Any], None]] = []
        self._cancelled = False
        self._exception = None

    def done(self):
        return self._done

    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self):
        self._cancelled = True

    def result(self):
        return self._result

    def add_done_callback(self, cb: Callable[[Any], None]) -> None:
        if self._done:
            cb(self._result)
        else:
            self._callbacks.append(cb)

    def remove_done_callback(self, fn: Callable[[Any], None]):
        self._callbacks.remove(fn)

    def _invoke_callbacks(self):
        for cb in self._callbacks:
            cb(self)

    def set_result(self, ret: Any):
        if self._cancelled or self._done:
            return
        self._result = ret
        self._done = True
        self._invoke_callbacks()

    def set_exception(self, exception: Any):
        if self._cancelled or self._done:
            return
        self._exception = exception
        self._done = True
        self._invoke_callbacks()

    def __await__(self):
        if not self._done:
            yield self
        if self._exception is not None:
            raise self._exception
        assert self._done
        return self._result


class Lock:
    def __init__(self) -> None:
        self.queue = Queue()
        self.queue.put_nowait(None)

    async def acquire(self) -> None:
        await self.queue.get()

    def release(self) -> None:
        self.queue.put_nowait(None)


class Task(Future):
    def __init__(self, coro: Coroutine[Any, Any, None], *, name: Optional[str] = None,
                 context: Optional[contextvars.Context] = None):
        super().__init__()
        self.coro = coro
        self.send_val = None
        self._source_traceback = None
        self._name = name
        self._context = context if context is not None else contextvars.copy_context()

    def __str__(self):
        return str(self.coro.cr_frame)  # type: ignore

    def get_name(self) -> str:
        return self._name or f"Task-{id(self)}"

    def set_name(self, value: Any) -> None:
        self._name = str(value)

    def step(self):
        assert not self.cancelled()
        try:
            fut = self._context.run(self.coro.send, self.send_val)
            if fut is None:
                return None
            return fut
        except StopIteration as e:
            self.set_result(e.value)
            return None
        except Exception as e:
            self.set_exception(e)
            # print(f"exception: {e}")
            raise e


class Call(Future):
    def __init__(self, cb: Any, args: Tuple[Any], *,
                 context: Optional[contextvars.Context] = None) -> None:
        super().__init__()
        self.cb = cb
        self.args = args
        self._context = context if context is not None else contextvars.copy_context()

    def step(self):
        # print("call")
        assert not self.cancelled()
        self.set_result(self._context.run(self.cb, *self.args))


class Sleep:
    def __init__(self, delay: float, wake_time: Optional[float]):
        self.wake_time: float = \
            wake_time if (wake_time is not None) \
            else (time.time() + delay)

    def __await__(self):
        yield self
        # print("continued")


def sleep(delay: float):
    return Sleep(delay, wake_time=None)


def sleep_until(wake_time: float):
    return Sleep(0, wake_time=wake_time)


class Queue:
    def __init__(self):
        self._items: List[Any] = []
        self._get_waiters: List[Future] = []

    def empty(self) -> bool:
        return len(self._items) == 0

    def get_nowait(self) -> Any:
        assert self._items
        return self._items.pop(0)

    def get(self) -> Any:
        fut = Future()
        if self._items:
            fut.set_result(self._items.pop(0))
        else:
            self._get_waiters.append(fut)
        return fut

    def put_nowait(self, item: Any):
        while True:
            if not self._get_waiters:
                break
            fut = self._get_waiters.pop(0)
            if not fut.done():
                fut.set_result(item)
                return
        self._items.append(item)


golib.Future = Future
golib.Queue = Queue


class EventLoopBase:
    def __init__(self):
        self.tasks: collections.deque[Union[Task, Call]] = collections.deque()
        # self.tasks: List[Task] = []
        # The monotonically increasing sequence is a deterministic heap
        # tie-breaker.  Two network-delay callbacks can have the same
        # wake_time; Call and Task objects are intentionally not orderable.
        self.sleeping: List[Tuple[float, int, Union[Task, Call]]] = []
        self._timer_sequence = 0

    def schedule_sleeping(
        self, wake_time: float, task: Union[Task, Call]
    ) -> None:
        self._timer_sequence += 1
        heapq.heappush(
            self.sleeping, (wake_time, self._timer_sequence, task)
        )


class WaitFirst(Future):
    def __init__(self, tasks: Tuple[Task, ...]):
        super().__init__()
        for t in tasks:
            assert isinstance(t, Task)
        if len(tasks) == 0:
            self.set_result((set(), set()))
        self.complete = False
        self.tasks = tasks

    def start(self, loop: EventLoopBase):
        runnings: set[Task] = set()
        dones: set[Task] = set()

        def enclosure(t: Task):
            def on_complete(ret: Any):
                if self.complete:
                    return
                self.complete = True
                runnings.remove(t)
                dones.add(t)
                self.set_result((dones, runnings))
            return on_complete
        for task in self.tasks:
            if task.done():
                dones.add(task)
            else:
                task.add_done_callback(enclosure(task))
                runnings.add(task)


async def wait_first(tasks: Tuple[Task]):
    if len(tasks) == 0:
        return set(), set()
    g = WaitFirst(tasks)
    g.start(cast(EventLoopBase, asyncio.get_event_loop()))
    return await g


class Gather(Future):
    def __init__(self, coros: Tuple[Union[Coroutine[Any, Any, None], Task], ...]):
        super().__init__()
        self.coros = coros
        self.results: List[Any] = [None] * len(coros)
        self.done_count = 0

        if not coros:
            self.set_result([])

    def start(self, loop: EventLoopBase):
        for i, coro in enumerate(self.coros):
            if not isinstance(coro, Task):
                task = Task(self._wrap(coro, i))
                loop.tasks.append(task)
            else:
                self._wrap_task(coro, i)

    async def _wrap(self, coro: Coroutine[Any, Any, Any], index: int):
        result = await coro
        self.results[index] = result
        self.done_count += 1
        # print(f"done: {self.done} of {len(self.coros)}")
        if self.done_count == len(self.coros):
            self.set_result(self.results)

    def _wrap_task(self, task: Task, index: int):
        def cb(result: Any):
            self.results[index] = result
            self.done_count += 1
            if self.done_count == len(self.coros):
                self.set_result(self.results)
        task.add_done_callback(cb)


def gather(*coros: Coroutine[Any, Any, Any]):
    if len(coros) == 0:
        fut = Future()
        fut.set_result(())
        return fut
    g = Gather(coros)
    g.start(cast(EventLoopBase, asyncio.get_event_loop()))
    return g


def wait_for(fut: Union[Future, Coroutine[Any, Any, Any]], timeout: Optional[float]):
    if timeout is None:
        return fut
    loop = cast(GolangEventLoop, asyncio.get_event_loop())
    if not isinstance(fut, Future):
        new_fut = Task(fut)
        loop.tasks.append(new_fut)
    else:
        new_fut = fut
        # Future()
        # old_fut = fut

        # def set_result_if_not_timeout(f: Future):
        #     if not new_fut.done() or new_fut.cancelled():
        #         new_fut.set_result(f.result())
        # old_fut.add_done_callback(set_result_if_not_timeout)

    def set_timeout():
        if not new_fut.done():
            new_fut.set_exception(TimeoutError())
    timeout_setter = loop.call_later(timeout, set_timeout)
    # new_fut.add_done_callback(lambda _: timeout_setter.cancel())
    return new_fut


class GolangEventLoop(EventLoopBase, Loop, asyncio.AbstractEventLoop):
    def __init__(self, busy_wait: bool):
        super().__init__()
        self.selector = golib.Selector(busy_wait=busy_wait)
        self.blocking_profiler = MixtureProfiler()
        self.blocking_profiler.enable(enable=False)
        self.nonblocking_profiler = MixtureProfiler()
        self.nonblocking_profiler.enable(enable=False)
        self._blocking_scheduler: Optional[BlockingScheduler] = None
        self.done: bool = False
        self._is_running = False
        self._tasks_from_other_threads: queue.Queue[Union[Task, Call]] = queue.Queue()
        self._clock_resolution = time.get_clock_info('monotonic').resolution
        self.postpone_flag = False
        self.postpone_tasks: collections.deque[Union[Task, Call]] = collections.deque()

        def exit_on_except(loop: Loop, e: Any):
            if hasattr(e, "__traceback__"):
                print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            print(e)
            os._exit(-1)

        self._except_callback = exit_on_except

    def is_running(self) -> bool:
        return self._is_running

    def set_blocking_scheduler(self, scheduler: BlockingScheduler) -> None:
        assert self._blocking_scheduler is None
        self._blocking_scheduler = scheduler

    @property
    def blocking_scheduler(self) -> Optional[BlockingScheduler]:
        return self._blocking_scheduler

    # async def run_blocking_soon(self, call: Callable[..., RetType], *call_args: Any, **schedule_kwargs: Any) -> RetType:
    #     fut = Future()

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
        self._except_callback = callback

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

    def create_task(self, coro: Coroutine[Any, Any, Any], *, name: Optional[str] = None,
                    context: Optional[contextvars.Context] = None) -> Any:  # type:ignore
        # fut = Future()
        task = Task(coro, name=name, context=context)
        # task.add_done_callback(lambda t: fut.set_result(t.result()))
        self.tasks.append(task)
        return task

    def call_later(self, delay: float, cb: Any, *args: Any,
                   context: Optional[contextvars.Context] = None):  # type:ignore
        wake_time = time.time() + delay
        call = Call(cb, args, context=context)
        self.schedule_sleeping(wake_time, call)
        return call

    def call_soon(self, callback: Callable[..., Any], *args: Any,
                  context: Optional[contextvars.Context] = None):  # type:ignore
        task = Call(cb=callback, args=args, context=context)
        self.tasks.append(task)
        return task

    def call_soon_threadsafe(self, callback: Callable[..., Any], *args: Any,
                             context: Optional[contextvars.Context] = None):  # type:ignore
        self._tasks_from_other_threads.put_nowait(Call(cb=callback, args=args, context=context))
        self.selector.interrupt()

    async def run_in_executor(self, executor: Executor, task: Callable[..., Any], *args: Any) -> Any:  # type:ignore
        fut = Future()

        def enclosure():
            ret = task(*args)
            self.call_soon_threadsafe(fut.set_result, ret)
        executor.submit(enclosure)
        return await fut

    async def gather(self, *coros: Coroutine[Any, Any, Any]):
        g = Gather(coros)
        g.start(self)
        return await g

    def _run_one(self, task: Union[Task, Call]):
        if task.cancelled() or task.done():
            return
        with self.nonblocking_profiler:
            fut = task.step()
        if not task.done():
            if isinstance(fut, Sleep):
                assert isinstance(task, Task)
                self.schedule_sleeping(fut.wake_time, task)
            elif fut is None:
                # asyncio.sleep(0)
                # call task later
                if self.postpone_flag:
                    self.postpone_tasks.append(task)
                    self.postpone_flag = False
                else:
                    self.tasks.append(task)
            else:
                assert isinstance(fut, Future), fut

                def cb(fut: Future):
                    self.tasks.append(task)
                fut.add_done_callback(cb)
        else:
            assert fut is None

    def run(self, forever: bool = False):
        self._is_running = True
        events._set_running_loop(self)
        try:
            while True:
                while not self._tasks_from_other_threads.empty():
                    task = self._tasks_from_other_threads.get()
                    self.tasks.append(task)

                wait_until = - 1  # infinity wait
                if self.tasks or self.postpone_tasks or self._blocking_scheduler:
                    wait_until = 0  # wait 0
                elif not forever:
                    if self.done:
                        events._set_running_loop(None)
                        self._is_running = False
                        print("event loop quit")
                        return
                if wait_until == - 1 and self.sleeping:
                    wait_until = self.sleeping[0][0]  # wait specific
                # print(f"select: {(wait_until-time.time_ns())/1e6:.4f}")
                # if wait_until == -1:
                #     print("wait infinity")
                self.selector.select_by_until(wait_until)

                now = time.time()+self._clock_resolution
                # 把到时间的 sleep 任务加入运行队列
                while self.sleeping and self.sleeping[0][0] <= now:
                    _, _, task = heapq.heappop(self.sleeping)
                    self.tasks.append(task)

                current_count = len(self.tasks)
                for _ in range(current_count):
                    task = self.tasks.popleft()
                    self._run_one(task)
                if current_count == 0:
                    if len(self.postpone_tasks):
                        task = self.postpone_tasks.popleft()
                        self._run_one(task)
                        continue
                    elif self._blocking_scheduler:
                        with self.blocking_profiler:
                            self._blocking_scheduler.run_once()

        except Exception as e:
            self._except_callback(self, e)

    def run_forever(self):
        self.run(forever=True)

    def run_until_complete(self, future: Coroutine[Any, Any, Any]) -> Any:  # type:ignore
        if future is None:
            return
        ret = None

        async def wrap():
            nonlocal ret
            ret = await future
            self.done = True
        self.tasks.append(Task(wrap()))
        self.run(False)
        return ret

    def debug_callback_after(self, seconds: float, ev_type: str, on_result_cb: Callable[[Any], bool]) -> None:
        return self.selector.debug_callback_after(seconds, ev_type, on_result_cb)

    def debug_event_after(self, seconds: float, ev_type: str) -> Any:
        return self.selector.debug_event_after(seconds, ev_type)

    async def tcp_connect(self, addr: str) -> FrameConn:
        return cast(FrameConn, await self.selector.tcp_connect(addr))

    async def tcp_server(self, addr: str, on_conn: Callable[[FrameConn], Coroutine[Any, Any, None]]):
        return await self.selector.tcp_server(self, addr, on_conn)

    def pipe_reader(self, read_conn: MultiProcessConn) -> FrameConnReader:
        return cast(FrameConn, self.selector.pipe_reader(read_conn))

    def pipe_writer(self, write_conn: MultiProcessConn) -> FrameConnWriter:
        return cast(FrameConn, self.selector.pipe_writer(write_conn))

    def delegate_c_blocking_closure(self, load: BlockingLoad) -> Awaitable[Any]:
        return self.selector.delegate_c_blocking_closure(load)  # type:ignore

    def delegate_c_poll_closure(self, load: PollBlockingLoad) -> Awaitable[Any]:
        return self.selector.delegate_c_poll_closure(load)

    # async def delegate_py_blocking_closure(self, load: BlockingLoad, with_gil: bool) -> Any:
    #     return await self.selector.delegate_py_blocking_closure(load, with_gil)

    # def torch_event_sync(self) -> Awaitable[int]:
    #     return self.selector.torch_event_sync()

    def create_future(self) -> Future:  # type:ignore
        return Future()

    def close(self) -> None:
        return None

    def shutdown_asyncgens(self) -> None:  # type:ignore
        return None

    def shutdown_default_executor(self, timeout: Optional[float] = None) -> None:  # type:ignore
        return None
# def new_event_loop() -> EventLoop:
#     return EventLoop()


# _thread_local = threading.local()


# def set_event_loop(loop: EventLoop):
#     eloop = getattr(_thread_local, 'loop', None)
#     if eloop is not None:
#         raise ValueError("loop already set")
#     _thread_local.loop = loop


# def get_running_loop() -> EventLoop:
#     loop = getattr(_thread_local, 'loop', None)
#     if loop is None:
#         raise ValueError("no current loop")
#     return loop


# def get_event_loop() -> EventLoop:
#     loop = getattr(_thread_local, 'loop', None)
#     if loop is None:
#         loop = new_event_loop()
#         set_event_loop(loop)
#     return loop
