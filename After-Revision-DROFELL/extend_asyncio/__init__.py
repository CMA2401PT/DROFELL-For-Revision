import asyncio
import threading
import time
from typing import Any, Awaitable, Coroutine, Generic, List, Literal, Optional, Union, cast, Tuple
import typing

from .lite_asyncio import GolangEventLoop
from . import lite_asyncio
from .common import Loop, BlockingLoad, FrameConn, FrameConnReader, FrameConnWriter, Frames, BasicBlockingScheduler, BlockingScheduler, PollBlockingLoad
from .chimera_asyncio import ChimeraEventLoop
from .std_asyncio import StdEventLoop
from . import golib
_make_lint_happy = (BlockingLoad, FrameConn, FrameConnReader, FrameConnWriter,
                    Frames, BasicBlockingScheduler, BlockingScheduler, PollBlockingLoad)

Task = asyncio.Task


def work_load_simulate(seconds: float, work_name: str):
    golib.goselector.work_load_simulate(seconds, work_name)


T = typing.TypeVar('T')


class Queue(asyncio.Queue[T]):
    pass


class Future(asyncio.Future[T]):
    pass


class Event(asyncio.Event):
    pass


class Lock(asyncio.Lock):
    pass


def _ensure_loop_notset():
    try:
        asyncio.get_running_loop()
        raise ValueError("loop already set!")
    except:
        pass


def impl_not_implement():
    raise Exception("asyncio implement not config")


def config_impl(impl: Literal["py", "chimera", "golang"], busy_wait: bool):
    policy = asyncio.get_event_loop_policy()
    assert isinstance(policy, EventLoopPolicy)
    assert policy.impl == impl_not_implement
    _ensure_loop_notset()

    def make_loop() -> Loop:
        global Queue, Future, Lock, gather, wait_first, sleep_until, wait_for, sleep
        if impl == "py":
            golib.Future = asyncio.Future
            golib.Queue = asyncio.Queue
            return StdEventLoop(busy_wait=busy_wait)
        elif impl == "chimera":
            golib.Future = asyncio.Future
            golib.Queue = asyncio.Queue
            return ChimeraEventLoop(busy_wait=busy_wait)
        elif impl == "golang":
            golib.Future = lite_asyncio.Future
            golib.Queue = lite_asyncio.Queue
            Queue = lite_asyncio.Queue
            Future = lite_asyncio.Future
            Lock = lite_asyncio.Lock
            gather = lite_asyncio.gather  # type:ignore
            wait_first = lite_asyncio.wait_first  # type:ignore
            sleep = lite_asyncio.sleep  # type:ignore
            sleep_until = lite_asyncio.sleep_until  # type:ignore
            wait_for = lite_asyncio.wait_for  # type:ignore
            return GolangEventLoop(busy_wait=busy_wait)  # type:ignore
        raise ValueError(impl)
    asyncio.set_event_loop_policy(EventLoopPolicy(make_loop))


class EventLoopPolicy(asyncio.AbstractEventLoopPolicy):
    class _Local(threading.local):
        loop = None
        set_called = False

    def __init__(self, impl: Any):
        self._local = self._Local()
        self._watcher: Any = None
        self.impl = impl

    def get_event_loop(self):
        if (self._local.loop is None and not self._local.set_called and threading.current_thread() is threading.main_thread()):
            self.set_event_loop(self.new_event_loop())

        if self._local.loop is None:
            raise RuntimeError(
                'There is no current event loop in thread %r.' % threading.current_thread().name)

        return self._local.loop

    def set_event_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        self._local.set_called = True
        if loop is not None:
            assert self._local.loop is None
            assert loop is None or isinstance(loop, asyncio.AbstractEventLoop)
        self._local.loop = loop  # type: ignore
        if (self._watcher is not None and
                threading.current_thread() is threading.main_thread()):
            self._watcher.attach_loop(loop)

    def new_event_loop(self):
        loop = self.impl()
        return loop

    def _init_watcher(self):
        with asyncio.events._lock:  # type: ignore
            if self._watcher is None:  # pragma: no branch
                self._watcher = asyncio.ThreadedChildWatcher()
                if threading.current_thread() is threading.main_thread():
                    self._watcher.attach_loop(self._local.loop)

    def get_child_watcher(self):
        """Get the watcher for child processes.

        If not yet set, a ThreadedChildWatcher object is automatically created.
        """
        if self._watcher is None:
            self._init_watcher()

        return self._watcher

    def set_child_watcher(self, watcher: Any):
        """Set the watcher for child processes."""

        assert watcher is None or isinstance(
            watcher, asyncio.AbstractChildWatcher)

        if self._watcher is not None:
            self._watcher.close()

        self._watcher = watcher


asyncio.set_event_loop_policy(EventLoopPolicy(impl_not_implement))


def run(coro: Coroutine[Any, Any, T]):
    return asyncio.run(coro)


def sleep(timeout: float):
    return asyncio.sleep(timeout)


def sleep_until(wake_time_s: float):
    return asyncio.sleep(wake_time_s-time.time())


def gather(*coros: Union[Coroutine[Any, Any, T], asyncio.Task[T]]) -> Future[List[T]]:
    return asyncio.gather(*coros)


def wait_first(tasks: Tuple[asyncio.Task[T], ...]):
    return asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)


async def wait_for(fut: Awaitable[T], timeout: Optional[float]) -> T:
    return await asyncio.wait_for(fut, timeout=timeout)


wait_for = asyncio.wait_for


def get_event_loop() -> Loop:
    loop = cast(Loop, asyncio.get_event_loop())
    assert isinstance(loop, Loop)
    # asyncio._set_running_loop(loop)
    return loop


def get_running_loop() -> Loop:
    loop = cast(Loop, asyncio.get_running_loop())
    assert isinstance(loop, Loop)
    # asyncio._set_running_loop(loop)
    return loop


def new_event_loop() -> Loop:
    return cast(Loop, asyncio.new_event_loop())


def create_task(coro: Coroutine[Any, Any, T]):
    return asyncio.create_task(coro)


def run_coroutine_threadsafe(coro: Coroutine[Any, Any, Any], loop: Loop):
    # type:ignore
    return asyncio.run_coroutine_threadsafe(coro=coro, loop=loop)


async def postpone():
    loop = asyncio.get_running_loop()
    if isinstance(loop, GolangEventLoop):
        loop.postpone_flag = True
        await asyncio.sleep(0)
    else:
        for _ in range(3):
            await asyncio.sleep(0)
