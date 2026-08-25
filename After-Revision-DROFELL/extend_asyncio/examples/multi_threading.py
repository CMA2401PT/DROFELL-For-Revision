from dataclasses import dataclass
import threading
import time
from typing import Any, Awaitable, Callable, Dict, Optional
import extend_asyncio as asyncio
from queue import Queue

asyncio.config_impl("golang", True)
FrameConn = asyncio.FrameConn
Frames = asyncio.Frames


def run_simple():
    async def simple_sleep_coro(t: int):
        print(f"Start: {t}")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        print(f"{t} Slept 0 sec")
        await asyncio.sleep(1)
        print(f"{t} Slept 1 sec")
        await asyncio.sleep(2)
        print(f"{t} Slept 2 more sec")

    def thread_task(t: int):
        asyncio.run(simple_sleep_coro(t))

    start_time = time.time()
    t1 = threading.Thread(target=thread_task, args=(1,))
    t2 = threading.Thread(target=thread_task, args=(2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(f"total: {time.time()-start_time}")

# run_simple()
# run_simple()


@dataclass
class Operate:
    time: int
    fut: asyncio.Future[Any]
    sync: Optional[Awaitable[None]]

    def set_result(self, ret: Any, sp: Awaitable[None]):
        self.ret = ret
        self.fut.set_result(ret)
        self.sync = sp

    async def get_result(self):
        ret = await self.fut
        if self.sync is not None:
            await self.sync
        return ret


class ThreadOperate:
    def __init__(self, op: Operate, loop: asyncio.Loop) -> None:
        self.base_op = op
        self.loop = loop
        self.sync: Optional[Awaitable[None]]

    def __getattr__(self, name: str) -> Any:
        return self.base_op.__getattribute__(name)

    def set_result(self, ret: Any, sp: Optional[Awaitable[None]] = None):
        if sp is None:
            self.loop.call_soon_threadsafe(self.base_op.set_result, ret, None)
            return
        else:
            async def sync_and_set():
                await sp
                self.loop.call_soon_threadsafe(self.base_op.set_result, ret, None)
            self.sync = sync_and_set()


async def run_op(op: Operate):
    await asyncio.sleep(op.time/2)
    ret: Dict[str, Optional[int]] = {"r": None}
    async_end = time.time()+op.time/2

    async def sync_point():
        await asyncio.sleep(async_end-time.time())
        ret["r"] = op.time
    op.set_result(ret, sync_point())


class AsyncTaskRunner:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[Operate] = asyncio.Queue()
        self.task = asyncio.create_task(self.loop())

    async def loop(self):
        while True:
            op = await self.queue.get()
            await run_op(op)

    def __call__(self, op: Operate) -> Any:
        self.queue.put_nowait(op)


class ThreadTaskRunner:
    def __init__(self, num_workers: int = 4) -> None:
        self._loop = asyncio.get_running_loop()
        self.queue: Queue[Operate] = Queue(-1)
        for _ in range(num_workers):
            threading.Thread(target=self._working_thread).start()

    async def _working_thread_coro(self):
        while True:
            op = self.queue.get()
            await run_op(op)
            if op.sync is not None:
                await op.sync

    def _working_thread(self):
        asyncio.run(self._working_thread_coro())

    def __call__(self, op: Operate) -> Any:
        top = ThreadOperate(op, self._loop)
        self.queue.put_nowait(top)


async def run():
    asyncio_runner = AsyncTaskRunner()
    thread_runner = ThreadTaskRunner()
    st = time.time()
    op1 = Operate(1, fut=asyncio.Future(), sync=None)
    asyncio_runner(op1)
    op2 = Operate(1, fut=asyncio.Future(), sync=None)
    thread_runner(op2)
    ret1 = await op1.get_result()
    ret2 = await op2.get_result()
    print(ret1, ret2, time.time()-st)
asyncio.run(run())

# class ThreadTaskFromAsyncio:
#     def __init__(self, task: Callable[[], Any]) -> None:
#         self._loop = asyncio.get_running_loop()
#         self._task = task
#         self._result: asyncio.Future[Any] = asyncio.Future()

#     def execute(self) -> None:
#         ret = self._task()
#         self._loop.call_soon_threadsafe(lambda: self._result.set_result(ret))

#     async def wait_result(self) -> Any:
#         return await self._result


# class QueueThreadWorker:
#     def __init__(self, num_workers: int = 1) -> None:
#         self.task_queue: Queue[ThreadTaskFromAsyncio] = Queue(-1)
#         for _ in range(num_workers):
#             threading.Thread(target=self._working_thread).start()

#     def submit(self, task: ThreadTaskFromAsyncio):
#         self.task_queue.put(task)

#     def _working_thread(self):
#         while True:
#             t = self.task_queue.get()
#             t.execute()

#     async def run(self, task: Callable[[], Any]) -> Any:
#         wrap = ThreadTaskFromAsyncio(task)
#         self.submit(wrap)
#         return await wrap.wait_result()
