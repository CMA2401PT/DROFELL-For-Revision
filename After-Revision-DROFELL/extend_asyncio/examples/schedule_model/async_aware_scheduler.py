import time
import extend_asyncio as asyncio
from normal_scheduler import BlockingSubmitter, Ret
from typing import Any, Callable, List


class WorkCtrl:
    def __init__(self, load: float, work_name: str, start_time: float, **extra_work_describe: Any) -> None:
        self.start_time: float = start_time
        self.load = load
        self.progress = 0
        self.work_name = work_name
        self.extra_work_describe = extra_work_describe
        self.fut = asyncio.Future()

    async def wait_complete(self):
        return await self.fut

    def notify_finish(self):
        self.fut.set_result(None)

    @property
    def residue(self) -> float:
        if self.progress >= self.load:
            return 0
        return self.load-self.progress

    def increase_progress(self, progress: float):
        self.progress += progress
        if self.progress >= self.load:
            self.progress = self.load


class LoadSimulator:
    def __init__(self) -> None:
        self._step_start_time = time.time()
        self._current_works: List[WorkCtrl] = []
        self._effiency_discount: List[float] = []
        self._new_work_queue = asyncio.Queue()

        self.work_task = asyncio.create_task(self._work_loop())

    def _update_work_progress(self):
        new_start_time = time.time()
        delta = new_start_time-self._step_start_time
        self._step_start_time = new_start_time
        if len(self._current_works) == 0:
            return
        for e, t in zip(self._effiency_discount, self._current_works):
            t.increase_progress(e*delta)

    def _finish_complete_work(self):
        _current_works: List[WorkCtrl] = []
        _effiency_discount: List[float] = []
        for e, t in zip(self._effiency_discount, self._current_works):
            if t.residue == 0:
                t.notify_finish()
                continue
            _current_works.append(t)
            _effiency_discount.append(e)
        self._current_works = _current_works
        self._effiency_discount = _effiency_discount

    def _get_most_recent_finish(self) -> float:
        self._update_discount()
        if len(self._current_works) == 0:
            return 0
        earliest = self._current_works[0].residue/self._effiency_discount[0]
        for e, t in zip(self._effiency_discount[1:], self._current_works[1:]):
            remain = t.residue/e
            if remain < earliest:
                earliest = remain
        return earliest

    def _on_new_work(self, work: WorkCtrl) -> float:
        self._update_work_progress()
        self._finish_complete_work()
        work.increase_progress(time.time()-work.start_time)
        self._current_works.append(work)
        return self._get_most_recent_finish()

    def _on_task_finish(self) -> float:
        self._update_work_progress()
        self._finish_complete_work()
        return self._get_most_recent_finish()

    def _update_discount(self) -> None:
        if len(self._current_works) == 0:
            self._effiency_discount = []
            return
        self._effiency_discount = [1/len(self._current_works)]*len(self._current_works)

    async def _work_loop(self):
        finish_timeout: float = 0
        while True:
            if len(self._current_works) == 0:
                new_work = await self._new_work_queue.get()
                finish_timeout = self._on_new_work(new_work)
            else:
                try:
                    new_work = await asyncio.wait_for(self._new_work_queue.get(), finish_timeout)
                    finish_timeout = self._on_new_work(new_work)
                except Exception as e:
                    finish_timeout = self._on_task_finish()

    async def submit(self, load: float, work_name: str,  start_time: float, **extra_work_describe: Any):
        work_control = WorkCtrl(load, work_name, start_time, **extra_work_describe)
        self._new_work_queue.put_nowait(work_control)
        await work_control.wait_complete()


class AsyncAwareSubmitter(BlockingSubmitter):
    def __init__(self, base: BlockingSubmitter, low_async_time: float):
        self.base = base
        self.async_load_simulator = LoadSimulator()
        self.low_async_time = low_async_time

    async def submit(self, call: Callable[..., Ret], *args: Any, task_name: str) -> Ret:
        complete_time = 0

        def enclosure():
            nonlocal complete_time
            ret = call(*args)
            complete_time = time.time()
            return ret, complete_time
        ret, complete_time = await self.base.submit(enclosure, task_name=task_name)
        if task_name == "low":
            await self.async_load_simulator.submit(self.low_async_time, task_name, complete_time)
        return ret
