import time
from typing import Any, List
import extend_asyncio as asyncio


class WorkCtrl:
    def __init__(self, load: float, work_name: str, **extra_work_describe: Any) -> None:
        self.start_time: float = time.time()
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
                    print("timeout!")
                    finish_timeout = self._on_task_finish()

    async def submit(self, work_name: str, load: float, **extra_work_describe: Any):
        work_control = WorkCtrl(load, work_name, **extra_work_describe)
        self._new_work_queue.put_nowait(work_control)
        await work_control.wait_complete()


async def load():
    simulator = LoadSimulator()

    async def submit(work_name: str, load: float):
        st = time.time()
        await simulator.submit(work_name, load)
        print(f"work: {work_name} completed in {time.time()-st:.2f}, orig load:{load}")
    asyncio.create_task(submit("wait-2s", 2))
    asyncio.create_task(submit("wait-3s", 3))
    await asyncio.sleep(4.5)
    asyncio.create_task(submit("wait-1s", 1))
    await asyncio.Future()
asyncio.config_impl("golang", False)
asyncio.run(load())
