import time
from time import time_ns
from typing import Callable, Dict, List, Optional, Tuple

import extend_asyncio as asyncio
from orbit.defines import Packet

from .. import flags
from ..dealer_node.protocols import (API_GroupedInferTask, GroupedInferTask,
                                     MultiPathInferTask)

_Pack = Tuple[MultiPathInferTask, Optional[Callable[[], None]]]
_CancelTrap = Callable[[], None]


class InferBatchSender:
    def __init__(self, name: str, send: Callable[[Packet, bool], None]) -> None:
        self.name = name
        self.base_send = send
        self.pendings: Dict[str, List[_Pack]] = {}
        self.trappings: List[Tuple[str, int, _Pack, _CancelTrap]] = []
        self._ev_loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._loop())
        self._sleep_fut: Optional[asyncio.Future[None]] = asyncio.Future()

    def send_no_group(
        self,
        target: str,
        task: MultiPathInferTask,
        start_trap_cb: Optional[Callable[[], None]] = None,
        end_trap_cb: Optional[Callable[[], None]] = None,
        send_cb: Optional[Callable[[], None]] = None,
    ):
        if send_cb is not None:
            send_cb()
        self.base_send((target, API_GroupedInferTask, GroupedInferTask(tasks=[task])), True)

    def send(
        self,
        target: str,
        task: MultiPathInferTask,
        start_trap_cb: Optional[Callable[[], None]] = None,
        end_trap_cb: Optional[Callable[[], None]] = None,
        send_cb: Optional[Callable[[], None]] = None,
    ):
        if not flags.GROUP_INFERENCE:
            if send_cb is not None:
                send_cb()
            self.base_send((target, API_GroupedInferTask, GroupedInferTask(tasks=[task])), True)
            return
        if flags.SHOULD_INTERCEPT(task.task.step):
            if start_trap_cb is not None:
                start_trap_cb()
            self._send_with_trap(target, task, end_trap_cb, send_cb)
        else:
            if self._sleep_fut is not None:
                fut = self._sleep_fut
                self._sleep_fut = None
                fut.set_result(None)
            self.pendings.setdefault(target, []).append((task, send_cb))

    def _send_with_trap(
        self,
        target: str,
        task: MultiPathInferTask,
        end_trap_cb: Optional[Callable[[], None]] = None,
        send_cb: Optional[Callable[[], None]] = None,
    ):
        canceled = False
        end_time = time_ns()+int(flags.INTERCEPT_WINDOW_TIME*1e9)

        def _cancel_trap():
            nonlocal canceled
            if canceled:
                return
            canceled = True
            if end_trap_cb:
                end_trap_cb()

        def _trap_timeup():
            nonlocal canceled
            if canceled:
                return
            canceled = True
            if end_trap_cb:
                end_trap_cb()
            if self._sleep_fut is not None:
                fut = self._sleep_fut
                self._sleep_fut = None
                fut.set_result(None)
        self.trappings.append((target, end_time, (task, send_cb), _cancel_trap))
        self._ev_loop.call_later(flags.INTERCEPT_WINDOW_TIME, _trap_timeup)

    def _step(self):
        pendings = self.pendings
        trappings = self.trappings
        self.pendings = {}
        self.trappings = []
        current_time = time.time_ns()
        for tgt, _tl, _p, _c in trappings:
            _lst = self.pendings.get(tgt)
            if current_time < _tl and ((_lst is None) or (len(_lst) < flags.INTERCEPT_MAX_SIZE)):
                self.trappings.append((tgt, _tl, _p, _c))
            else:
                _c()
                pendings.setdefault(tgt, []).append(_p)
        for target, sub_pendings in pendings.items():
            grouped_tasks = GroupedInferTask(
                tasks=[p[0] for p in sub_pendings]
            )
            for _, cb in sub_pendings:
                if cb is None:
                    continue
                cb()
            self.base_send((target, API_GroupedInferTask, grouped_tasks), True)
        self._sleep()

    async def _postpone(self):
        if self._sleep_fut is not None:
            fut = self._sleep_fut
            await fut
            await asyncio.postpone()

    def _sleep(self):
        assert self._sleep_fut is None
        self._sleep_fut = asyncio.Future()

    async def _loop(self):
        while True:
            await self._postpone()
            self._step()
