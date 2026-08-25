from typing import List, Optional, Dict, Set, Tuple

from .brain import *
from .base_types import *
import extend_asyncio as asyncio


class TrainTaskReorderAndDropCtrl:
    def __init__(self, version: int, with_fwd_and_bwd: bool) -> None:
        self.version: int = version
        self.full_batch: Optional[Set[str]] = None
        self.forwarded: Set[str] = set()
        self.completed: Set[str] = set()
        self.with_fwd_and_bwd = with_fwd_and_bwd
        self.version_waitor: Dict[int, Set[asyncio.Future[None]]] = {}
        self.forward_waitor: Dict[str, asyncio.Future[None]] = {}
        self.reorders: Set[Tuple[int, str, bool]] = set()
        self.running: Set[Tuple[int, str, bool]] = set()

        # should_drop -> (should_reorder -> wait_reorder) -> set_running -> operate -> add_complete_task -> should_update -> version_update

    @property
    def should_update(self):
        return (self.full_batch is not None) and (len(self.forwarded) == 0) and (len(self.completed) == len(self.full_batch))

    def version_update(self):
        self.version += 1
        self.full_batch = None
        self.forwarded = set()
        self.completed = set()
        waitings = self.version_waitor.get(self.version)
        if waitings is not None:
            del self.version_waitor[self.version]
            for w in waitings:
                w.set_result(None)

    def can_operate(self, req_version: int, minibatch_id: str, is_forward: bool = False):
        return (req_version == self.version) and \
            (self.full_batch is not None) and \
            (minibatch_id in self.full_batch) and \
            (minibatch_id not in self.completed) and \
            ((not self.with_fwd_and_bwd) or (is_forward != (minibatch_id in self.forwarded)))

    def _should_drop(self, req_version: int, minibatch_id: str, full_batch: Union[Set[str], List[str]], is_forward: bool = False):
        full_batch = set(full_batch)
        if req_version < self.version:
            return True
        if (req_version, minibatch_id, is_forward) in self.running:
            return True
        if self._should_reorder(req_version, minibatch_id, is_forward):
            reorder_id: Tuple[int, str, bool] = (req_version, minibatch_id, is_forward)
            return reorder_id in self.reorders
        if self.full_batch is None:
            self.full_batch = full_batch
            return False
        if minibatch_id not in self.full_batch:
            return True
        if self.with_fwd_and_bwd:
            if is_forward and minibatch_id in self.forwarded:
                return True
        return minibatch_id in self.completed

    def _should_reorder(self, req_version: int, minibatch_id: str, is_forward: bool = False):
        if req_version > self.version:
            return True
        return self.with_fwd_and_bwd and (not is_forward) and (minibatch_id not in self.forwarded)

    async def _wait_reorder(self, req_version: int, minibatch_id: str, is_forward: bool = False) -> None:
        reorder_id: Tuple[int, str, bool] = (req_version, minibatch_id, is_forward)
        assert not reorder_id in self.reorders
        self.reorders.add(reorder_id)
        if req_version == self.version:
            assert not is_forward
            fut: asyncio.Future[None] = asyncio.Future()
            self.forward_waitor[minibatch_id] = fut
            # print(f"{req_version} {minibatch_id} forward={is_forward} wait for forward")
            await fut
            # print(f"{req_version} {minibatch_id} forward={is_forward} free by forward")
            self.reorders.remove(reorder_id)
            return
        assert req_version > self.version
        fut: asyncio.Future[None] = asyncio.Future()
        if req_version not in self.version_waitor:
            self.version_waitor[req_version] = set()
        self.version_waitor[req_version].add(fut)
        # print(f"{req_version} {minibatch_id} forward={is_forward} wait for version")
        await fut
        self.reorders.remove(reorder_id)
        # print(f"{req_version} {minibatch_id} forward={is_forward} free by version")
        if is_forward or (not self.with_fwd_and_bwd):
            return
        await self._wait_reorder(req_version, minibatch_id, False)

    async def can_continue_or_reoder(self,
                                     req_version: int, minibatch_id: str, full_batch: Union[Set[str], List[str]],
                                     is_forward: bool = False,
                                     on_reoders_start: Callable[[], None] = lambda: None, on_reoders_done: Callable[[], None] = lambda: None,
                                     ):
        if self._should_drop(req_version, minibatch_id, full_batch, is_forward):
            return False
        if self._should_reorder(req_version, minibatch_id, is_forward):
            on_reoders_start()
            await self._wait_reorder(req_version, minibatch_id, is_forward)
            on_reoders_done()
            if self._should_drop(req_version, minibatch_id, full_batch, is_forward):
                return False
        return True

    def set_running(self, req_version: int, minibatch_id: str, is_forward: bool = False):
        self.running.add((req_version, minibatch_id, is_forward))

    def add_complete_task(self, req_version: int, minibatch_id: str, is_forward: bool = False):
        self.running.discard((req_version, minibatch_id, is_forward))
        assert self.can_operate(req_version, minibatch_id, is_forward)
        if self.with_fwd_and_bwd:
            if is_forward:
                self.forwarded.add(minibatch_id)
                w = self.forward_waitor.get(minibatch_id)
                if w is not None:
                    del self.forward_waitor[minibatch_id]
                    w.set_result(None)
                return
            else:
                self.forwarded.remove(minibatch_id)
        self.completed.add(minibatch_id)


class InferTaskReorderAndDropCtrl:
    def __init__(self, init_step: int) -> None:
        self.step = init_step
        self.running = False
        self.pendings: Dict[int, asyncio.Future[None]] = {}

    def can_operate(self, step: int, is_prefill: bool):
        if is_prefill:
            return self.step == 0
        else:
            return step == self.step and step > 0

    async def can_continue_or_reoder(self, step: int, is_prefill: bool, on_reorder_start: Callable[[], None], on_reorder_end: Callable[[], None]) -> bool:
        if step < self.step+(1 if self.running else 0):
            return False
        if is_prefill:
            return self.step == 0 and (not self.running)
        if step == self.step:
            assert not self.running
            return True
        if step in self.pendings:
            return False
        # print(f"{self.name}: trigger reorder: {step}")
        fut: asyncio.Future[None] = asyncio.Future()
        self.pendings[step] = fut
        on_reorder_start()
        await fut
        on_reorder_end()
        # print(f"finish reorder: {step}")
        return await self.can_continue_or_reoder(step, is_prefill, on_reorder_start, on_reorder_end)

    def set_running(self):
        assert not self.running
        self.running = True

    def add_complete_task(
        self,
        step: int,
        is_prefill: bool,
        count: int = 1,
    ):
        if count <= 0:
            raise ValueError("completed inference task count must be positive")
        if is_prefill and count != 1:
            raise ValueError("prefill completion count must be one")
        self.running = False
        if is_prefill:
            self.step = step+1
        else:
            self.step += count
        if self.step in self.pendings:
            fut = self.pendings[self.step]
            del self.pendings[self.step]
            fut.set_result(None)

    def shallow_copy(self):
        return InferTaskReorderAndDropCtrl(init_step=self.step)
