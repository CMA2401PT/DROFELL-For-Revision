import collections
from typing import Dict, Optional
import extend_asyncio as asyncio


class FreeStateOpMantainer:
    def __init__(self) -> None:
        self.free_op: Dict[str, int] = collections.defaultdict(lambda: 0)
        self.managed_op: Dict[str, int] = collections.defaultdict(lambda: 0)
        self.barrier: Optional[asyncio.Future[None]] = None

    def add_free_op(self, name: str):
        self.free_op[name] += 1
        if self.barrier is None:
            self.barrier = asyncio.Future()

    def remove_free_op(self, name: str):
        self.free_op[name] -= 1
        c = self.free_op[name]
        assert c >= 0
        if c == 0:
            del self.free_op[name]
        if len(self.free_op) == 0 and self.barrier is not None:
            b = self.barrier
            self.barrier = None
            b.set_result(None)

    def add_managed_op(self, name: str):
        self.free_op[name] -= 1
        c = self.free_op[name]
        assert c >= 0
        if c == 0:
            del self.free_op[name]
        self.managed_op[name] += 1
        if len(self.free_op) == 0 and self.barrier is not None:
            b = self.barrier
            self.barrier = None
            b.set_result(None)

    def remove_managed_op(self, name: str):
        self.managed_op[name] -= 1
        c = self.managed_op[name]
        assert c >= 0
        if c == 0:
            del self.managed_op[name]
        self.free_op[name] += 1
        if self.barrier is None:
            self.barrier = asyncio.Future()

    async def wait(self):
        if self.barrier is None:
            return
        else:
            b = self.barrier
            await b
