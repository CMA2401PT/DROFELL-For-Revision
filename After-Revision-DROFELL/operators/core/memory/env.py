from abc import ABC, abstractmethod
import collections
from typing import Any, Callable, Deque, Dict, Generic, List, TypeVar, Union

import torch

from dealers.common.base_types import DeviceTensorCarrier, TensorMeta
from .static import MemoryItem
from utils.format import format_size

UNMANAGED_POOL = "unmanaged"

MemoryItemType = TypeVar("MemoryItemType", bound=MemoryItem)


class Env:
    def __init__(self) -> None:
        self.memory_holders: Dict[int, MemoryHolder[Any]] = {}
        self.pools: Dict[str, int] = collections.defaultdict(lambda: 0)
        self.revert_stack: Deque[Callable[[], None]] = collections.deque()
        self.childrens: Dict[Any, Any] = {}
        self.actuals: Dict[int, MemoryHolder[Any]] = {}
        self.snapshots: List[Snapshot] = []
        self.unmanaged = MemoryDict(self)
        self.set(UNMANAGED_POOL, self.unmanaged)
        self.merge_all()

    @property
    def total(self):
        return sum(self.pools.values())

    def trace(self, item: MemoryItemType, pool: str = UNMANAGED_POOL) -> "MemoryHolder[MemoryItemType]":
        return MemoryHolder(item=item, env=self, pool=pool)

    def add_holder(self, holder: "MemoryHolder[Any]"):
        self.memory_holders[id(holder)] = holder
        self.pools[holder.pool] += holder.size

    def remove_holder(self, holder: "MemoryHolder[Any]"):
        del self.memory_holders[id(holder)]
        self.pools[holder.pool] -= holder.size

    def retrack(self, actual: Union[int, DeviceTensorCarrier, TensorMeta, torch.Tensor]):
        if isinstance(actual, (torch.Tensor, tuple)):
            actual = id(actual)
        elif not isinstance(actual, int):
            actual = actual.id
        h = self.actuals[actual]
        del self.unmanaged[actual]
        return h

    def untrack(self, holder: "MemoryHolder[Any]", actual: Union[int, DeviceTensorCarrier, TensorMeta, torch.Tensor]):
        if isinstance(actual, (torch.Tensor, tuple)):
            actual = id(actual)
        elif not isinstance(actual, int):
            actual = actual.id
        self.actuals[actual] = holder
        self.unmanaged.set(actual, holder)
        self.merge_all()

    def vuntrack(self, holder: "MemoryHolder[Any]"):
        self.unmanaged.set(id(holder), holder)

    def snapshot(self) -> "Snapshot":
        snap = Snapshot(self, len(self.revert_stack))
        self.snapshots.append(snap)
        return snap

    def merge_all(self):
        self.revert_stack = collections.deque()
        for s in self.snapshots:
            s.pos = -1
        self.snapshots = []

    def revert_all(self):
        for r in list(self.revert_stack)[::-1]:
            r()
        self.revert_stack = collections.deque()
        for s in self.snapshots:
            s.pos = -1
        self.snapshots = []

    def _set(self, name: str, value: "MemoryNode") -> None:
        assert not name in self.childrens
        if name in self.childrens:
            self.childrens[name].dec_ref()
        value.inc_ref()
        self.childrens[name] = value

    def _remove(self, name: str):
        value = self.childrens[name]
        value.dec_ref()
        del self.childrens[name]

    def set(self, name: Any, value: "MemoryNode") -> None:
        assert name not in self.childrens
        value.inc_ref()
        self.childrens[name] = value

        def revert():
            self._remove(name)
        self.revert_stack.append(revert)

    def get(self, name: Any):
        return self.childrens[name]

    def remove(self, name: Any) -> None:
        value = self.get(name)
        value.dec_ref()
        del self.childrens[name]

        def revert():
            self._set(name, value)
        self.revert_stack.append(revert)

    def _format_pool(self) -> str:
        return ", ".join(
            [f"{k}={format_size(v)}" for k, v in self.pools.items()])

    def __str__(self) -> str:
        return f"<Env: {self._format_pool()}>"

    def __repr__(self) -> str:
        return f"<Env: {self._format_pool()}>"

    def __delitem__(self, name: Any):
        self.remove(name)

    def __getitem__(self, name: Any):
        return self.childrens[name]

    def __contains__(self, item: Any):
        return self.childrens.__contains__(item)

    def total_memory(self) -> int:
        return sum(v.total_memory() for v in self.childrens.values() if isinstance(v, MemoryNode))

    def detail_memory(self) -> str:
        s = f"{format_size(self.total_memory())}@root"
        cs = [
            f"|-{k}\n| " +
            v.detail_memory().replace("\n", "\n|  ") for k, v in self.childrens.items() if isinstance(v, MemoryNode)
        ]
        return "\n".join([s, *cs])


class Snapshot:
    def __init__(self, env: Env, pos: int) -> None:
        self.env = env
        self.pos = pos

    @property
    def valid(self):
        return self.pos >= 0

    def apply(self):
        assert self.valid
        while len(self.env.revert_stack) > self.pos:
            op = self.env.revert_stack.pop()
            op()

    def merge(self):
        assert self.valid
        n_to_remove = self.pos
        while n_to_remove > 0:
            self.env.revert_stack.popleft()
            remove = False
            for s in self.env.snapshots:
                s.pos -= 1
                if s.pos < 0:
                    remove = True
            if remove:
                self.env.snapshots = [s for s in self.env.snapshots if s.valid]
            n_to_remove -= 1


class MemoryNode(ABC):
    def __init__(self, env: Env) -> None:
        self.ref_count = 0
        self.env = env

    @property
    def in_env(self):
        return self.ref_count > 0

    def inc_ref(self):
        self.ref_count += 1
        if self.ref_count == 1:
            self.on_connect_env()

    def dec_ref(self):
        assert self.in_env
        self.ref_count -= 1
        if self.ref_count == 0:
            self.on_disconntect_env()

    @abstractmethod
    def on_connect_env(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_disconntect_env(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def total_memory(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def detail_memory(self) -> str:
        raise NotImplementedError


class MemoryItemValue(MemoryItem):
    def __init__(self, bytes: int) -> None:
        self._memory = bytes

    @property
    def memory(self) -> int:
        return self._memory

    def __str__(self) -> str:
        return f"<Memory: {format_size(self._memory)}>"

    def __repr__(self) -> str:
        return f"<Memory: {format_size(self._memory)}>"


class MemoryHolder(MemoryNode, Generic[MemoryItemType]):
    def __init__(self, item: MemoryItemType, env: Env, pool: str = UNMANAGED_POOL) -> None:
        super().__init__(env)
        self.item = item
        self._pool = pool

    def _change_pool(self, pool: str):
        orig_pool = self._pool
        self._pool = pool
        if self.in_env:
            self.env.pools[orig_pool] -= self.size
            self.env.pools[pool] += self.size

    def change_pool(self, pool: str):
        if pool == self._pool:
            return
        orig_pool = self._pool
        self._change_pool(pool)

        def revert():
            self._change_pool(orig_pool)
        self.env.revert_stack.append(revert)

    def on_connect_env(self) -> None:
        self.env.add_holder(self)

    def on_disconntect_env(self) -> None:
        self.env.remove_holder(self)

    def __str__(self) -> str:
        return f"<Item:{self.item.__str__()}@{self.pool}>"

    def __repr__(self) -> str:
        return f"<Item:{self.item.__repr__()}@{self.pool}>"

    @property
    def pool(self):
        return self._pool

    @property
    def size(self):
        return self.item.memory

    def total_memory(self) -> int:
        return self.item.memory

    def detail_memory(self) -> str:
        return f"{format_size(self.total_memory())}@{self.pool}"


class MemoryDict(MemoryNode):
    def __init__(self, env: Env, default_pool: str = UNMANAGED_POOL) -> None:
        super().__init__(env)
        self.childrens: Dict[Any, Any] = {}
        self.default_pool = default_pool

    def __len__(self):
        return len(self.childrens)

    @property
    def pool(self):
        return self.default_pool

    def on_connect_env(self) -> None:
        for c in self.childrens.values():
            if isinstance(c, MemoryNode):
                c.inc_ref()

    def on_disconntect_env(self) -> None:
        for c in self.childrens.values():
            if isinstance(c, MemoryNode):
                c.dec_ref()

    def _set(self, name: Any, value: Any) -> None:
        assert not name in self.childrens
        if self.in_env:
            if isinstance(value, MemoryNode):
                value.inc_ref()
        self.childrens[name] = value

    def _remove(self, name: Any):
        value = self.childrens[name]
        if self.in_env:
            if isinstance(value, MemoryNode):
                value.dec_ref()
        del self.childrens[name]

    def _circulate_check(self, v: int):
        for c in self.childrens.values():
            if isinstance(c, MemoryDict):
                if c._circulate_check(v):
                    return True
                elif id(c) == v:
                    return True
        return False

    def set(self, name: Any, value: Any) -> None:
        if name in self.childrens:
            self.remove(name)
        if isinstance(value, MemoryNode):
            if self.in_env:
                value.inc_ref()
        self.childrens[name] = value

        # assert not self._circulate_check(id(self))

        def revert():
            self._remove(name)
        self.env.revert_stack.append(revert)

        if isinstance(value, (MemoryDict, MemoryHolder)) and value.pool == UNMANAGED_POOL:
            value.change_pool(self.default_pool)

    def set_item(self, name: Union[int, str], item: MemoryItem):
        holder = MemoryHolder(item, self.env, self.default_pool)
        self.set(name, holder)
        return holder

    def get(self, name: Any):
        return self.childrens[name]

    def __getitem__(self, name: Any):
        return self.childrens[name]

    def __contains__(self, item: Any):
        return self.childrens.__contains__(item)

    # def get_item(self, name: Union[int, str]):
    #     return cast(MemoryHolder[MemoryItem], self.childrens[name]).item

    def remove(self, name: Any) -> None:
        if name not in self.childrens:
            return
        value = self.childrens[name]
        if isinstance(value, MemoryNode):
            if self.in_env:
                value.dec_ref()
        del self.childrens[name]

        def revert():
            self._set(name, value)
        self.env.revert_stack.append(revert)

    def __delitem__(self, name: Any):
        self.remove(name)

    def change_pool(self, pool: str):
        if pool == self.default_pool:
            return
        orig_pool = self.default_pool
        self.default_pool = pool
        for v in self.childrens.values():
            if isinstance(v, (MemoryDict, MemoryHolder)) and v.pool == UNMANAGED_POOL:
                v.change_pool(pool)

        def revert():
            self.default_pool = orig_pool
        self.env.revert_stack.append(revert)

    def total_memory(self) -> int:
        return sum(v.total_memory() for v in self.childrens.values() if isinstance(v, MemoryNode))

    def detail_memory(self) -> str:
        s = f"{format_size(self.total_memory())}@{self.pool}"
        cs = [
            f"|-{k}\n| " +
            v.detail_memory().replace("\n", "\n|  ") for k, v in self.childrens.items() if isinstance(v, MemoryNode)
        ]
        return "\n".join([s, *cs])
