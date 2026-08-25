
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Union, Tuple, cast

from ..real_basic.memory import MemoryConfigBasic

from operators.core.memory.env import UNMANAGED_POOL, Env
from utils.format import format_size

Rule = Union[
    Tuple[Literal[
        "fix",  # fix size, never change, specific total size at lim
        "share",  # allow to access all available memory, lim=0
        "limit-min",  # preserve lim size of memory for this node, add allow to access all available memory
        "limit-max"
    ], int],
    Tuple[Literal["limit-both"], Tuple[int, int]]
]


class MemoryLeaf:
    def __init__(
        self,
        get_consumed_fn: Callable[[], int]
    ) -> None:
        self.get_consumed_fn = get_consumed_fn
        self._consumed = 0
        self.available: int = 0
        self.total: int = 0

    def update_consumed(self) -> int:
        self._consumed = self.get_consumed_fn()
        return self._consumed

    def update_available(self, total: int):
        self.total = total
        self.available = self.total-self._consumed

    def __str__(self):
        return f"Total:{format_size(self.total)}=Consumed:{format_size(self._consumed)}+Available:{format_size(self.available)}"

    def __repr__(self) -> str:
        return self.__str__()


class MemoryNode:
    def __init__(
        self, **kwargs: Tuple[Rule, Union[MemoryLeaf, "MemoryNode"]]
    ) -> None:
        self._childrens: Dict[str, Tuple[Rule, Union[MemoryLeaf, MemoryNode]]] = {}
        self._leaves: Dict[str, MemoryLeaf] = {}
        self._consumed: Dict[str, int] = {}
        self._total_consumed: int = 0
        self.total: int = 0
        self.available: int = 0

        for name, (rule, node) in kwargs.items():
            self.add_node(name, rule, node)

    @property
    def leaves(self):
        return self._leaves

    @property
    def oom(self):
        for n in self.leaves.values():
            if n.available < 0:
                return True
        return False

    def oom_items(self) -> Dict[str, MemoryLeaf]:
        return {k: n for k, n in self.leaves.items() if n.available < 0}

    def add_node(self, name: str, rule: Rule, node: Union[MemoryLeaf, "MemoryNode"]):
        self._childrens[name] = (rule, node)
        if rule[0] == "share":
            assert rule[1] == 0
        if isinstance(node, MemoryLeaf):
            self._leaves[name] = node
            return
        for _name, _leaf in node.leaves.items():
            self._leaves[f"{name}.{_name}"] = _leaf

    def update_consumed(self) -> int:
        self._total_consumed = 0
        for name, ((rule, lim), node) in self._childrens.items():
            consumed = node.update_consumed()
            if rule == "fix":
                consumed = cast(int, lim)
            elif rule == "limit-min":
                consumed = max(cast(int, lim), consumed)
            elif rule == "limit-both":
                consumed = max(cast(Tuple[int, int], lim)[0], consumed)
            self._consumed[name] = consumed
            self._total_consumed += consumed
        return self._total_consumed

    def update_available(self, total: int):
        self.total = total
        self.available = self.total-self._total_consumed
        for name, ((rule, lim), node) in self._childrens.items():
            if rule == "fix":
                node.update_available(cast(int, lim))
                continue
            consumed = self._consumed[name]
            if rule == "limit-min" or rule == "share":
                node.update_available(self.available+consumed)
            elif rule == "limit-both":
                total = min(cast(Tuple[int, int], lim)[1], self.available+consumed)
                node.update_available(total)
            else:
                total = min(cast(int, lim), self.available+consumed)
                node.update_available(total)

    def update(self, total: int):
        self.update_consumed()
        self.update_available(total)

    def __str__(self):
        s = f"Total:{format_size(self.total)}=Consumed:{format_size(self._total_consumed)}+Available:{format_size(self.available)}"
        for name, ((rule, lim), node) in self._childrens.items():
            node_str = str(node).replace('\n', '\n|  ')
            lims = f"{format_size(lim)}"if isinstance(lim, int) else f"{format_size(lim[0])}-{format_size(lim[1])}"
            s += f"\n|-{name} ({rule},{lims})" +\
                f" -> Modeling As: {format_size(self._consumed[name])}: \n" +\
                f"|  {node_str}"
        if self.oom:
            for name, node in self.leaves.items():
                if node.available < 0:
                    s += f"\nOOM: {name} {node}"
        return s


class MemoryConfig(ABC, MemoryConfigBasic):
    @property
    @abstractmethod
    def train_ctx_memory_key(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def kvcache_memory_key(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def restriction(self, env: Env) -> MemoryNode:
        raise NotImplementedError


@dataclass
class MemoryConfigStatic(MemoryConfig):
    major_reserved: int = 8*(2**30)
    major_percentage: float = 0.8
    unmanaged_reserved: int = 32*(2**20)
    minor_maximum: int = 4*(2**30)
    minor_minimum: int = 0
    operate_reserved: int = 1*(2**30)

    @property
    def train_ctx_memory_key(self) -> str:
        return "major.train_ctx"

    @property
    def kvcache_memory_key(self) -> str:
        return "major.kvcache"

    def restriction(self, env: Env) -> MemoryNode:
        pool = env.pools

        def _wrap(name: str):
            def get_fn():
                return pool[name]
            return get_fn
        root = MemoryNode(
            base=(("share", 0), MemoryLeaf(_wrap(self.base_models_name))),
            reserved=(("fix", self.operate_reserved), MemoryLeaf(lambda: 0)),
            unmanaged=(("limit-min", self.unmanaged_reserved), MemoryLeaf(_wrap(UNMANAGED_POOL))),
            major=(("limit-min", self.major_reserved), MemoryNode(
                train_ctx=(("share", 0), MemoryLeaf(_wrap(self.train_ctx_name))),
                kvcache=(("share", 0), MemoryLeaf(_wrap(self.kv_cache_name))),
            )),
            minor=(("limit-both", (self.minor_minimum, self.minor_maximum)), MemoryNode(
                user_model=(("share", 0), MemoryLeaf(_wrap(self.train_bindings_name))),
                grad_and_optim=(("share", 0), MemoryLeaf(_wrap(self.train_grad_optim_name))),
            ))
        )
        return root


@dataclass
class MemoryConfigPercentage(MemoryConfig):
    major_percentage: float = 0.8
    unmanaged_reserved: int = 32*(2**20)
    minor_maximum: int = 4*(2**30)
    minor_minimum: int = 0
    operate_reserved: int = 1*(2**30)

    @property
    def train_ctx_memory_key(self) -> str:
        return "major.train_ctx"

    @property
    def kvcache_memory_key(self) -> str:
        return "major.kvcache"

    def restriction(self, env: Env) -> MemoryNode:
        base_model_memory = env.pools.get(self.base_models_name, 0)
        root_memory = self.root_memory
        major_reserved = int(self.major_percentage*(root_memory-base_model_memory-self.operate_reserved))
        pool = env.pools

        def _wrap(name: str):
            def get_fn():
                return pool[name]
            return get_fn
        root = MemoryNode(
            base=(("share", 0), MemoryLeaf(_wrap(self.base_models_name))),
            reserved=(("fix", self.operate_reserved), MemoryLeaf(lambda: 0)),
            unmanaged=(("limit-min", self.unmanaged_reserved), MemoryLeaf(_wrap(UNMANAGED_POOL))),
            major=(("limit-min", major_reserved), MemoryNode(
                train_ctx=(("share", 0), MemoryLeaf(_wrap(self.train_ctx_name))),
                kvcache=(("share", 0), MemoryLeaf(_wrap(self.kv_cache_name))),
            )),
            minor=(("limit-both", (self.minor_minimum, self.minor_maximum)), MemoryNode(
                user_model=(("share", 0), MemoryLeaf(_wrap(self.train_bindings_name))),
                grad_and_optim=(("share", 0), MemoryLeaf(_wrap(self.train_grad_optim_name))),
            ))
        )
        return root
