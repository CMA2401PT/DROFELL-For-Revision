from math import prod
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union, Sequence
from abc import ABC, abstractmethod
import torch
from dealers.common.base_types import DeviceTensorCarrier
from utils.format import format_size

MemoryType = Tuple[str, int]
MemoryFloat16 = ("float16", 2)
MemoryBFloat16 = ("bfloat16", 2)
MemoryFloat32 = ("float32", 4)
MemoryInt32 = ("int32", 4)
MemoryLong = ("int64", 8)
MemoryInt64 = ("int64", 8)
_torch_dtypes: Dict[int, MemoryType] = {}
MemoryDType = Union[MemoryType, torch.dtype]


class MemoryItem(ABC):
    @property
    @abstractmethod
    def memory(self) -> int:
        raise NotImplementedError


def get_dtype(dtype: MemoryDType) -> MemoryType:
    if not isinstance(dtype, tuple):
        name = id(dtype)
        if name in _torch_dtypes:
            dtype = _torch_dtypes[name]
        else:
            dtype = ((str(dtype)).removeprefix("torch."), dtype.itemsize)
            _torch_dtypes[name] = dtype
    return dtype


class MemoryTensor(MemoryItem):
    def __init__(self, size: Tuple[int, ...], dtype: MemoryDType = MemoryFloat32, cpu: bool = False) -> None:
        self.size = size
        self.dtype = get_dtype(dtype)
        self.is_cpu = cpu

    @classmethod
    def like(cls, t: "MemoryTensor", size: Optional[Tuple[int, ...]] = None, dtype: Optional[MemoryDType] = None, cpu: Optional[bool] = None):
        return MemoryTensor(size or t.size, dtype or t.dtype, cpu or t.is_cpu)

    @classmethod
    def from_tensor(cls, t: torch.Tensor):
        return MemoryTensor(size=tuple(t.shape), dtype=t.dtype, cpu=t.is_cpu)

    @classmethod
    def from_carrier(cls, t: DeviceTensorCarrier):
        return MemoryTensor(size=tuple(t.shape), dtype=t.dtype, cpu=t.is_cpu)

    @property
    def requested(self):
        if self.is_cpu:
            return 0
        return prod(self.size)*self.dtype[1]

    @property
    def memory(self):
        return self.requested

    def numel(self):
        return prod(self.size)

    def repeat_interleave(self, repeats: int, dim: Optional[int] = None):
        if dim is None:
            return MemoryTensor(size=(self.numel()*repeats,), dtype=self.dtype, cpu=self.is_cpu)
        else:
            return MemoryTensor(size=tuple([(v if d != dim else v*repeats) for d, v in enumerate(self.size)]), dtype=self.dtype, cpu=self.is_cpu)

    @property
    def data(self):
        return self

    def half(self):
        return self.to(MemoryFloat16)

    def cpu(self):
        assert not self.is_cpu
        return MemoryTensor(self.size, self.dtype, cpu=True)

    def device(self):
        assert self.is_cpu
        return MemoryTensor(self.size, self.dtype, cpu=False)

    def to(self, dtype: MemoryDType):
        new_dtype = get_dtype(dtype)
        assert self.dtype != new_dtype
        #     return self
        # else:
        return MemoryTensor(self.size, dtype, cpu=self.is_cpu)

    def __str__(self) -> str:
        return f"<Tensor:{self.size},{self.dtype[0]},Memory:{format_size(self.requested)}>"

    def __repr__(self) -> str:
        return f"<Tensor:{self.size},{self.dtype[0]},Memory:{format_size(self.requested)}>"


class MemoryTensors:
    def __init__(self, tensors: List[MemoryTensor], unique: bool = False) -> None:
        self.tensors: List[MemoryTensor] = tensors
        self.unique = unique

    @property
    def requested(self):
        tensors: Dict[int, MemoryTensor] = {}
        if self.unique:
            return sum(t.requested for t in self.tensors)
        for t in self.tensors:
            tensors[id(t)] = t
        return sum(t.requested for t in tensors.values())

    def __str__(self) -> str:
        return f"<Tensors: Requested:{format_size(self.requested)}>"


class MemoryModule(MemoryItem):
    def __init__(self):
        self._parameters: Dict[str, MemoryTensor] = {}
        self._modules: Dict[str, MemoryModule] = {}
        self._cached_tensors: Optional[MemoryTensors] = None
        self._cached_size: Optional[int] = None

    def __setattr__(self, name: str, value: Union["MemoryModule", MemoryTensor, Any]):
        if isinstance(value, MemoryTensor):
            self._parameters[name] = value
        elif isinstance(value, MemoryModule):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def mount_list(self, l: Sequence["MemoryModule"], prefix: str):
        for i, e in enumerate(l):
            self._modules[f"{prefix}.{i}"] = e

    def parameters(self, memo: Optional[Set[int]] = None) -> Generator[MemoryTensor, Any, Any]:
        if memo is None:
            memo = set()
        for p in self._parameters.values():
            _id = id(p)
            if _id in memo:
                continue
            memo.add(_id)
            yield p
        for m in self._modules.values():
            _id = id(m)
            if _id in memo:
                continue
            memo.add(_id)
            yield from m.parameters(memo)

    def named_parameters(self, prefix: str = "") -> Generator[Tuple[str, MemoryTensor], Any, Any]:
        for name, param in self._parameters.items():
            yield (prefix + name, param)
        for name, module in self._modules.items():
            sub_prefix = prefix + name + "."
            yield from module.named_parameters(prefix=sub_prefix)

    @property
    def parameter_memories(self):
        if self._cached_tensors is None:
            self._cached_tensors = MemoryTensors(list(t for t in self.parameters()))
        return self._cached_tensors

    @property
    def memory(self):
        if self._cached_size is None:
            self._cached_size = self.parameter_memories.requested
        return self._cached_size
