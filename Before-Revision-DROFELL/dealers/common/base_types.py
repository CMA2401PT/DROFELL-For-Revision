from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from orbit.link import FasterSerializeObject
from cuda_extends.hack import tensor_from_memory_view, tensor_to_memory_view


@dataclass
class TensorMeta:
    id: int
    ptr: int
    dtype: torch.dtype
    is_cpu: bool
    shape: Tuple[int, ...]

    def __str__(self):
        return f"<Tensor Meta: {self.shape},{self.dtype},"+("cpu>" if self.is_cpu else "device>")


class DeviceTensorCarrier(FasterSerializeObject):
    def __init__(self, t: torch.Tensor, no_gc: Optional[Any]) -> None:
        self.t = t
        self.no_gc = no_gc

    @classmethod
    def place_holder(cls):
        return DeviceTensorCarrier(None, None)  # type:ignore

    @classmethod
    def from_data(cls, data: Union[torch.Tensor, 'DeviceTensorCarrier']):
        if isinstance(data, DeviceTensorCarrier):
            return data.shallow_copy()
        return DeviceTensorCarrier(data, None)

    def __str__(self):
        ngc = f", WithNoGC" if self.no_gc is not None else ""
        if self.t is None:
            return f"<Tensor Carrier: Empty{ngc}>"
        else:
            return f"<Tensor Carrier: {self.t}{ngc}>"

    def __repr__(self) -> str:
        return self.__str__()

    def free(self):
        self.t = None  # type:ignore
        self.no_gc = None

    def shallow_copy(self):
        return DeviceTensorCarrier(self.tensor, self.no_gc)

    @property
    def id(self):
        return 0 if self.t is None else id(self.t)

    @property
    def ptr(self):
        return 0 if self.t is None else self.t.data_ptr()

    def data_ptr(self):
        return 0 if self.t is None else self.t.data_ptr()

    @property
    def dtype(self):
        return self.t.dtype  # type:ignore

    @property
    def is_cpu(self):
        return self.t.is_cpu  # type:ignore

    @property
    def shape(self):
        return self.t.shape  # type:ignore

    @property
    def meta(self):
        try:
            return TensorMeta(self.id, self.ptr, self.dtype, self.is_cpu, tuple(self.shape))
        except:
            return TensorMeta(self.id, 0, None, None, None)  # type:ignore

    @property
    def tensor(self) -> torch.Tensor:
        return self.t  # type:ignore

    @property
    def tensor_memory_view(self) -> memoryview:
        return tensor_to_memory_view(self.tensor)

    def faster_reduce(self) -> Tuple[Any, List[Union[bytes, memoryview]], Optional[Any]]:
        assert self.t is not None
        assert self.t.dtype == torch.float16 or self.t.dtype == torch.bfloat16
        assert self.t.is_cpu
        return (self.t.size(), self.t.dtype), [tensor_to_memory_view(self.t)], self.no_gc

    @staticmethod
    def faster_recover(extras: Any, large_data: List[Union[bytes, memoryview]], no_gc: Optional[Any]):
        mv = large_data[0]
        if not isinstance(mv, memoryview):
            mv = memoryview(mv)
        sizes, dt = extras
        t = tensor_from_memory_view(mv, sizes, dt)
        s = DeviceTensorCarrier(t, no_gc)
        return s

    def __getstate__(self):
        assert self.t is not None
        return self.t

    def __setstate__(self, state: torch.Tensor):
        self.t = state
        self.no_gc = None

    def debug_check_cpu(self):
        assert self.is_cpu


TokenID = int
LogProb = float
SampleLogProb = Dict[TokenID, LogProb]
SampleBatchID = str

ModelDescribeStr = str


class BaseModelName:
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def head_name(self):
        return self.name+":head"

    @property
    def tail_name(self):
        return self.name+":tail"

    def decoders_group_name(self, s: int, e: int):
        return self.name+f":decoders-{s:0>3}-{e:0>3}"


class ModelDescribe:
    def __init__(self, str_or_base: Optional[str] = None, is_head: bool = False, is_tail: bool = False, start: int = -1, end: int = -1) -> None:
        self._base_model: str = ""
        self._part: str = ""
        self._start_decoder: int = -1
        self._end_decoder: int = -1

    @property
    def base_model(self):
        assert self._base_model is not None
        return self._base_model

    @property
    def part(self):
        assert self._part is not None
        return self._part

    def __str__(self):
        return self.base_model+":"+self.part

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def is_head(self):
        return self._part == "head"

    @property
    def is_tail(self):
        return self._part == "tail"

    @property
    def is_decoders(self):
        return self._part.startswith("decoders-")

    @property
    def num_decoders(self):
        if self.is_decoders:
            s, e = self.decoders_range
            return e-s
        else:
            return -1

    @property
    def decoders_range(self):
        return self._start_decoder, self._end_decoder

    @classmethod
    def from_str(cls, describe_str: str):
        base_model_name, part = describe_str.split(":")
        d = cls()
        d._part = part
        d._base_model = base_model_name
        if not part.startswith("decoders-"):
            assert part == "head" or part == "tail"
            return d
        _, s, e = part.split("-")
        assert len(s) == 3
        assert len(e) == 3
        d._start_decoder = int(s)
        d._end_decoder = int(e)
        return d

    @classmethod
    def new(cls, base_model: str, is_head: bool = False, is_tail: bool = False, start: int = -1, end: int = -1):
        d = cls()
        assert base_model != ""
        d._base_model = base_model
        if is_head:
            d._part = "head"
        elif is_tail:
            d._part = "tail"
        else:
            assert start != -1 and end != -1
            d._start_decoder = start
            d._end_decoder = end
            d._part = f"decoders-{start:0>3}-{end:0>3}"
        return d


def group_model_with_split(base_model_name: str, decoders: int, group_fn: Callable[[int], List[int]]):
    group_sizes = group_fn(decoders)
    group = [
        (sum(group_sizes[:i]), sum(group_sizes[:i+1]))
        for i in range(len(group_sizes))
    ]
    group_names: List[str] = [
        BaseModelName(base_model_name).decoders_group_name(s, e)
        for s, e in group
    ]
    head_name, tail_name =\
        BaseModelName(base_model_name).head_name, BaseModelName(base_model_name).tail_name
    return head_name, group_names, tail_name


def group_model(base_model_name: str, decoders: int, group_fn: Callable[[int], List[int]]):
    head, group_names, tail = group_model_with_split(base_model_name, decoders, group_fn)
    return [head, *group_names, tail]


def gen_user_model_name(base_model: str, user_model_prefix: str):
    desc = ModelDescribe.from_str(base_model)
    new_desc = ModelDescribe.new(user_model_prefix, desc.is_head, desc.is_tail, *desc.decoders_range)
    return new_desc.__str__()


def gen_user_model_names(models: List[str], user_model_prefix: str):
    pairs: List[Tuple[str, str]] = []
    for model in models:
        pairs.append((model, gen_user_model_name(model, user_model_prefix)))
    return pairs


def balance_divide(
    base_models: List[str],
    with_head: bool,
    with_tail: bool,
    decoders_groups: int,
    user_model_prefix: str
) -> List[List[Tuple[str, str]]]:
    base_models_describe = [
        ModelDescribe.from_str(desc)
        for desc in base_models
    ]
    head: Optional[ModelDescribe] = None
    tail: Optional[ModelDescribe] = None
    decoders: List[ModelDescribe] = []
    for d in base_models_describe:
        if d.is_head:
            head = d
        elif d.is_tail:
            tail = d
        else:
            decoders.append(d)
    decoders = sorted(decoders, key=lambda d: str(d))
    assert len(decoders) % decoders_groups == 0
    group_size = len(decoders) // decoders_groups
    groups: List[List[Tuple[str, str]]] = [[] for _ in range(decoders_groups)]
    for group_i in range(decoders_groups):
        start, end = group_i*group_size, (group_i+1)*group_size
        src = decoders[start:end]
        dst: List[ModelDescribe] = []
        for s in src:
            a, b = s.decoders_range
            dst.append(ModelDescribe.new(user_model_prefix, start=a, end=b))
        groups[group_i] = [
            (str(s), str(d)) for s, d in zip(src, dst)
        ]
    dst_head = ModelDescribe.new(user_model_prefix, is_head=True)
    dst_tail = ModelDescribe.new(user_model_prefix, is_tail=True)
    if with_head:
        groups[0].insert(0, (str(head), str(dst_head)))
    else:
        groups.insert(0, [(str(head), str(dst_head))])

    if with_tail:
        groups[-1].append((str(tail), str(dst_tail)))
    else:
        groups.append([(str(tail), str(dst_tail))])
    return groups
