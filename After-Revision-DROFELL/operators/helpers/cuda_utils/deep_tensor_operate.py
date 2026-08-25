import torch
from typing import Callable
import types
from typing import Any, Callable, Optional, Set
import weakref

from dealers.common.base_types import DeviceTensorCarrier
from scripts.offline.testbed.infer_node import Navigate

from scripts.offline.testbed.infer_node import InferTask

ignore_types = set([
    type(None), int, float, bool, complex, str,
    bytes, frozenset, type, property,
    types.BuiltinFunctionType, type(Ellipsis), type(NotImplemented),
    types.FunctionType, weakref.ref
])


class TensorChecker:
    def __init__(self, check_fn: Callable[[torch.Tensor], None]) -> None:
        self.memo: Set[int] = set()
        self.check_fn = check_fn

    def __call__(self, x: Any) -> None:
        d = id(x)
        if d in self.memo:
            return
        cls = type(x)
        if cls == torch.Tensor:
            self.check_fn(x)
        if cls in ignore_types:
            return
        if cls in [tuple, slice, range, list]:
            for v in x:
                self(v)
            return
        if cls == dict:
            for k, v in x.items():
                self(k)
                self(v)
            return
        if cls == types.MethodType:
            self(x.__func__)
            return
        reductor = getattr(x, "__reduce_ex__", None)
        if reductor is not None:
            rv = reductor(4)
        else:
            reductor = getattr(x, "__reduce__", None)
            if reductor:
                rv = reductor()
            else:
                raise ValueError(x)
        if isinstance(rv, str):
            return
        else:
            rv = rv[1:]
            for v in rv:
                self(v)


def new_checker(check_fn: Callable[..., Optional[Any]]):
    checker = TensorChecker(check_fn)
    return checker


def _check_no_cuda_tensor(t: torch.Tensor):
    assert not t.is_cuda


check_no_cuda_tensor = new_checker(_check_no_cuda_tensor)


class TensorConvertor:
    def __init__(self, convert_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.memo: Set[int] = set()
        self.convert_fn = convert_fn

    def __call__(self, x: Any) -> Any:
        d = id(x)
        if d in self.memo:
            return d
        cls = type(x)
        if cls == torch.Tensor:
            return self.convert_fn(x)
        elif cls in ignore_types:
            return x
        elif cls in [tuple, slice, range, list]:
            l = [
                self(v)
                for v in x
            ]
            return cls(l)
        elif cls == dict:
            for k, v in x.items():
                x[self(k)] = self(v)
            return x
        else:
            print(cls)
            for k, v in x.__dict__.items():
                x.__dict__[k] = self(v)
            return x
        # if cls == types.MethodType:
        #     self(x.__func__)
        #     return
        # reductor = getattr(x, "__reduce_ex__", None)
        # if reductor is not None:
        #     rv = reductor(4)
        # else:
        #     reductor = getattr(x, "__reduce__", None)
        #     if reductor:
        #         rv = reductor()
        #     else:
        #         raise ValueError(x)
        # if isinstance(rv, str):
        #     return
        # else:
        #     rv = rv[1:]
        #     for v in rv:
        #         self(v)


base_data = InferTask(
    "seq_id",
    step=100,
    navi=Navigate(
        navigate=["model:head", "model:tail"],
        current=1,
    ),
    is_prefill=True,
    input=DeviceTensorCarrier(torch.zeros((10,)), no_gc=None),
    output=(0, {1: 0})
)

print(base_data.__repr__())
cvt_data = TensorConvertor(lambda x: x)(base_data)
print(cvt_data.__repr__())
