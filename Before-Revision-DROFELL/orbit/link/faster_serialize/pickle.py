from inspect import isclass
import io
import threading
from .obj import NO_GC, FasterSerializeObject
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union


class _FastUnpicklerOuterData:
    def __init__(self) -> None:
        self.large_datas: List[Union[bytes, memoryview]] = []
        self.consumed: int = -1
        self.no_gc: NO_GC = None

    def unused(self):
        return self.large_datas[self.consumed+1:]

    def clear(self):
        self.large_datas = []
        self.consumed = -1
        self.no_gc = None


class FasterUnpickler(pickle.Unpickler):
    _thread_outer_data: Dict[int, _FastUnpicklerOuterData] = {}
    cached_class: Dict[Tuple[str, str], Any] = {}

    @staticmethod
    def get_outer_data():
        tid = threading.get_ident()
        od = FasterUnpickler._thread_outer_data.get(tid)
        if od is None:
            od = _FastUnpicklerOuterData()
            FasterUnpickler._thread_outer_data[tid] = od
        return od

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def find_class(self, module: str, name: str):
        key = (module, name)
        cls = self.cached_class.get(key)
        if cls is None:
            cls = super().find_class(module, name)
            assert cls is not None
            if isclass(cls) and issubclass(cls, FasterSerializeObject):
                def wrapped_constructor(idxs: List[int], extras: Any):
                    outer = FasterUnpickler.get_outer_data()
                    outer.consumed = max(outer.consumed, *idxs)
                    large_datas = [outer.large_datas[i] for i in idxs]
                    no_gc = outer.no_gc
                    return cls.faster_recover(extras, large_datas, no_gc)
                self.cached_class[key] = wrapped_constructor
                return wrapped_constructor
            else:
                self.cached_class[key] = cls
        return cls

    @staticmethod
    def loads(datas: List[Union[bytes, memoryview]], no_gc: Optional[Any] = None):
        up = FasterUnpickler(io.BytesIO(datas[0]))
        outer = FasterUnpickler.get_outer_data()
        outer.large_datas = datas[1:]
        outer.no_gc = no_gc
        obj = up.load()
        unused = outer.unused()
        consumed = outer.consumed+2
        outer.clear()
        return (obj, unused, consumed)


class FasterPickler(pickle.Pickler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, protocol=-1)
        self.large_datas: List[Union[bytes, memoryview]] = []
        self.no_gcs: NO_GC = None

    def reducer_override(self, obj: object):  # type:ignore
        if isinstance(obj, FasterSerializeObject):
            deobj, datas, no_gc = obj.faster_reduce()
            idxs: List[int] = []
            if no_gc is not None:
                if self.no_gcs is None:
                    self.no_gcs = []
                self.no_gcs.append(no_gc)
            for d in datas:
                idx = len(self.large_datas)
                self.large_datas.append(d)
                idxs.append(idx)
            return (obj.__class__, (idxs, deobj))

        else:
            return NotImplemented

    @staticmethod
    def dumps(obj: Any) -> Tuple[List[Union[bytes, memoryview]], Optional[Any]]:
        f = io.BytesIO()
        p = FasterPickler(f)
        p.dump(obj)
        return [f.getvalue()]+p.large_datas, p.no_gcs
