from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

NO_GC = Optional[Any]


# class FasterDeSerializeObject(ABC):
#     @abstractmethod
#     def full_recover(self, large_data: List[Union[bytes, memoryview]], no_gc: NO_GC) -> Tuple["FasterSerializeObject", int]:
#         raise NotImplementedError


class FasterSerializeObject(ABC):
    @abstractmethod
    def faster_reduce(self) -> Tuple[Any, List[Union[bytes, memoryview]], NO_GC]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def faster_recover(extras: Any, large_data: List[Union[bytes, memoryview]], no_gc: NO_GC) -> "FasterSerializeObject":
        raise NotImplementedError
