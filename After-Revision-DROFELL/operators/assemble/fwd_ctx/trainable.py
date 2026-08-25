from abc import abstractmethod, ABC
from typing import List, Optional, Tuple
import torch

from ...core.memory.env import Env, MemoryDict, MemoryHolder

from ...core.lora import LoraRenderConfig
from ...core.memory.static import MemoryDType, MemoryModule, MemoryTensor
from ...core.swappable.real import CanSwapObject, NoSwapAny


class TrainableUserModel(ABC):
    @abstractmethod
    def train_forward(self, dtype: torch.dtype, reproduce_seed: Optional[int]) -> Tuple[CanSwapObject, List[torch.Tensor], int]:
        raise NotImplementedError

    @abstractmethod
    def train_backward(self, ctx: CanSwapObject, grad: List[torch.Tensor]) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def mode(self) -> Optional[LoraRenderConfig]:
        # if is ptuning mode, return None
        # is is lora mode, return lora config
        return None


class PlaceHolderTrainableUserModel:
    def __init__(self, tensors: List[torch.Tensor]) -> None:
        self.tensors = tensors

    def train_forward(self, dtype: torch.dtype, reproduce_seed: Optional[int]) -> Tuple[CanSwapObject, List[torch.Tensor], int]:
        return NoSwapAny(None), self.tensors, 0

    def train_backward(self, ctx: CanSwapObject, grad: List[torch.Tensor]) -> None:
        pass

    @property
    def mode(self) -> Optional[LoraRenderConfig]:
        return None


class TrainableUserMemoryModel(MemoryModule):
    @abstractmethod
    def train_forward(self, env: Env, dtype: MemoryDType) -> Tuple[MemoryDict, List[MemoryHolder[MemoryTensor]]]:
        raise NotImplementedError

    @abstractmethod
    def train_backward(self, env: Env, ctx: MemoryDict, grad: MemoryHolder[MemoryTensor]) -> MemoryHolder[MemoryTensor]:
        raise NotImplementedError

    @property
    @abstractmethod
    def mode(self) -> Optional[LoraRenderConfig]:
        # if is ptuning mode, return None
        # is is lora mode, return lora config
        return None
