from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar
import torch


class CanSwapObject(ABC):
    @abstractmethod
    # stream for swap out
    def swap_out(self, stream: Optional[torch.cuda.Stream]) -> None:
        raise NotImplementedError

    @abstractmethod
    # device & stream to swap in
    def swap_in(self, device: torch.device, stream: Optional[torch.cuda.Stream]) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    # a swap happened, could be cpu -> device or device -> cpu
    def swapped(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    # data is in device, could be no swapped or swap in is launched
    def expect_device(self) -> bool:
        raise NotImplementedError


Data = TypeVar('Data')


class SwapableTensor(CanSwapObject):
    def __init__(self, t: torch.Tensor) -> None:
        assert not t.requires_grad
        assert t.grad_fn is None
        self.tensor = t
        self._swapped = False
        self._expect_device = True

    def swap_out(self, stream: Optional[torch.cuda.Stream]) -> None:
        if stream is None:
            self.tensor = self.tensor.to(device="cpu", non_blocking=True)
        else:
            with torch.cuda.stream(stream):
                self.tensor = self.tensor.to(device="cpu", non_blocking=True)
        self._swapped = True
        self._expect_device = False

    def swap_in(self, device: torch.device, stream: Optional[torch.cuda.Stream]) -> None:
        if stream is None:
            self.tensor = self.tensor.to(device=device, non_blocking=True)
        else:
            with torch.cuda.stream(stream):
                self.tensor = self.tensor.to(device=device, non_blocking=True)
        self._swapped = True
        self._expect_device = True

    @property
    def swapped(self) -> bool:
        return self._swapped

    @property
    def expect_device(self) -> bool:
        return self._expect_device

    def get_tensor(self) -> torch.Tensor:
        return self.tensor


class SwapableTensors(CanSwapObject):
    def __init__(self, tensors: List[torch.Tensor]) -> None:
        for t in tensors:
            assert not t.requires_grad
            assert t.grad_fn is None
        self.tensors: List[torch.Tensor] = tensors
        self._swapped = False
        self._expect_device = True

    def swap_out(self, stream: Optional[torch.cuda.Stream]) -> None:
        if stream is None:
            self.tensors = [t.to(device="cpu", non_blocking=True) for t in self.tensors]
        else:
            with torch.cuda.stream(stream):
                self.tensors = [t.to(device="cpu", non_blocking=True) for t in self.tensors]
        self._swapped = True
        self._expect_device = False

    def swap_in(self, device: torch.device, stream: Optional[torch.cuda.Stream]) -> None:
        if stream is None:
            self.tensors = [t.to(device=device, non_blocking=True) for t in self.tensors]
        else:
            with torch.cuda.stream(stream):
                self.tensors = [t.to(device=device, non_blocking=True) for t in self.tensors]
        self._swapped = True
        self._expect_device = True

    @property
    def swapped(self) -> bool:
        return self._swapped

    @property
    def expect_device(self) -> bool:
        return self._expect_device

    def get_tensors(self) -> List[torch.Tensor]:
        return self.tensors


class UnSwapableAny(CanSwapObject, Generic[Data]):
    """
    cannot call swap on this object
    """

    def __init__(self, data: Data) -> None:
        self.data = data

    def swap_out(self, stream: Optional[torch.cuda.Stream]) -> None:
        raise NotImplementedError

    def swap_in(self, device: torch.device, stream: Optional[torch.cuda.Stream]) -> None:
        raise NotImplementedError

    @property
    def swapped(self) -> bool:
        return False

    @property
    def expect_device(self) -> bool:
        return True

    def get_data(self) -> Data:
        return self.data


class NoSwapAny(CanSwapObject, Generic[Data]):
    """
    this object no need to swap, e.g. pure cpu object
    """

    def __init__(self, data: Data) -> None:
        self.data = data

    def swap_out(self, stream: Optional[torch.cuda.Stream]) -> None:
        return

    def swap_in(self, device: torch.device, stream: Optional[torch.cuda.Stream]) -> None:
        return

    @property
    def swapped(self) -> bool:
        return False

    @property
    def expect_device(self) -> bool:
        return True

    def get_data(self) -> Data:
        return self.data
