from abc import ABC, abstractmethod


class CanSwapObjectMemory(ABC):
    @abstractmethod
    def swap_out(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def swap_in(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    # a swap happened, could be cpu -> device or device -> cpu
    def swapped(self) -> bool:
        raise NotImplementedError
