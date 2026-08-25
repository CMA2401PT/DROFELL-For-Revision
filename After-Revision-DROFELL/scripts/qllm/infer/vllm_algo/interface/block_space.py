from abc import ABC, abstractmethod
from .block import VirtualPhysicalTokenBlock


class VirtualBlockSpaceHolder(ABC):
    @abstractmethod
    def allocate(self) -> VirtualPhysicalTokenBlock:
        raise NotImplementedError

    @abstractmethod
    def free(self, block: VirtualPhysicalTokenBlock) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_total_blocks(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def block_size(self) -> int:
        raise NotImplementedError
