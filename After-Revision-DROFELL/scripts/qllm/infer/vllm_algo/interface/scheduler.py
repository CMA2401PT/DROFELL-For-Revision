from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from .sequence import SequenceGroup
from .block_space_io import VirtualBlockSpaceIO
from .meta import MemoryOperations


class Scheduler(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def schedule_prefill(self, block_space_io: VirtualBlockSpaceIO) -> List[SequenceGroup]:
        raise NotImplementedError

    @abstractmethod
    def schedule_generating(self, block_space_io: VirtualBlockSpaceIO) -> \
            Tuple[List[SequenceGroup], Optional[MemoryOperations]]:
        raise NotImplementedError

    @abstractmethod
    def free_finished_seq_groups(self) -> List[SequenceGroup]:
        raise NotImplementedError

    @property
    @abstractmethod
    def max_model_len(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def prompt_limit(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def waiting(self) -> List[SequenceGroup]:
        raise NotImplementedError
