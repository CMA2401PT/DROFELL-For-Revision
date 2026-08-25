from abc import ABC, abstractmethod
from typing import Tuple
from .sequence import *
from .block import BlockTable


class AllocStatus(enum.Enum):
    """Result for BlockSpaceManager.can_allocate

    1. Ok: seq_group can be allocated now.
    2. Later: seq_group cannot be allocated.
      The capacity of allocator is larger than seq_group required.
    3. Never: seq_group can never be allocated.
      The seq_group is too large to allocated in GPU.
    """
    OK = enum.auto()
    LATER = enum.auto()
    NEVER = enum.auto()


class VirtualBlockSpaceIO(ABC):
    """block space manager contains info of which physic block is in use"""
    @property
    @abstractmethod
    def block_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def can_allocate_ptuning(self, blocks_count: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def allocate_ptuning(self, blocks_count: int) -> BlockTable:
        raise NotImplementedError

    @abstractmethod
    def free_ptuning(self, ptuning_block_table: BlockTable) -> None:
        raise NotImplementedError

    @abstractmethod
    def can_allocate_prefill(self, seq_group: SequenceGroup) -> AllocStatus:
        # seq_group has only 1 seq with only prompt tokens, return if can allocate space for prompt tokens
        raise NotImplementedError

    @abstractmethod
    def allocate_prefill(self, seq_group: SequenceGroup) -> None:
        # seq_group has only 1 seq with only prompt tokens
        # update maintained block tables
        raise NotImplementedError

    @abstractmethod
    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # seq group has multiple seq, check if can allocate at least one block for each seq
        raise NotImplementedError

    @abstractmethod
    def append_slot(self, seq: Sequence) -> Optional[Tuple[VirtualBlockNumber, VirtualBlockNumber]]:
        # allocate a slot for seq, if last block of this seq is shared with other seq, do copy on write
        # copy on write:
        # when subsequent opeartion will write the last block,
        # the virtual space manager de-ref the seq to last block and "allocate" a new block
        # then the subsequent operation should copy data in old block to new block first
        raise NotImplementedError

    @abstractmethod
    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # fork does not allocate a new physical block.
        # it make clild seq has save block table as parent seq
        # child seq must be new
        raise NotImplementedError

    @abstractmethod
    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        # can move the cpu blocks in seq group to gpu
        raise NotImplementedError

    @abstractmethod
    def swap_in(self, seq_group: SequenceGroup) -> Dict[VirtualBlockNumber, VirtualBlockNumber]:
        # move the cpu blocks in seq group from cpu to gpu
        # return Dict[CPU block number,GPU Block number]
        raise NotImplementedError

    @abstractmethod
    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        # can move the gpu blocks in seq group to cpu
        raise NotImplementedError

    @abstractmethod
    def swap_out(self, seq_group: SequenceGroup) -> Dict[VirtualBlockNumber, VirtualBlockNumber]:
        # move the cpu blocks in seq group from gpu to cpu
        # return Dict[GPU Block number,CPU block number]
        raise NotImplementedError

    @abstractmethod
    def free(self, seq: Sequence) -> None:
        # free all blocks in matained block tables, but not actually free it
        raise NotImplementedError

    @abstractmethod
    def get_block_table_by_id(self, seq_id: SeqID) -> List[VirtualBlockNumber]:
        raise NotImplementedError
