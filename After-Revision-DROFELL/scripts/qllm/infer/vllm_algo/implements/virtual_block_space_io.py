from typing import Dict, List, Optional, Set, Tuple
from ..interface.basic_types import VirtualBlockNumber, VirtualDevice
from ..interface.basic_types import SeqID
from ..interface.sequence import Sequence, SequenceGroup, SequenceStatus
from ..interface.block import BlockTable, LogicalTokenBlock, VirtualPhysicalTokenBlock
from ..interface.block_space import VirtualBlockSpaceHolder
from ..interface.block_space_io import AllocStatus, VirtualBlockSpaceIO


class VllmLikeVirtualBlockSpaceIO(VirtualBlockSpaceIO):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(self, gpu_space_holder: VirtualBlockSpaceHolder, cpu_space_holder: VirtualBlockSpaceHolder, watermark: float = 0.01) -> None:
        super().__init__()
        assert gpu_space_holder.block_size == cpu_space_holder.block_size
        test_block = gpu_space_holder.allocate()
        assert test_block.device == VirtualDevice.GPU
        gpu_space_holder.free(test_block)
        test_block = cpu_space_holder.allocate()
        assert test_block.device == VirtualDevice.CPU
        cpu_space_holder.free(test_block)
        self._block_size: int = gpu_space_holder.block_size

        self.watermark: float = watermark
        assert watermark >= 0.0
        self.gpu_space_holder = gpu_space_holder
        self.cpu_space_holder = cpu_space_holder
        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[SeqID, BlockTable] = {}

    def reset(self):
        self.block_tables = {}
        if hasattr(self.gpu_space_holder, "reset"):
            getattr(self.gpu_space_holder, "reset")()
        if hasattr(self.cpu_space_holder, "reset"):
            getattr(self.cpu_space_holder, "reset")()

    @property
    def block_size(self) -> int:
        return self._block_size

    def can_allocate_prefill(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq: Sequence = seq_group.get_seqs()[0]
        num_required_blocks: int = len(seq.logical_token_blocks)

        total_gpu_blocks = self.gpu_space_holder.get_total_blocks()
        water_mark_blocks = total_gpu_blocks * self.watermark
        num_free_gpu_blocks: int = self.gpu_space_holder.get_num_free_blocks()

        # Use watermark to avoid frequent cache eviction.
        if (total_gpu_blocks - num_required_blocks) < water_mark_blocks:
            return AllocStatus.NEVER
        if (num_free_gpu_blocks - num_required_blocks) >= water_mark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def allocate_prefill(self, seq_group: SequenceGroup) -> None:
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seqs: List[Sequence] = seq_group.get_seqs()
        assert len(seqs) == 1
        seq: Sequence = seqs[0]

        # Allocate new physical token blocks that will store the prompt tokens.
        block_table: BlockTable = []
        for _ in range(len(seq.logical_token_blocks)):
            block: VirtualPhysicalTokenBlock = self.gpu_space_holder.allocate()
            # Set the reference counts of the token blocks.
            block.ref_count = seq_group.num_seqs()
            block_table.append(block)

        # Assign the block table for each sequence.
        for seq in seq_group.get_seqs():
            self.block_tables[seq.seq_id] = block_table.copy()

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_free_gpu_blocks: int = self.gpu_space_holder.get_num_free_blocks()
        # print(num_free_gpu_blocks)
        num_seqs: int = seq_group.num_seqs(state=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def append_slot(self, seq: Sequence) -> Optional[Tuple[VirtualBlockNumber, VirtualBlockNumber]]:
        """Allocate a physical slot for a new token."""
        logical_blocks: List[LogicalTokenBlock] = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]

        if len(block_table) < len(logical_blocks):
            # The sequence has a new logical block.
            # Allocate a new physical block.
            block: VirtualPhysicalTokenBlock = self.gpu_space_holder.allocate()
            block_table.append(block)
            return None

        # We want to append the token to the last physical block.
        last_block: VirtualPhysicalTokenBlock = block_table[-1]
        assert last_block.device == VirtualDevice.GPU
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            return None
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            new_block: VirtualPhysicalTokenBlock = self.gpu_space_holder.allocate()
            block_table[-1] = new_block
            self.gpu_space_holder.free(block=last_block)
            return last_block.block_number, new_block.block_number

    def can_allocate_ptuning(self, blocks_count: int) -> bool:
        num_free_gpu_blocks: int = self.gpu_space_holder.get_num_free_blocks()
        return num_free_gpu_blocks >= blocks_count

    def allocate_ptuning(self, blocks_count: int) -> BlockTable:
        num_free_gpu_blocks: int = self.gpu_space_holder.get_num_free_blocks()
        assert num_free_gpu_blocks >= blocks_count
        block_table: BlockTable = []
        for _ in range(blocks_count):
            block: VirtualPhysicalTokenBlock = self.gpu_space_holder.allocate()
            block.ref_count = 1
            block_table.append(block)
        return block_table

    def free_ptuning(self, ptuning_block_table: BlockTable):
        self._free_block_table(block_table=ptuning_block_table)

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1

    def _get_physical_blocks(
            self, seq_group: SequenceGroup) -> List[VirtualPhysicalTokenBlock]:
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[VirtualPhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            blocks.update(self.block_tables[seq.seq_id])
        return list(blocks)

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        blocks: List[VirtualPhysicalTokenBlock] = self._get_physical_blocks(
            seq_group=seq_group)
        num_swapped_seqs: int = seq_group.num_seqs(
            state=SequenceStatus.SWAPPED)
        num_free_blocks: int = self.gpu_space_holder.get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks: int = len(blocks) + num_swapped_seqs
        watermark_blocks = self.gpu_space_holder.get_total_blocks() * self.watermark
        return num_free_blocks - num_required_blocks >= watermark_blocks

    def swap_in(self, seq_group: SequenceGroup) -> Dict[VirtualBlockNumber, VirtualBlockNumber]:
        mapping: Dict[VirtualPhysicalTokenBlock, VirtualPhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(state=SequenceStatus.SWAPPED):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for cpu_block in block_table:
                if cpu_block in mapping:
                    gpu_block: VirtualPhysicalTokenBlock = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self.gpu_space_holder.allocate()
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                self.cpu_space_holder.free(cpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping: dict[VirtualBlockNumber, VirtualBlockNumber] = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks: List[VirtualPhysicalTokenBlock] = self._get_physical_blocks(
            seq_group=seq_group)
        return len(blocks) <= self.cpu_space_holder.get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> Dict[VirtualBlockNumber, VirtualBlockNumber]:
        # GPU block -> CPU block.
        mapping: Dict[VirtualPhysicalTokenBlock, VirtualPhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(state=SequenceStatus.RUNNING):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for gpu_block in block_table:
                if gpu_block in mapping:
                    cpu_block: VirtualPhysicalTokenBlock = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_space_holder.allocate()
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                self.gpu_space_holder.free(gpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping: dict[VirtualBlockNumber, VirtualBlockNumber] = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping

    def _free_block_table(self, block_table: BlockTable) -> None:
        for block in set(block_table):
            if block.device == VirtualDevice.GPU:
                self.gpu_space_holder.free(block=block)
            else:
                self.cpu_space_holder.free(block=block)

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table=block_table)
        del self.block_tables[seq.seq_id]

    def get_block_table_by_id(self, seq_id: SeqID) -> List[VirtualBlockNumber]:
        block_table = self.block_tables[seq_id]
        return [block.block_number for block in block_table]
