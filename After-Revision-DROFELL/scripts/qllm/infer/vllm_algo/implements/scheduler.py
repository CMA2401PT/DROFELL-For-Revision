import enum
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from ..interface.basic_types import VirtualBlockNumber
from ..interface.sequence import SequenceGroup, SequenceStatus
from .policy import PolicyFactory
from ..interface.block_space_io import AllocStatus, VirtualBlockSpaceIO
from ..interface.meta import MemoryOperations
from ..interface.scheduler import Scheduler


@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 0
    max_num_seqs: int = 0
    max_model_len: int = 0
    max_paddings: int = 0

    def __post_init__(self):
        if self.max_num_batched_tokens == 0:
            self.max_num_batched_tokens = max(self.max_model_len, 2048)
        assert self.max_num_batched_tokens >= self.max_model_len
        assert self.max_num_batched_tokens >= self.max_num_seqs


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


class VllmScheduler(Scheduler):

    def __init__(self, scheduler_config: SchedulerConfig) -> None:
        super().__init__()
        self.scheduler_config = scheduler_config
        self._prompt_limit = min(
            self.scheduler_config.max_model_len, self.scheduler_config.max_num_batched_tokens)

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Sequence groups in the WAITING state.
        self._waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []

    def reset(self):
        self._waiting = []
        self.running = []
        self.swapped = []

    @property
    def max_model_len(self) -> int:
        return self.scheduler_config.max_model_len

    @property
    def waiting(self) -> List[SequenceGroup]:
        return self._waiting

    @property
    def prompt_limit(self) -> int:
        return self._prompt_limit

    def schedule_prefill(self, block_space_io: VirtualBlockSpaceIO) -> List[SequenceGroup]:
        # Join waiting sequences if possible.
        if not self.swapped:
            prefill_sequences: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs() for seq_group in self.running)
            seq_lens: List[int] = []

            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self._waiting:
                seq_group = self._waiting[0]
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                # If the sequence group cannot be allocated, stop.
                can_allocate = block_space_io.can_allocate_prefill(seq_group)
                if can_allocate == AllocStatus.LATER:
                    break
                assert can_allocate != AllocStatus.NEVER
                # If the number of batched tokens exceeds the limit, stop.
                new_seq_lens: list[int] = seq_lens + [num_prompt_tokens]
                num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
                if (num_batched_tokens > self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                num_paddings = num_batched_tokens - sum(new_seq_lens)
                if num_paddings > self.scheduler_config.max_paddings:
                    break
                seq_lens = new_seq_lens

                seq_group = self._waiting.pop(0)
                block_space_io.allocate_prefill(seq_group)
                for seq in seq_group.get_seqs():
                    seq.state = SequenceStatus.RUNNING
                self.running.append(seq_group)
                num_curr_seqs += num_new_seqs
                prefill_sequences.append(seq_group)

            return prefill_sequences
        return []

    def schedule_generating(self, block_space_io: VirtualBlockSpaceIO) -> \
            Tuple[List[SequenceGroup], Optional[MemoryOperations]]:
        # Fix the current time.
        now = time.monotonic()

        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.pop(0)
            while not block_space_io.can_append_slot(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group: SequenceGroup = self.running.pop(-1)
                    self._swap_out(seq_group=victim_seq_group, blocks_to_swap_out=blocks_to_swap_out, block_space_io=block_space_io)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._swap_out(seq_group=seq_group, blocks_to_swap_out=blocks_to_swap_out, block_space_io=block_space_io)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq_group=seq_group, blocks_to_copy=blocks_to_copy, block_space_io=block_space_io)
                running.append(seq_group)
        self.running = running

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now=now, seq_groups=self.swapped)
        if not preempted:
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()for seq_group in self.running)
            while self.swapped:
                seq_group: SequenceGroup = self.swapped[0]
                # If the sequence group cannot be swapped in, stop.
                if not block_space_io.can_swap_in(seq_group=seq_group):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs:
                    break

                seq_group = self.swapped.pop(0)
                self._swap_in(seq_group=seq_group, blocks_to_swap_in=blocks_to_swap_in, block_space_io=block_space_io)
                self._append_slot(seq_group=seq_group, blocks_to_copy=blocks_to_copy, block_space_io=block_space_io)
                num_curr_seqs += num_new_seqs
                self.running.append(seq_group)

        memory_operations = MemoryOperations(
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )
        if memory_operations.is_empty():
            memory_operations = None

        return self.running, memory_operations

    def free_finished_seq_groups(self) -> List[SequenceGroup]:
        new_running: List[SequenceGroup] = []
        finished_seq_groups: List[SequenceGroup] = []
        for seq_group in self.running:
            if not seq_group.is_finished():
                new_running.append(seq_group)
            else:
                finished_seq_groups.append(seq_group)
        self.running = new_running
        return finished_seq_groups

    def _append_slot(
            self,
            seq_group: SequenceGroup,
            blocks_to_copy: Dict[int, List[int]],
            block_space_io: VirtualBlockSpaceIO,
    ) -> None:
        for seq in seq_group.get_seqs(state=SequenceStatus.RUNNING):
            ret: Optional[Tuple[VirtualBlockNumber, VirtualBlockNumber]] = block_space_io.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _swap_in(
            self,
            seq_group: SequenceGroup,
            blocks_to_swap_in: Dict[int, int],
            block_space_io: VirtualBlockSpaceIO,
    ) -> None:
        mapping: Dict[VirtualBlockNumber, VirtualBlockNumber] = block_space_io.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(state=SequenceStatus.SWAPPED):
            seq.state = SequenceStatus.RUNNING

    def _swap_out(
            self,
            seq_group: SequenceGroup,
            blocks_to_swap_out: Dict[int, int],
            block_space_io: VirtualBlockSpaceIO,
    ) -> None:
        if not block_space_io.can_swap_out(seq_group=seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping: Dict[VirtualBlockNumber, VirtualBlockNumber] = block_space_io.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(state=SequenceStatus.RUNNING):
            seq.state = SequenceStatus.SWAPPED
        self.swapped.append(seq_group)
