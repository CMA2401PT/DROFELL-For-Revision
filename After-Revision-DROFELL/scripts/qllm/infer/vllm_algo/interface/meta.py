from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .sampling_params import SamplingParams
from .ptuning import PtuningBlocks
from .sequence import SequenceGroup, SequenceTokenIDsData
from .basic_types import SeqID, VirtualBlockNumber, SeqGroupSelectedSeqIDs


@dataclass
class SeqPrefillMetaForExecuter:
    tokens: List[int]
    positions: List[int]
    slots: List[int]
    ptuning: Optional[PtuningBlocks]


@dataclass
class SeqGroupMetaForSampler:
    seqs_token_ids_data: Dict[SeqID, SequenceTokenIDsData]
    sampling_parameters: SamplingParams


@dataclass
class SeqGroupPrefillMeta:
    seq_group: SequenceGroup
    # in prefill stage, a seq group contains only on sequence
    excuter_meta: SeqPrefillMetaForExecuter
    sampler_meta: SeqGroupMetaForSampler


@dataclass
class SeqGeneratingMetaForExecutor:
    token: int = 0
    position: int = 0
    slot: int = 0
    context_len: int = 0
    block_table: List[VirtualBlockNumber] = field(default_factory=list)


@dataclass
class GeneratingSeqGroupMeta:
    seq_group: SequenceGroup
    executer_meta: List[SeqGeneratingMetaForExecutor]
    sampler_meta: SeqGroupMetaForSampler


@dataclass
class MemoryOperations:
    blocks_to_swap_in: Dict[VirtualBlockNumber, VirtualBlockNumber]
    blocks_to_swap_out: Dict[VirtualBlockNumber, VirtualBlockNumber]
    blocks_to_copy: Dict[VirtualBlockNumber, List[VirtualBlockNumber]]

    def __post_init__(self):
        assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return not self.blocks_to_swap_in and not self.blocks_to_swap_out and not self.blocks_to_copy


@dataclass
class GeneratingMeta:
    memory_operations: Optional[MemoryOperations] = None
    seq_group_metas: List[Tuple[SeqGroupSelectedSeqIDs, GeneratingSeqGroupMeta]] =\
        field(default_factory=list)
