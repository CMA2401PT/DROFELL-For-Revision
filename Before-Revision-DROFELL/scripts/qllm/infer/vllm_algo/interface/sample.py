from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
from .basic_types import SampleLogProb, TokenID
from .basic_types import SeqGroupSelectedSeqIDs, SeqID
from .sequence import SequenceTokenIDsData
from .meta import SeqGroupMetaForSampler
from .sampling_params import SamplingParams, SamplingType


@dataclass
class SamplingMetadata:
    seq_groups_sampling_params: List[Tuple[SeqGroupSelectedSeqIDs, SamplingParams]]
    is_prompts: List[bool]
    seq_data: Dict[SeqID, SequenceTokenIDsData]
    categorized_sample_indices: Dict[SamplingType, torch.Tensor]


class SequenceSampleOutput:
    """The model output associated with a sequence.

    Args:
        parent_seq_id: The ID of the parent sequence (for forking in beam
            search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
    """

    def __init__(
            self,
            parent_seq_id: SeqID,
            output_token: TokenID,
            logprobs: SampleLogProb,
    ) -> None:
        self.parent_seq_id = parent_seq_id
        self.output_token = output_token
        self.logprobs = logprobs

    def __repr__(self) -> str:
        return (f"SequenceOutput(parent_seq_id={self.parent_seq_id}, "
                f"output_token={self.output_token}, "
                f"logprobs={self.logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceSampleOutput):
            raise NotImplementedError()
        return (self.parent_seq_id == other.parent_seq_id
                and self.output_token == other.output_token
                and self.logprobs == other.logprobs)


class Sampler(ABC):
    @abstractmethod
    def sample(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> List[List[SequenceSampleOutput]]:
        raise NotImplementedError

    @staticmethod
    def gen_sampling_meta_data(metas: List[Tuple[SeqGroupSelectedSeqIDs, SeqGroupMetaForSampler]], is_prompt: bool):
        seq_group_sampling_params: List[Tuple[SeqGroupSelectedSeqIDs, SamplingParams]] = []
        seq_data: Dict[SeqID, SequenceTokenIDsData] = {}
        categorized_sample_indices: Dict[SamplingType, List[int]] = {t: [] for t in SamplingType}
        ptr = 0
        for (seq_group_seq_ids, seq_group_sampler_meta) in metas:
            seq_group_sampling_params.append((seq_group_seq_ids, seq_group_sampler_meta.sampling_parameters))
            seq_data.update(seq_group_sampler_meta.seqs_token_ids_data)
            for _ in seq_group_seq_ids:
                categorized_sample_indices[seq_group_sampler_meta.sampling_parameters.sampling_type].append(ptr)
                ptr += 1
        return SamplingMetadata(
            seq_groups_sampling_params=seq_group_sampling_params,
            is_prompts=[is_prompt for _ in range(len(metas))],
            seq_data=seq_data,
            categorized_sample_indices={
                t: torch.tensor(seq_ids, dtype=torch.int)
                for t, seq_ids in categorized_sample_indices.items()
            },
        )
