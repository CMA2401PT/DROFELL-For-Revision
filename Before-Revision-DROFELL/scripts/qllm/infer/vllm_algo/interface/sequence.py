"""Sequence and its related classes."""
import copy
from typing import Optional, Union
from .sampling_params import SamplingParams
from .basic_types import *
from .block import LogicalTokenBlock


class SequenceStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAPPED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(state: "SequenceStatus") -> bool:
        return state in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED,
            SequenceStatus.FINISHED_IGNORED,
        ]

    @staticmethod
    def get_finished_reason(state: "SequenceStatus") -> Union[str, None]:
        if state == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif state == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif state == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif state == SequenceStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class SequenceTokenIDsData:
    """Data associated with a sequence.


    Args:
        prompt_token_ids: The token IDs of the prompt.

    Attributes:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output.
        cumulative_logprob: The cumulative log probability of the output.
    """

    def __init__(
            self,
            prompt_token_ids: List[TokenID],
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids: List[TokenID] = []
        self.cumulative_logprob = 0.0

    def append_token_id(self, token_id: TokenID, logprob: LogProb) -> None:
        self.output_token_ids.append(token_id)
        self.cumulative_logprob += logprob

    def get_len(self) -> int:
        return len(self.output_token_ids) + len(self.prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_token_ids(self) -> List[TokenID]:
        return self.prompt_token_ids + self.output_token_ids

    def get_last_token_id(self) -> TokenID:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob})")


class Sequence:
    """Stores the data, state, and block information of a sequence.

    Args:
        seq_id: The ID of the sequence.
        prompt: The prompt of the sequence.
        prompt_token_ids: The token IDs of the prompt.
        block_size: The block size of the sequence. Should be the same as the
            block size used by the block manager and cache engine.
    """

    def __init__(
            self,
            seq_id: SeqID,
            prompt: Text,
            prompt_token_ids: List[TokenID],
            block_size: int,
            # ptuning_id:str,
    ) -> None:
        self.seq_id = seq_id
        self.prompt = prompt
        self.block_size = block_size
        # self.ptuning_id=ptuning_id

        self.token_ids_data = SequenceTokenIDsData(prompt_token_ids)
        self.output_logprobs: SeqSampleLogprobsInSeqGroup = []
        self.output_text = ""

        self.logical_token_blocks: List[LogicalTokenBlock] = []
        # Initialize the logical token blocks with the prompt token ids.
        self._append_tokens_to_blocks(prompt_token_ids)
        self.state = SequenceStatus.WAITING

        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens
        self.decoded_tokens: Optional[List[TokenDecoded]] = None

    def _append_logical_block(self) -> None:
        block = LogicalTokenBlock(
            block_number=len(self.logical_token_blocks),
            block_size=self.block_size,
        )
        self.logical_token_blocks.append(block)

    def _append_tokens_to_blocks(self, token_ids: List[TokenID]) -> None:
        cursor = 0
        while cursor < len(token_ids):
            if not self.logical_token_blocks:
                self._append_logical_block()

            last_block = self.logical_token_blocks[-1]
            if last_block.is_full():
                self._append_logical_block()
                last_block = self.logical_token_blocks[-1]

            num_empty_slots = last_block.get_num_empty_slots()
            last_block.append_tokens(token_ids[cursor:cursor + num_empty_slots])
            cursor += num_empty_slots

    def append_token_id(
            self,
            token_id: TokenID,
            logprobs: Dict[TokenID, LogProb],
    ) -> None:
        assert token_id in logprobs
        self._append_tokens_to_blocks([token_id])
        self.output_logprobs.append(logprobs)
        self.token_ids_data.append_token_id(token_id, logprobs[token_id])

    def get_len(self) -> int:
        return self.token_ids_data.get_len()

    def get_prompt_len(self) -> int:
        return self.token_ids_data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.token_ids_data.get_output_len()

    def get_token_ids(self) -> List[TokenID]:
        return self.token_ids_data.get_token_ids()

    def get_last_token_id(self) -> TokenID:
        return self.token_ids_data.get_last_token_id()

    def get_output_token_ids(self) -> List[TokenID]:
        return self.token_ids_data.output_token_ids

    def get_cumulative_logprob(self) -> LogProb:
        return self.token_ids_data.cumulative_logprob

    def get_beam_search_score(self,
                              length_penalty: float = 0.0,
                              seq_len: Optional[int] = None,
                              eos_token_id: Optional[TokenID] = None) -> float:
        """Calculate the beam search score with length penalty.

        Adapted from

        https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
        """
        if seq_len is None:
            seq_len = self.get_len()
            # NOTE: HF implementation does not count the EOS token
            # towards the length, we align with that here for testing.
            if (eos_token_id is not None
                    and self.get_last_token_id() == eos_token_id):
                seq_len -= 1
        return self.get_cumulative_logprob() / (seq_len ** length_penalty)

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.state)

    def fork(self, new_seq_id: SeqID) -> "Sequence":
        new_seq = copy.deepcopy(self)
        new_seq.seq_id = new_seq_id
        return new_seq

    def __repr__(self) -> str:
        return (f"Sequence(seq_id={self.seq_id}, "
                f"state={self.state.name}, "
                f"num_blocks={len(self.logical_token_blocks)})")


class SequenceGroup:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
        seqs: The list of sequences.
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
    """

    def __init__(
            self,
            request_id: str,
            seqs: List[Sequence],
            sampling_params: SamplingParams,
            arrival_time: Time,
            ptuning_id: Optional[str],
    ) -> None:
        self.request_id = request_id
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time
        self.ptuning_id = ptuning_id

    @property
    def prompt(self) -> Text:
        # All sequences in the group should have the same prompt.
        # We use the prompt of an arbitrary sequence.
        return next(iter(self.seqs_dict.values())).prompt

    @property
    def prompt_token_ids(self) -> List[TokenID]:
        # All sequences in the group should have the same prompt.
        # We use the prompt of an arbitrary sequence.
        return next(iter(self.seqs_dict.values())).token_ids_data.prompt_token_ids

    def get_max_num_running_seqs(self) -> int:
        # """The maximum number of sequences running in parallel in the remaining
        # lifetime of the request."""
        # if self.sampling_params.use_beam_search:
        #     # For beam search, maximally there will always be `best_of` beam
        #     # candidates running in the future.
        #     return self.sampling_params.best_of
        # else:
        #     if self.sampling_params.best_of > self.num_seqs():
        #         # At prompt stage, the sequence group is not yet filled up
        #         # and only have one sequence running. However, in the
        #         # generation stage, we will have `best_of` sequences running.
        #         return self.sampling_params.best_of
        #     # At sampling stages, return the number of actual sequences
        #     # that are not finished yet.
        #     return self.num_unfinished_seqs()
        return 1

    def get_seqs(
            self,
            state: Optional[SequenceStatus] = None,
    ) -> List[Sequence]:
        if state is None:
            return list(self.seqs_dict.values())
        else:
            return [
                seq for seq in self.seqs_dict.values() if seq.state == state
            ]

    def get_unfinished_seqs(self) -> List[Sequence]:
        return [
            seq for seq in self.seqs_dict.values() if not seq.is_finished()
        ]

    def get_finished_seqs(self) -> List[Sequence]:
        return [seq for seq in self.seqs_dict.values() if seq.is_finished()]

    def num_seqs(self, state: Optional[SequenceStatus] = None) -> int:
        return len(self.get_seqs(state))

    def num_unfinished_seqs(self) -> int:
        return len(self.get_unfinished_seqs())

    def num_finished_seqs(self) -> int:
        return len(self.get_finished_seqs())

    def find(self, seq_id: str) -> Sequence:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        return self.seqs_dict[seq_id]

    def add(self, seq: Sequence) -> None:
        if seq.seq_id in self.seqs_dict:
            raise ValueError(f"Sequence {seq.seq_id} already exists.")
        self.seqs_dict[seq.seq_id] = seq

    def remove(self, seq_id: str) -> None:
        if seq_id not in self.seqs_dict:
            raise ValueError(f"Sequence {seq_id} not found.")
        del self.seqs_dict[seq_id]

    def is_finished(self) -> bool:
        return all(seq.is_finished() for seq in self.get_seqs())

    def __repr__(self) -> str:
        return (f"SequenceGroup(request_id={self.request_id}, "
                f"sampling_params={self.sampling_params}, "
                f"num_seqs={len(self.seqs_dict)})")


# class SequenceGroupMetadata:
#     """Metadata for a sequence group. Used to create `InputMetadata`.
#
#
#     Args:
#         request_id: The ID of the request.
#         is_prompt: Whether the request is at prompt stage.
#         seq_data: The sequence data. (Seq id -> sequence data)
#         sampling_params: The sampling parameters used to generate the outputs.
#         block_tables: The block tables. (Seq id -> list of physical block
#             numbers)
#     """
#
#     def __init__(
#             self,
#             request_id: str,
#             is_prompt: bool,
#             seq_data: Dict[SeqID, SequenceTokenIDsData],
#             sampling_params: SamplingParams,
#             block_tables: Dict[int, List[int]],
#     ) -> None:
#         self.request_id = request_id
#         self.is_prompt = is_prompt
#         self.seq_data = seq_data
#         self.sampling_params = sampling_params
#         self.block_tables = block_tables
#


# class SequenceGroupOutput:
#     """The model output associated with a sequence group."""

#     def __init__(
#         self,
#         samples: List[SequenceOutput],
#     ) -> None:
#         self.samples = samples

#     def __repr__(self) -> str:
#         return (f"SequenceGroupOutput(samples={self.samples}")

#     def __eq__(self, other: object) -> bool:
#         if not isinstance(other, SequenceGroupOutput):
#             raise NotImplementedError()
#         return (self.samples == other.samples)


# For each sequence group, we generate a list of SequenceOutput object,
# each of which contains one possible candidate for the next token.
# SamplerOutput = List[List[SequenceOutput]]
