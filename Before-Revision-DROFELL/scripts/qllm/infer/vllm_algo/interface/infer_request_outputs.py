from typing import List, Optional, Callable
from .basic_types import LogProb, Text, TokenID
from .sequence import SeqSampleLogprobsInSeqGroup, SequenceGroup, SequenceStatus, Sequence


class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        finish_reason: The reason why the sequence is finished.
    """

    def __init__(
            self,
            index: int,
            text: str,
            token_ids: List[int],
            cumulative_logprob: LogProb,
            logprobs: Optional[SeqSampleLogprobsInSeqGroup],
            finish_reason: Optional[str] = None,
    ) -> None:
        self.index: int = index
        self.text: str = text
        self.token_ids: List[int] = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs:  Optional[SeqSampleLogprobsInSeqGroup] = logprobs
        self.finish_reason: Optional[str] = finish_reason

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (f"text={self.text!r}, "
                f"finish_reason={self.finish_reason})")


class RequestOutput:
    """The output data of a request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
    """

    def __init__(
            self,
            request_id: str,
            prompt: str,
            prompt_token_ids: List[int],
            outputs: List[CompletionOutput],
            finished: bool,
    ) -> None:
        self.request_id: str = request_id
        self.prompt: Text = prompt
        self.prompt_token_ids: List[TokenID] = prompt_token_ids
        # multiple seqs
        self.outputs: List[CompletionOutput] = outputs
        self.finished: bool = finished

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"prompt={self.prompt!r}, "
                f"outputs={self.outputs}, "
                f"finished={self.finished})")

    @classmethod
    def create_from_seq_group(cls, seq_group: SequenceGroup):
        # Get the top-n sequences.
        # n: int = seq_group.sampling_params.n
        n = 1
        seqs: List[Sequence] = seq_group.get_seqs()
        # if seq_group.sampling_params.use_beam_search:
        #     sorting_key: Callable[[Sequence], float] = lambda seq: seq.get_beam_search_score(
        #         length_penalty=seq_group.sampling_params.length_penalty)
        # else:
        sorting_key: Callable[[Sequence], float] = lambda seq: seq.get_cumulative_logprob()
        sorted_seqs: list[Sequence] = sorted(seqs, key=sorting_key, reverse=True)
        top_n_seqs: list[Sequence] = sorted_seqs[:n]

        # Create the outputs.
        outputs: List[CompletionOutput] = []
        for seq in top_n_seqs:
            logprobs = seq.output_logprobs
            if seq_group.sampling_params.logprobs is None:
                # NOTE: We need to take care of this case because the sequence
                # always has the logprobs of the sampled tokens even if the
                # logprobs are not requested.
                logprobs = None
            finshed_reason: Optional[str] = SequenceStatus.get_finished_reason(
                state=seq.state)
            output = CompletionOutput(index=seqs.index(seq), text=seq.output_text,
                                      token_ids=seq.get_output_token_ids(),
                                      cumulative_logprob=seq.get_cumulative_logprob(), logprobs=logprobs,
                                      finish_reason=finshed_reason)
            outputs.append(output)

        # Every sequence in the sequence group should have the same prompt.
        prompt = seq_group.prompt
        prompt_token_ids: List[TokenID] = seq_group.prompt_token_ids
        finished: bool = seq_group.is_finished()
        return cls(request_id=seq_group.request_id, prompt=prompt, prompt_token_ids=prompt_token_ids, outputs=outputs, finished=finished)
