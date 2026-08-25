"""Sampling parameters for text generation."""
from enum import IntEnum
from functools import cached_property
from typing import Optional

_SAMPLING_EPS = 1e-5


class SamplingType(IntEnum):
    GREEDY = 0
    RANDOM = 1


class SamplingParams:
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    In addition, we support beam search, which is not supported by OpenAI.

    Args:
        n: Number of output sequences to return for the given prompt.
        best_of: Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. This is treated as
            the beam width when `use_beam_search` is True. By default, `best_of`
            is set to `n`.
        presence_penalty: Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        repetition_penalty: Float that penalizes new tokens based on whether
            they appear in the prompt and the generated text so far. Values > 1
            encourage the model to use new tokens, while values < 1 encourage
            the model to repeat tokens.
        temperature: Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p: Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k: Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        min_p: Float that represents the minimum probability for a token to be
            considered, relative to the probability of the most likely token.
            Must be in [0, 1]. Set to 0 to disable this.
        use_beam_search: Whether to use beam search instead of sampling.
        length_penalty: Float that penalizes sequences based on their length.
            Used in beam search.
        logprobs: Number of log probabilities to return per output token.
            Note that the implementation follows the OpenAI API: The return
            result includes the log probabilities on the `logprobs` most likely
            tokens, as well the chosen tokens. The API will always return the
            log probability of the sampled token, so there  may be up to
            `logprobs+1` elements in the response.
        skip_special_tokens: Whether to skip special tokens in the output.
        spaces_between_special_tokens: Whether to add spaces between special
            tokens in the output.  Defaults to True.
        logits_processors: List of functions that modify logits based on
            previously generated tokens.
    """

    def __init__(
            self,
            presence_penalty: float = 0.0,
            frequency_penalty: float = 0.0,
            repetition_penalty: float = 1.0,
            temperature: float = 1.0,
            top_p: float = 1.0,
            top_k: int = -1,
            min_p: float = 0.0,
            length_penalty: float = 1.0,
            logprobs: Optional[int] = None,
            skip_special_tokens: bool = True,
            spaces_between_special_tokens: bool = True,
    ) -> None:
        self.presence_penalty: float = presence_penalty
        self.frequency_penalty: float = frequency_penalty
        self.repetition_penalty: float = repetition_penalty
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.top_k: int = top_k
        self.min_p: float = min_p
        self.length_penalty: float = length_penalty
        self.logprobs: Optional[int] = logprobs
        self.skip_special_tokens: bool = skip_special_tokens
        self.spaces_between_special_tokens: bool = spaces_between_special_tokens
        self._verify_args()
        if self.temperature < _SAMPLING_EPS:
            # Zero temperature means greedy sampling.
            self.top_p = 1.0
            self.top_k = -1
            self.min_p = 0.0

    def _verify_args(self) -> None:
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2, 2], got "
                             f"{self.presence_penalty}.")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2, 2], got "
                             f"{self.frequency_penalty}.")
        if not 0.0 < self.repetition_penalty <= 2.0:
            raise ValueError("repetition_penalty must be in (0, 2], got "
                             f"{self.repetition_penalty}.")
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disable), or at least 1, "
                             f"got {self.top_k}.")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1], got "
                             f"{self.min_p}.")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(
                f"logprobs must be non-negative, got {self.logprobs}.")

    @cached_property
    def sampling_type(self) -> SamplingType:
        if self.temperature < _SAMPLING_EPS:
            return SamplingType.GREEDY
        return SamplingType.RANDOM

    def __repr__(self) -> str:
        return (f"presence_penalty={self.presence_penalty}, "
                f"frequency_penalty={self.frequency_penalty}, "
                f"repetition_penalty={self.repetition_penalty}, "
                f"temperature={self.temperature}, "
                f"top_p={self.top_p}, "
                f"top_k={self.top_k}, "
                f"min_p={self.min_p}, "
                f"length_penalty={self.length_penalty}, "
                f"logprobs={self.logprobs}, "
                f"skip_special_tokens={self.skip_special_tokens}, "
                "spaces_between_special_tokens="
                f"{self.spaces_between_special_tokens})")
