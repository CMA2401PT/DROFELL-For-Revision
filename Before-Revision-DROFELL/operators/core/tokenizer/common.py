from dataclasses import dataclass, field
from typing import List, Optional, cast
from ... import flags


@dataclass
class TokenizerSpecialTokens:
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int


@dataclass
class SequenceLengthData:
    prompt_len: int
    output_count: int

    def clone(self):
        return SequenceLengthData(self.prompt_len, self.output_count)

    def append_token(self) -> None:
        self.output_count += 1

    @property
    def num_activated_blocks(self):
        return (self.num_slots+flags.BLOCK_SIZE-1)//flags.BLOCK_SIZE

    @property
    def num_fulled_blocks(self):
        return self.num_slots//flags.BLOCK_SIZE

    @property
    def num_tokens(self) -> int:
        return self.prompt_len+self.output_count

    @property
    def num_slots(self) -> int:
        return self.num_tokens-1

    @staticmethod
    def get_blocks(num_token: int) -> List[int]:
        required_blocks = (num_token+flags.BLOCK_SIZE-1)//flags.BLOCK_SIZE
        return [i for i in range(required_blocks)]


@dataclass
class SequenceTokensData:
    prompt_token_ids: List[int]
    output_token_ids: List[int]

    def append_token_id(
            self,
            token_id: int,
    ):
        self.output_token_ids.append(token_id)


@dataclass
class StopConfig:
    max_tokens: int
    ignore_eos: bool = False
    stopped_eos_token_ids: int = 2
    stopped_tokens_sequences: List[List[int]] = cast(List[List[int]], field(default_factory=list))
    stop_token_ids: List[int] = cast(List[int], field(default_factory=list))


@dataclass
class SequenceCore:
    stop_config: StopConfig
    prompt_token_ids: List[int] = cast(List[int], field(default_factory=list))
    output_token_ids: List[int] = cast(List[int], field(default_factory=list))
    stopped: bool = False

    @property
    def output_count(self):
        return len(self.output_token_ids)

    def clone(self):
        return SequenceCore(
            stop_config=self.stop_config,
            prompt_token_ids=self.prompt_token_ids.copy(),
            output_token_ids=self.output_token_ids.copy(),
            stopped=self.stopped
        )

    def append_token_id(
            self,
            token_id: int,
    ) -> None:
        assert not self.stopped
        self.output_token_ids.append(token_id)
        for stopped_tokens_sequence in self.stop_config.stopped_tokens_sequences:
            if self.output_token_ids[-len(stopped_tokens_sequence):] == stopped_tokens_sequence:
                self.stopped = True
                return
        if self.output_token_ids[-1] in self.stop_config.stop_token_ids:
            self.stopped = True
            return
        if len(self.output_token_ids) == self.stop_config.max_tokens:
            self.stopped = True
            return
        if (not self.stop_config.ignore_eos)\
                and self.output_token_ids[-1] == self.stop_config.stopped_eos_token_ids:
            self.stopped = True
            return

    @property
    def prompt_len(self):
        return len(self.prompt_token_ids)

    @property
    def length_data(self) -> SequenceLengthData:
        return SequenceLengthData(prompt_len=len(self.prompt_token_ids), output_count=self.output_count)


@dataclass
class Sequence:
    core: SequenceCore
    prompt: Optional[str] = None
    output_text: str = ""
    new_output_text: str = ""
    stop_cond_text: List[str] = cast(List[str], field(default_factory=list))
    prefix_offset: int = 0
    read_offset: int = 0
    prev_decoded_tokens: Optional[List[str]] = None

    def clone(self):
        return Sequence(
            core=self.core.clone(),
            prompt=self.prompt,
            output_text=self.output_text,
            new_output_text=self.new_output_text,
            stop_cond_text=self.stop_cond_text.copy(),
            prefix_offset=self.prefix_offset,
            read_offset=self.read_offset,
            prev_decoded_tokens=self.prev_decoded_tokens.copy() if self.prev_decoded_tokens is not None else None
        )
