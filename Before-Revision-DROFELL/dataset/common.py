from dataclasses import dataclass
from typing import List
from transformers.tokenization_utils import PreTrainedTokenizer  # type:ignore
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast  # type:ignore
from transformers import AutoTokenizer  # type:ignore
from typing import Callable, List, Optional, Union


@dataclass
class Record:
    history: List[str]
    query: str
    answer: str


@dataclass
class ProcessedRecord:
    input_ids: List[int]
    output_ids: List[int]


IGNORE_PAD_TOKEN_ID_IN_LABEL = -100


class BuildQueryAndTranslate:
    def __init__(self, build_prompt_str: str, tokenizer_path: str, tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None) -> None:
        self.build_prompt_str = build_prompt_str
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    def __call__(self, record: Record) -> ProcessedRecord:
        query = self.build_prompt_str.format(record.query)
        input_ids: List[int] = self.tokenizer.encode(query)
        output_ids: List[int] = self.tokenizer.encode(record.answer)
        return ProcessedRecord(input_ids=input_ids, output_ids=output_ids)


class MakeTrainRecord:
    def __init__(self, eos_token_id: int,
                 max_query_tokens: int, max_response_tokens: int,
                 max_total_tokens: int, force_pad_tokens: int,
                 force_query_tokens: int, force_response_tokens: int,
                 ) -> None:
        self.eos_token_id = eos_token_id
        self.pad_token_in_train_label: int = IGNORE_PAD_TOKEN_ID_IN_LABEL
        self.max_query_tokens: int = max_query_tokens
        self.max_response_tokens: int = max_response_tokens
        self.max_total_tokens = max_total_tokens
        self.force_pad_tokens = force_pad_tokens
        self.force_query_tokens: int = force_query_tokens
        self.force_response_tokens: int = force_response_tokens
        assert self.max_query_tokens % 8 == 0 or self.max_query_tokens <= 0
        assert self.max_response_tokens % 8 == 0 or self.max_response_tokens <= 0
        assert self.max_total_tokens % 8 == 0 or self.max_total_tokens <= 0
        assert self.force_query_tokens % 8 == 0 or self.force_query_tokens <= 0
        assert self.force_response_tokens % 8 == 0 or self.force_response_tokens <= 0

    def __call__(self, record: ProcessedRecord) -> ProcessedRecord:
        if self.max_query_tokens > 0 and len(record.input_ids) > self.max_query_tokens:
            record.input_ids = record.input_ids[:self.max_query_tokens]
        if self.force_query_tokens > 0:
            record.input_ids = self.force_len(record.input_ids, self.force_query_tokens, pad_start=True)
        record.output_ids.append(self.eos_token_id)
        if self.max_response_tokens > 0 and len(record.output_ids) > self.max_response_tokens:
            record.output_ids = record.output_ids[:self.max_response_tokens]
        if self.max_total_tokens > 0 and (len(record.input_ids)+len(record.output_ids)) > self.max_total_tokens:
            if len(record.input_ids) > self.max_total_tokens:
                record.output_ids = []
                record.input_ids = record.input_ids[:self.max_total_tokens]
            else:
                record.output_ids = record.output_ids[:self.max_total_tokens-len(record.input_ids)]
        if self.force_response_tokens > 0:
            record.output_ids = self.force_len(record.output_ids, self.force_response_tokens, pad_start=False)
        if self.force_pad_tokens > 0:
            if (len(record.input_ids)+len(record.output_ids)) > self.force_pad_tokens:
                if len(record.input_ids) > self.force_pad_tokens:
                    record.output_ids = []
                    record.input_ids = record.input_ids[:self.force_pad_tokens]
                else:
                    record.output_ids = record.output_ids[:self.force_pad_tokens-len(record.input_ids)]
            else:
                record.output_ids += [self.eos_token_id]*(self.force_pad_tokens-len(record.input_ids)-len(record.output_ids))
        if (len(record.input_ids) + len(record.output_ids)) % 8 != 0:
            record.output_ids += [self.eos_token_id] * (8 - ((len(record.input_ids) + len(record.output_ids)) % 8))
        return ProcessedRecord(
            input_ids=record.input_ids + record.output_ids,
            output_ids=[self.pad_token_in_train_label] * len(record.input_ids) + record.output_ids)

    def force_len(self, tokens: List[int], force_len: int, pad_start: bool) -> List[int]:
        out = [0]*force_len
        if pad_start:
            out[-min(force_len, len(tokens)):] = tokens[-min(force_len, len(tokens)):]
        else:
            out[:min(force_len, len(tokens))] = tokens[:min(force_len, len(tokens))]
        return out


def make_dataset_process_fn(build_prompt_str: str, tokenizer_path: str, eos_token_id: int,
                            max_query_tokens: int, max_response_tokens: int,
                            max_total_tokens: int,
                            force_pad_tokens: int,
                            force_query_tokens: int,
                            force_response_tokens: int,
                            tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
                            max_samples: int = -1) -> Callable[[Record], ProcessedRecord]:
    translator = BuildQueryAndTranslate(build_prompt_str=build_prompt_str, tokenizer_path=tokenizer_path, tokenizer=tokenizer)
    transform = MakeTrainRecord(eos_token_id=eos_token_id,
                                max_query_tokens=max_query_tokens, max_response_tokens=max_response_tokens,
                                max_total_tokens=max_total_tokens, force_pad_tokens=force_pad_tokens,
                                force_query_tokens=force_query_tokens, force_response_tokens=force_response_tokens)

    def process(raw_record: Record) -> ProcessedRecord:
        return transform(translator(raw_record))

    return process
