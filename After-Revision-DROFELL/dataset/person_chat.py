import json
from typing import Iterable, Optional, Union
from transformers.tokenization_utils import PreTrainedTokenizer  # type:ignore
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast  # type:ignore
from .common import Record, make_dataset_process_fn
from .dataset_builder import SimpleDataset, build_dataset


def make_train_dataset_person_chat(
    json_file: str,
    build_prompt_str: str,
    tokenizer_path: str,
    eos_token_id: int,
    max_query_tokens: int,
    max_response_tokens: int,
    cache_dir: str,
    tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
    force_query_tokens: int = 0, force_response_tokens: int = 0,
    max_total_tokens: int = 0,
    max_samples: int = 0,
) -> SimpleDataset:
    def read_fn() -> Iterable[Record]:
        sample_count = 0
        with open(json_file, "r") as f:
            groups = json.load(f)
            total_groups = len(groups)
            for group_i, examples in enumerate(groups):
                print(f"\rprogress: {group_i}/{total_groups} {group_i/total_groups:.2%}", end="")
                for example in examples['utterances']:
                    history = "\n-----\n".join(example["history"])
                    for candidate in example["candidates"]:
                        sample_count += 1
                        yield Record(history=[], query=history, answer=candidate)
                        if max_samples > 0 and sample_count > max_samples:
                            return
    return build_dataset(
        source_name=json_file, source_provide_fn=read_fn,
        process_fn_name="BuildQueryAndTranslate->MakeTrainRecord->Pad8", process_fn=make_dataset_process_fn,
        process_fn_args=None, process_fn_kwargs={
            "build_prompt_str": build_prompt_str,
            "tokenizer_path": tokenizer_path,
            "eos_token_id": eos_token_id,
            "max_query_tokens": max_query_tokens,
            "max_response_tokens": max_response_tokens,
            "max_total_tokens": max_total_tokens,
            "force_query_tokens": force_query_tokens,
            "force_response_tokens": force_response_tokens,
            "max_samples": max_samples,
        },
        cache_dir=cache_dir,
        non_hashable_process_fn_kwargs={
            "tokenizer": tokenizer
        }
    )
