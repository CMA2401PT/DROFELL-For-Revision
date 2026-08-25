from typing import List, Tuple, Optional, Union
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer  # type: ignore

from .common import Sequence, SequenceCore, StopConfig, TokenizerSpecialTokens


class ExtendTokenizer:
    def __init__(self, actual_tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer]):
        self.underlay_tokenizer = actual_tokenizer
        self.skip_special_tokens: bool = False
        self.spaces_between_special_tokens: bool = True
        self.pad_token_id = actual_tokenizer.pad_token_id
        self.bos_token_id = actual_tokenizer.bos_token_id
        self.eos_token_id = actual_tokenizer.eos_token_id

    @staticmethod
    def load_tokenier(path: str, cfg: TokenizerSpecialTokens):
        tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer] = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=path, use_fast=True)
        tokenizer.pad_token_id = cfg.pad_token_id
        tokenizer.bos_token_id = cfg.bos_token_id
        tokenizer.eos_token_id = cfg.eos_token_id
        return ExtendTokenizer(actual_tokenizer=tokenizer)

    def encode(self, prompt: str) -> List[int]:
        return self.underlay_tokenizer.encode(prompt)

    def init_seq(self, prompt: str, max_tokens: int, ignore_eos: bool, stop_str: Optional[List[str]] = None) -> Sequence:
        seq = Sequence(
            core=SequenceCore(
                stop_config=StopConfig(
                    max_tokens=max_tokens,
                    ignore_eos=ignore_eos,
                )
            ),
            prompt=prompt,
            stop_cond_text=stop_str if stop_str is not None else [],
        )
        self.encode_seq(seq)
        return seq

    def encode_seq(self, seq: Sequence):
        assert (len(seq.core.prompt_token_ids)) == 0 != (seq.prompt is not None)
        if seq.prompt is not None:
            seq.core.prompt_token_ids = self.encode(seq.prompt)
        for stop_str in seq.stop_cond_text:
            stop_tokens = self.encode(stop_str)
            seq.core.stop_config.stopped_tokens_sequences.append(stop_tokens)

        assert self.underlay_tokenizer.eos_token_id is not None
        seq.core.stop_config.stopped_eos_token_ids = self.underlay_tokenizer.eos_token_id

    def _convert_tokens_to_string_with_added_encoders(self,
                                                      output_tokens: List[str],
                                                      skip_special_tokens: bool,
                                                      spaces_between_special_tokens: bool,
                                                      ) -> str:
        sub_texts: List[str] = []
        current_sub_text: List[str] = []
        all_special_tokens = set(self.underlay_tokenizer.all_special_tokens)
        for token in output_tokens:
            if skip_special_tokens and token in all_special_tokens:
                continue
            if token in self.underlay_tokenizer.get_added_vocab():
                if current_sub_text:
                    sub_text = self.underlay_tokenizer.convert_tokens_to_string(current_sub_text)
                    sub_texts.append(sub_text)
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_text = self.underlay_tokenizer.convert_tokens_to_string(current_sub_text)
            sub_texts.append(sub_text)
        if spaces_between_special_tokens:
            return " ".join(sub_texts)
        else:
            return "".join(sub_texts)

    def incrementally_decode(self,
                             all_input_ids: List[int],
                             prev_decoded_tokens: Optional[List[str]],
                             prefix_offset: int = 0,
                             read_offset: int = 0) -> Tuple[List[str], str, int, int]:
        new_token_id = all_input_ids[-1]
        # This is the first iteration for this sequence
        if prev_decoded_tokens is None:
            new_tokens: List[str] = self.underlay_tokenizer.convert_ids_to_tokens(
                all_input_ids, skip_special_tokens=self.skip_special_tokens)  # type: ignore
            output_tokens: List[str] = new_tokens
            # 5 is an arbitrary value that should work for all
            # tokenizers (bigger = more conservative).
            # Subtract 1 extra to account for the generated token.
            prefix_offset = max(len(output_tokens) - 6, 0)
            # If the first new token is a special token, we can't skip 1 extra token
            if self.skip_special_tokens and new_token_id in self.underlay_tokenizer.all_special_ids:
                read_offset = max(len(output_tokens), 0)
            else:
                read_offset = max(len(output_tokens) - 1, 0)
        else:
            # Put new_token_id in a list so skip_special_tokens is respected
            new_tokens = self.underlay_tokenizer.convert_ids_to_tokens(
                [new_token_id], skip_special_tokens=self.skip_special_tokens)  # type: ignore
            output_tokens = prev_decoded_tokens + new_tokens

        # The prefix text is necessary only to defeat cleanup algorithms in
        # the decode which decide to add a space or not depending on the
        # surrounding ids.
        if self.underlay_tokenizer.is_fast or not self.underlay_tokenizer.get_added_vocab():
            prefix_text = self.underlay_tokenizer.convert_tokens_to_string(
                output_tokens[prefix_offset:read_offset])
            new_text = self.underlay_tokenizer.convert_tokens_to_string(
                output_tokens[prefix_offset:])
        else:
            prefix_text = self._convert_tokens_to_string_with_added_encoders(
                output_tokens[prefix_offset:read_offset],
                skip_special_tokens=self.skip_special_tokens,
                spaces_between_special_tokens=self.spaces_between_special_tokens,
            )
            new_text = self._convert_tokens_to_string_with_added_encoders(
                output_tokens[prefix_offset:],
                skip_special_tokens=self.skip_special_tokens,
                spaces_between_special_tokens=self.spaces_between_special_tokens,
            )

        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            # utf-8 char at the end means it's a potential unfinished byte sequence
            # from byte fallback tokenization.
            # If it's in the middle, it's probably a real invalid id generated
            # by the model
            new_text = new_text[len(prefix_text):]
            return new_tokens, new_text, read_offset, len(output_tokens)
        else:
            return new_tokens, "", prefix_offset, read_offset

    def incrementally_decode_seq(self, seq: Sequence) -> None:
        new_tokens, new_output_text, prefix_offset, read_offset = \
            self.incrementally_decode(
                all_input_ids=seq.core.prompt_token_ids+seq.core.output_token_ids,
                prev_decoded_tokens=seq.prev_decoded_tokens,
                prefix_offset=seq.prefix_offset,
                read_offset=seq.read_offset,
            )
        if seq.prev_decoded_tokens is None:
            seq.prev_decoded_tokens = new_tokens
        else:
            seq.prev_decoded_tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_output_text
        seq.new_output_text = new_output_text

        if seq.core.stopped:
            for stop_str in seq.stop_cond_text:
                if seq.output_text.endswith(stop_str):
                    # Truncate the output text so that the stop string is
                    # not included in the output.
                    seq.output_text = seq.output_text[:-len(stop_str)]

    def decode_seq(self, seq: Sequence) -> None:
        seq.output_text = self.underlay_tokenizer.decode(seq.core.output_token_ids)

    def decode(self, tokens: List[int]) -> str:
        return self.underlay_tokenizer.decode(tokens)
