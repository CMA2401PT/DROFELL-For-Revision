import time
from typing import Dict, Tuple, Optional, List, Callable
from uuid import uuid4

import torch

from operators.cache_engine.common import CacheEngine
from ..interface.basic_types import Time, TokenID, VirtualBlockNumber
from ..interface.sample import SequenceSampleOutput
from ..interface.basic_types import SeqGroupSelectedSeqIDs, SeqID
from ..interface.ptuning import PtuningBlocks, PtuningHandler
from ..interface.infer_request_outputs import RequestOutput
from ..interface.meta import GeneratingSeqGroupMeta, SeqGeneratingMetaForExecutor, SeqGroupMetaForSampler, SeqGroupPrefillMeta, SeqPrefillMetaForExecuter
from ..interface.sampling_params import SamplingParams
from ..interface.sequence import Sequence, SequenceGroup, SequenceStatus, SequenceTokenIDsData
from ..interface.block_space_io import AllocStatus, VirtualBlockSpaceIO
from ..interface.meta import GeneratingMeta
from ..interface.scheduler import Scheduler

from operators import flags
from operators.core.tokenizer.real import ExtendTokenizer


class VllmInferManager:
    def __init__(self, scheduler: Scheduler, block_space_io: VirtualBlockSpaceIO, tokenizer: ExtendTokenizer) -> None:
        super().__init__()
        self.scheduler: Scheduler = scheduler
        self.block_space_io: VirtualBlockSpaceIO = block_space_io
        self.tokenizer: ExtendTokenizer = tokenizer
        self.get_time: Callable[[], Time] = lambda: time.monotonic()
        self.ptuning_handlers: Dict[str, PtuningHandler] = {}

    @staticmethod
    def next_seq_id() -> str:
        return str(uuid4())

    def add_ptuning(self, ptuning_id: str, build_prompt_str: str, ptuning_kv_cache_cpu: List[Tuple[torch.Tensor, torch.Tensor]], engine: CacheEngine) -> PtuningHandler:
        block_table = self.block_space_io.allocate_ptuning(blocks_count=ptuning_kv_cache_cpu[0][0].size(0))
        blocks = [b.block_number for b in block_table]

        engine.put_blocks(
            target_blocks=blocks,
            kv_cache=[
                [
                    (k, v) for (k, v) in zip(lk, lv)
                ]
                for lk, lv in ptuning_kv_cache_cpu
            ],
        )
        slot_mapping: List[int] = []
        for block_i in blocks:
            for i in range(self.block_space_io.block_size):
                slot_mapping.append(block_i * self.block_space_io.block_size + i)
        handler = PtuningHandler(
            ptuning_id=ptuning_id,
            build_prompt_str=build_prompt_str,
            ptuning_blocks=blocks,
            ptuning_table=block_table,
            ref_count=1
        )
        self.ptuning_handlers[ptuning_id] = handler
        return handler

    def free_ptuning(self, ptuning_id: str) -> None:
        assert ptuning_id in self.ptuning_handlers.keys(), ptuning_id
        ptuning_handler = self.ptuning_handlers[ptuning_id]
        ptuning_handler.ref_count -= 1
        if ptuning_handler.ref_count == 0:
            del self.ptuning_handlers[ptuning_id]
            self.block_space_io.free_ptuning(ptuning_block_table=ptuning_handler.ptuning_table)

    def add_request(self, request_id: str, sampling_parameters: SamplingParams, ptuning_id: Optional[str] = "", prompt: Optional[str] = None, prompt_token_ids: Optional[List[TokenID]] = None) -> bool:
        if ptuning_id is None:
            ptuning_id = ""
        if ptuning_id != "":
            assert ptuning_id in self.ptuning_handlers.keys()
            if prompt is not None:
                prompt = self.ptuning_handlers[ptuning_id].build_prompt_str.format(prompt)
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt=prompt)
        else:
            assert prompt is None
            prompt = ""

        num_prompt_tokens: int = len(prompt_token_ids)
        if num_prompt_tokens > self.scheduler.prompt_limit:
            return False
        arrival_time: Time = self.get_time()
        seq_id: str = self.next_seq_id()
        seq = Sequence(seq_id=seq_id,
                       prompt=prompt,
                       prompt_token_ids=prompt_token_ids,
                       block_size=self.block_space_io.block_size)
        seq_group = SequenceGroup(request_id=request_id,
                                  seqs=[seq],
                                  sampling_params=sampling_parameters,
                                  arrival_time=arrival_time,
                                  ptuning_id=ptuning_id)
        if self.block_space_io.can_allocate_prefill(seq_group=seq_group) != AllocStatus.NEVER:
            self.scheduler.waiting.append(seq_group)
            if ptuning_id != "":
                self.ptuning_handlers[ptuning_id].ref_count += 1
            return True
        else:
            return False

    def step_pre_prefill(self) -> List[Tuple[SeqGroupSelectedSeqIDs, SeqGroupPrefillMeta]]:
        prefill_sequence_seq_groups: List[SequenceGroup] = self.scheduler.schedule_prefill(block_space_io=self.block_space_io)
        if len(prefill_sequence_seq_groups) > 0:
            metas: List[Tuple[SeqGroupSelectedSeqIDs, SeqGroupPrefillMeta]] = []
            for prefill_sequence_seq_group in prefill_sequence_seq_groups:
                ptuning_blocks: Optional[PtuningBlocks] = None
                if prefill_sequence_seq_group.ptuning_id is not None and prefill_sequence_seq_group.ptuning_id != "":
                    ptuning_handler = self.ptuning_handlers[prefill_sequence_seq_group.ptuning_id]
                    ptuning_blocks = ptuning_handler.ptuning_blocks
                running_seqs: List[Sequence] = prefill_sequence_seq_group.get_seqs(
                    state=SequenceStatus.RUNNING)
                assert len(running_seqs) == 1
                seq: Sequence = running_seqs[0]
                seq_id = seq.seq_id

                tokens: List[TokenID] = seq.token_ids_data.get_token_ids()
                positions = list(range(len(tokens)))
                slots: List[int] = []
                block_table: List[VirtualBlockNumber] = self.block_space_io.get_block_table_by_id(
                    seq_id=seq_id)
                for i in range(len(tokens)):
                    block_number = block_table[i // self.block_space_io.block_size]
                    block_offset: int = i % self.block_space_io.block_size
                    slot: int = block_number * self.block_space_io.block_size + block_offset
                    slots.append(slot)

                meta = SeqGroupPrefillMeta(
                    seq_group=prefill_sequence_seq_group,
                    excuter_meta=SeqPrefillMetaForExecuter(
                        tokens=tokens,
                        positions=positions,
                        slots=slots,
                        ptuning=ptuning_blocks,
                    ),
                    sampler_meta=SeqGroupMetaForSampler(
                        seqs_token_ids_data={seq_id: seq.token_ids_data},
                        sampling_parameters=prefill_sequence_seq_group.sampling_params
                    )
                )

                metas.append(([seq_id], meta))
            return metas
        else:
            return []

    def step_pre_generating(self) -> Optional[GeneratingMeta]:
        generating_sequence_groups, memory_oprations = \
            self.scheduler.schedule_generating(block_space_io=self.block_space_io)
        seq_group_metas: List[Tuple[SeqGroupSelectedSeqIDs, GeneratingSeqGroupMeta]] = []
        for generating_sequence_group in generating_sequence_groups:
            running_seqs: List[Sequence] = generating_sequence_group.get_seqs(
                state=SequenceStatus.RUNNING)
            running_seq_ids: List[SeqID] = []
            executer_meta: List[SeqGeneratingMetaForExecutor] = []

            seqs_token_ids_data: Dict[SeqID, SequenceTokenIDsData] = {}

            for seq in running_seqs:
                seq_data: SequenceTokenIDsData = seq.token_ids_data
                seq_id = seq.seq_id

                running_seq_ids.append(seq_id)
                seqs_token_ids_data[seq.seq_id] = seq_data

                per_seq_generating_meta_for_executor = SeqGeneratingMetaForExecutor()
                per_seq_generating_meta_for_executor.token = seq_data.get_last_token_id()
                per_seq_generating_meta_for_executor.context_len = seq_data.get_len()
                per_seq_generating_meta_for_executor.position = per_seq_generating_meta_for_executor.context_len - 1

                block_table: List[VirtualBlockNumber] = self.block_space_io.get_block_table_by_id(
                    seq_id=seq_id)
                block_number = block_table[per_seq_generating_meta_for_executor.position //
                                           self.block_space_io.block_size]
                block_offset: int = per_seq_generating_meta_for_executor.position % self.block_space_io.block_size
                slot: int = block_number * self.block_space_io.block_size + block_offset
                per_seq_generating_meta_for_executor.slot = slot
                per_seq_generating_meta_for_executor.block_table = block_table

                if generating_sequence_group.ptuning_id is not None and generating_sequence_group.ptuning_id != "":
                    ptuning_handler = self.ptuning_handlers[generating_sequence_group.ptuning_id]
                    tuning_block_table: list[VirtualBlockNumber] = ptuning_handler.ptuning_blocks
                    per_seq_generating_meta_for_executor.block_table = tuning_block_table + \
                        per_seq_generating_meta_for_executor.block_table
                    per_seq_generating_meta_for_executor.context_len += len(ptuning_handler.ptuning_blocks)*flags.BLOCK_SIZE

                executer_meta.append(per_seq_generating_meta_for_executor)

            seq_group_metas.append((running_seq_ids, GeneratingSeqGroupMeta(
                seq_group=generating_sequence_group,
                executer_meta=executer_meta,
                sampler_meta=SeqGroupMetaForSampler(
                    seqs_token_ids_data=seqs_token_ids_data,
                    sampling_parameters=generating_sequence_group.sampling_params,
                ),
            )))
        generating_meta = GeneratingMeta(
            memory_operations=memory_oprations,
            seq_group_metas=seq_group_metas
        )
        if (generating_meta.memory_operations is None) and not generating_meta.seq_group_metas:
            return None
        return generating_meta

    def _decode_sequence(self, seq: Sequence):
        """Decodes the new token for a sequence."""
        (new_tokens, new_output_text, prefix_offset,
         read_offset) = self.tokenizer.incrementally_decode(
            all_input_ids=seq.get_token_ids(),
            prev_decoded_tokens=seq.decoded_tokens,
            prefix_offset=seq.prefix_offset,
            read_offset=seq.read_offset,
        )
        if seq.decoded_tokens is None:
            seq.decoded_tokens = new_tokens
        else:
            seq.decoded_tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_output_text

    def _check_stop(self, seq: Sequence, sampling_params: SamplingParams):
        """Stop the finished sequences."""
        for stop_str in sampling_params.stop:
            if seq.output_text.endswith(stop_str):
                # Truncate the output text so that the stop string is
                # not included in the output.
                seq.output_text = seq.output_text[:-len(stop_str)]
                seq.state = SequenceStatus.FINISHED_STOPPED
                return
        if seq.get_last_token_id() in sampling_params.stop_token_ids:
            seq.state = SequenceStatus.FINISHED_STOPPED
            return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self.scheduler.max_model_len:
            seq.state = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() == sampling_params.max_tokens:
            seq.state = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.get_last_token_id() == self.tokenizer.eos_token_id):
            seq.state = SequenceStatus.FINISHED_STOPPED
            return

    # def _check_beam_search_early_stopping(self,
    #                                       early_stopping: Union[bool, str],
    #                                       sampling_params: SamplingParams,
    #                                       best_running_seq: Sequence,
    #                                       current_worst_seq: Sequence,
    #                                       ) -> bool:
    #     assert sampling_params.use_beam_search
    #     length_penalty = sampling_params.length_penalty
    #     if early_stopping is True:
    #         return True

    #     current_worst_score = (current_worst_seq.get_beam_search_score(
    #         length_penalty=length_penalty,
    #         eos_token_id=self.tokenizer.eos_token_id))
    #     if early_stopping is False:
    #         highest_attainable_score = (best_running_seq.get_beam_search_score(
    #             length_penalty=length_penalty,
    #             eos_token_id=self.tokenizer.eos_token_id))
    #     else:
    #         assert early_stopping == "never"
    #         if length_penalty > 0.0:
    #             # If length_penalty > 0.0, beam search will prefer longer
    #             # sequences. The highest attainable score calculation is
    #             # based on the longest possible sequence length in this case.
    #             max_possible_length = max(
    #                 best_running_seq.get_prompt_len() +
    #                 sampling_params.max_tokens,
    #                 self.scheduler.max_model_len)
    #             highest_attainable_score = (
    #                 best_running_seq.get_beam_search_score(
    #                     length_penalty=length_penalty,
    #                     eos_token_id=self.tokenizer.eos_token_id,
    #                     seq_len=max_possible_length))
    #         else:
    #             # Otherwise, beam search will prefer shorter sequences. The
    #             # highest attainable score calculation is based on the current
    #             # sequence length.
    #             highest_attainable_score = (
    #                 best_running_seq.get_beam_search_score(
    #                     length_penalty=length_penalty,
    #                     eos_token_id=self.tokenizer.eos_token_id))
    #     return current_worst_score >= highest_attainable_score

    def step_post_process(self, sampler_output: List[List[SequenceSampleOutput]], seq_groups: List[SequenceGroup]) -> List[RequestOutput]:
        for seq_group, samples in zip(seq_groups, sampler_output):
            # append new token / kick off & add seq in seq group
            parent_seqs: List[Sequence] = seq_group.get_seqs(
                state=SequenceStatus.RUNNING)
            # existing_finished_seqs: List[Sequence] = seq_group.get_finished_seqs()
            parent_child_dict: Dict[SeqID, List[SequenceSampleOutput]] = {
                parent_seq.seq_id: []
                for parent_seq in parent_seqs
            }
            for sample in samples:
                parent_child_dict[sample.parent_seq_id].append(sample)
            child_seqs: List[Tuple[Sequence, Sequence]] = []
            for parent in parent_seqs:
                child_samples: List[SequenceSampleOutput] = parent_child_dict[parent.seq_id]
                if len(child_samples) == 0:
                    parent.state = SequenceStatus.FINISHED_ABORTED
                    seq_group.remove(seq_id=parent.seq_id)
                    self.block_space_io.free(seq=parent)
                    continue
                for child_sample in child_samples[:-1]:
                    new_child_seq_id = self.next_seq_id()
                    child: Sequence = parent.fork(new_seq_id=new_child_seq_id)
                    child.append_token_id(
                        token_id=child_sample.output_token, logprobs=child_sample.logprobs)
                    child_seqs.append((child, parent))

                # Continue the parent sequence for the last child sample.
                # We reuse the parent sequence here to reduce redundant memory
                # copies, especially when using non-beam search sampling methods.
                last_child_sample: SequenceSampleOutput = child_samples[-1]
                parent.append_token_id(
                    token_id=last_child_sample.output_token, logprobs=last_child_sample.logprobs)
                child_seqs.append((parent, parent))

            for seq, _ in child_seqs:
                self._decode_sequence(seq=seq)
                self._check_stop(
                    seq=seq, sampling_params=seq_group.sampling_params)

            # Non-beam search case
            # if not seq_group.sampling_params.use_beam_search:
            # For newly created child sequences, add them to the sequence group
            # and fork them in block manager if they are not finished.
            for seq, parent in child_seqs:
                if seq is not parent:
                    seq_group.add(seq=seq)
                    if not seq.is_finished():
                        self.block_space_io.fork(
                            parent_seq=parent, child_seq=seq)

            # Free the finished and selected parent sequences' memory in block
            # manager. Keep them in the sequence group as candidate output.
            # NOTE: we need to fork the new sequences before freeing the
            # old sequences.
            for seq, parent in child_seqs:
                if seq is parent and seq.is_finished():
                    self.block_space_io.free(seq=seq)
            # else:
            #     # Beam search case
            #     # Select the child sequences to keep in the sequence group.
            #     selected_child_seqs: List[Tuple[Sequence, Sequence]] = []
            #     unselected_child_seqs: List[Tuple[Sequence, Optional[Sequence]]] = [
            #     ]
            #     beam_width: int = seq_group.sampling_params.best_of
            #     length_penalty: float = seq_group.sampling_params.length_penalty

            #     # Select the newly finished sequences with the highest scores
            #     # to replace existing finished sequences.
            #     # Tuple of (seq, parent, is_new)
            #     existing_finished_seqs_info_extended: list[tuple[Sequence, None, Literal[False]]] = [
            #         (seq, None, False) for seq in existing_finished_seqs]
            #     new_finished_seqs: list[tuple[Sequence, Sequence, Literal[True]]] = [
            #         (seq, parent, True) for seq, parent in child_seqs if seq.is_finished()]
            #     all_finished_seqs: list[tuple[Sequence, Sequence, Literal[True]] | tuple[Sequence,
            #                                                                              None, Literal[False]]] = existing_finished_seqs_info_extended + new_finished_seqs
            #     # Sort the finished sequences by their scores.
            #     all_finished_seqs.sort(
            #         key=lambda x: x[0].get_beam_search_score(length_penalty=length_penalty,
            #                                                  eos_token_id=self.tokenizer.eos_token_id),
            #         reverse=True)

            #     for seq, parent, is_new in all_finished_seqs[beam_width:]:
            #         if is_new:
            #             # A newly generated child sequence finishes but has a low
            #             # score, so we will not add it into the sequence group.
            #             # Additionally, if this sequence is a continuation of a
            #             # parent sequence, we will need remove the parent sequence
            #             # from the sequence group.
            #             unselected_child_seqs.append((seq, parent))
            #         else:
            #             # An existing finished sequence has a low score, so we will
            #             # remove it from the sequence group.
            #             seq_group.remove(seq.seq_id)

            #     # select the top beam_width sequences from the running
            #     # sequences for the next iteration to continue the beam
            #     # search.
            #     running_child_seqs: list[tuple[Sequence, Sequence]] = [(seq, parent) for seq, parent in child_seqs
            #                                                            if not seq.is_finished()]
            #     # Sort the running sequences by their scores.
            #     running_child_seqs.sort(
            #         key=lambda x: x[0].
            #         get_beam_search_score(
            #             length_penalty=length_penalty, eos_token_id=self.tokenizer.eos_token_id),
            #         reverse=True)

            #     # Check if we can stop the beam search.
            #     if len(running_child_seqs) == 0:
            #         # No running sequences, stop the beam search.
            #         stop_beam_search = True
            #     elif len(all_finished_seqs) < beam_width:
            #         # Not enough finished sequences, continue the beam search.
            #         stop_beam_search = False
            #     else:
            #         # Check the early stopping criteria
            #         best_running_seq: Sequence = running_child_seqs[0][0]
            #         current_worst_seq: Sequence = all_finished_seqs[beam_width - 1][0]
            #         stop_beam_search: bool = self._check_beam_search_early_stopping(
            #             early_stopping=seq_group.sampling_params.early_stopping,
            #             sampling_params=seq_group.sampling_params,
            #             best_running_seq=best_running_seq,
            #             current_worst_seq=current_worst_seq,
            #         )

            #     if stop_beam_search:
            #         # Stop the beam search and remove all the running sequences from
            #         # the sequence group.
            #         unselected_child_seqs.extend(running_child_seqs)
            #     else:
            #         # Continue the beam search and select the top beam_width sequences
            #         # to continue the beam search.
            #         selected_child_seqs.extend(running_child_seqs[:beam_width])
            #         # The remaining running sequences will not be used in the next
            #         # iteration. Again, if these sequences are continuations of
            #         # parent sequences, we will need to remove the parent sequences
            #         # from the sequence group.
            #         unselected_child_seqs.extend(
            #             running_child_seqs[beam_width:])

            #     # For newly created child sequences, add them to the sequence group
            #     # and fork them in block manager if they are not finished.
            #     for seq, parent in selected_child_seqs:
            #         if seq is not parent:
            #             seq_group.add(seq=seq)
            #             if not seq.is_finished():
            #                 self.block_space_io.fork(
            #                     parent_seq=parent, child_seq=seq)

            #     # Free the finished and selected parent sequences' memory in block
            #     # manager. Keep them in the sequence group as candidate output.
            #     for seq, parent in selected_child_seqs:
            #         if seq is parent and seq.is_finished():
            #             self.block_space_io.free(seq=seq)

            #     # Remove the unselected parent sequences from the sequence group and
            #     # free their memory in block manager.
            #     for seq, parent in unselected_child_seqs:
            #         if seq is parent:
            #             # Remove the parent sequence if it is not selected for next
            #             # iteration
            #             seq_group.remove(seq_id=seq.seq_id)
            #             self.block_space_io.free(seq=seq)
        finished_seq_groups: List[SequenceGroup] = self.scheduler.free_finished_seq_groups()
        for finished_seq_group in finished_seq_groups:
            if finished_seq_group.ptuning_id is not None and finished_seq_group.ptuning_id != "":
                self.free_ptuning(ptuning_id=finished_seq_group.ptuning_id)

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in (seq_groups):
            request_output: RequestOutput = RequestOutput.create_from_seq_group(
                seq_group=seq_group)
            request_outputs.append(request_output)

        return request_outputs
