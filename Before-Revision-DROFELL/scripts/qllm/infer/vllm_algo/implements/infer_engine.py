from typing import Dict, List, Tuple, Optional, TypeVar
import torch

from operators import flags
from operators.cache_engine.common import CacheEngine


from ..interface.basic_types import TokenID
from ..interface.basic_types import SeqGroupSelectedSeqIDs, SeqID
from ..interface.meta import GeneratingSeqGroupMeta, SeqGeneratingMetaForExecutor, SeqGroupMetaForSampler, SeqGroupPrefillMeta, SeqPrefillMetaForExecuter
from ..interface.sequence import SequenceTokenIDsData
from ..interface.sample import SamplingMetadata, SequenceSampleOutput
from ..interface.sampling_params import SamplingParams, SamplingType
from ..interface.meta import MemoryOperations
from .sampler import RealWorldSampler

from operators.assemble.decoders.common import SimpleLayersPrefillCacheProvider
from operators.container.container import Container
from operators.assemble.decoders.real import DecodersGroup
from operators.core.head.real import ModelHead
from operators.core.tail.real import ModelTail

E = TypeVar('E')


def pad(mutli_lists: List[List[E]], default: E) -> List[List[E]]:
    pad_len = max([len(l) for l in mutli_lists])
    ls = [l+[default]*(pad_len-len(l))
          for l in mutli_lists]
    return ls


class LocalMachineInferEngine:
    def __init__(self,
                 model_head: ModelHead, model_tail: ModelTail,
                 all_decoders: DecodersGroup,
                 cache_engine: CacheEngine,
                 device: torch.device
                 ):
        self.device = device
        self.model_head = model_head
        self.model_tail = model_tail
        self.all_decoders = all_decoders
        self.cache_engine = cache_engine

    async def execute_prefill(
        self,
        container: Container,
        seq_metas: List[SeqPrefillMetaForExecuter]
    ) -> torch.Tensor:
        with container.mode("inference"):
            slot_mapping: List[int] = []
            prompts: List[TokenID] = []
            input_positions: List[int] = []
            prompt_lens: List[int] = []
            last_indices: List[int] = []
            # exist_ptuning_caches: Dict[int, int] = {}
            seq_ptunings: List[Optional[List[int]]] = []
            # seq_ptuning_caches_offset: List[int] = []
            related_blocks: List[int] = []
            offset = 0
            for seq_meta in seq_metas:
                prompt_lens.append(len(seq_meta.tokens))
                prompts += seq_meta.tokens
                input_positions += seq_meta.positions
                slot_mapping += seq_meta.slots
                related_blocks += [s//flags.BLOCK_SIZE for s in seq_meta.slots]
                seq_ptunings.append(seq_meta.ptuning)
                if seq_meta.ptuning is not None:
                    related_blocks += seq_meta.ptuning

                last_indices.append(offset + len(seq_meta.tokens) - 1)
                offset += len(seq_meta.tokens)

            related_blocks = list(set(related_blocks))

            def gen_input(prompts: List[int], slot_mapping: List[int], input_positions: List[int], seq_ptunings: List[Optional[List[int]]]):
                cache_provider, translator = self.cache_engine.get_cache_provider(related_blocks)
                slot_mapping = [translator.translate_slot(s) for s in slot_mapping]
                seq_ptunings = [
                    [translator.translate_block(b) for b in ptuning] if ptuning is not None else None
                    for ptuning in seq_ptunings
                ]
                hidden_states = torch.tensor(prompts, dtype=torch.int32)
                slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int64)
                input_positions_tensor = torch.tensor(input_positions, dtype=torch.int32)
                layers_cache_provider = SimpleLayersPrefillCacheProvider(
                    cache_provider=cache_provider,
                    seq_ptunings=seq_ptunings
                )
                return hidden_states, slot_mapping_tensor, input_positions_tensor, layers_cache_provider

            hidden_states, slot_mapping_tensor, input_positions_tensor, layers_cache_provider = \
                await container.step("gen_input", gen_input,
                                     prompts=prompts, slot_mapping=slot_mapping, input_positions=input_positions, seq_ptunings=seq_ptunings)
            hidden_states = await self.model_head.infer_to_embedding_tokens(container=container, input_tokens_tensor=hidden_states, dtype=torch.float16)
            hidden_states = await self.all_decoders.multi_seqs_prefill_forward(
                container=container,
                hidden_states=hidden_states, position_ids=input_positions_tensor, input_lens=prompt_lens,
                layers_cache_provider=layers_cache_provider,
                slot_mapping=slot_mapping_tensor,
            )
            logits = await self.model_tail.to_logits(container, hidden_states[last_indices])
            return logits

    def do_sampling(
        self,
        logits: torch.Tensor,
        metas: List[Tuple[SeqGroupSelectedSeqIDs, SeqGroupMetaForSampler]],
        is_prompt: bool
    ) -> List[List[SequenceSampleOutput]]:
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
        sampler = RealWorldSampler(vocab_size=self.model_head.vocab_size)

        return sampler.sample(logits=logits, sampling_metadata=SamplingMetadata(
            seq_groups_sampling_params=seq_group_sampling_params,
            is_prompts=[is_prompt for _ in range(len(metas))],
            seq_data=seq_data,
            categorized_sample_indices={
                t: torch.tensor(seq_ids, dtype=torch.int)
                for t, seq_ids in categorized_sample_indices.items()
            },
        ))

    async def infer_prefill(
        self,
        container: Container,
        seq_metas: List[Tuple[SeqGroupSelectedSeqIDs, SeqGroupPrefillMeta]]
    ) -> List[List[SequenceSampleOutput]]:
        executor_input: List[SeqPrefillMetaForExecuter] = []
        sampler_input: List[Tuple[SeqGroupSelectedSeqIDs,
                                  SeqGroupMetaForSampler]] = []
        for seq_group_ids, seq_meta in seq_metas:
            # print(seq_meta.excuter_meta.__dict__)
            executor_input.append(seq_meta.excuter_meta)
            sampler_input.append((seq_group_ids, seq_meta.sampler_meta))
        logits = await self.execute_prefill(container, executor_input)
        sampler_output = await container.step(
            "prompt_sampling", self.do_sampling,
            logits=logits, metas=sampler_input, is_prompt=True)
        return sampler_output

    # def _start_memory_operations(self, memory_operations: Optional[MemoryOperations]) -> None:
    #     if not memory_operations:
    #         return None
    #     # events: Optional[List[torch.cuda.Event]] = None
    #     if memory_operations.blocks_to_swap_in:
    #         self.cache_engine.swap_in(blocks_to_swap_in=memory_operations.blocks_to_swap_in)
    #     if memory_operations.blocks_to_swap_out:
    #         self.cache_engine.swap_out(blocks_to_swap_out=memory_operations.blocks_to_swap_out)
    #     if memory_operations.blocks_to_copy:
    #         raise NotImplementedError
    #         # self.cache_engine.copy(blocks_to_copy=memory_operations.blocks_to_copy)

    # def start_and_wait_memory_operations(self, memory_operations: MemoryOperations) -> None:
    #     raise NotImplementedError
    #     # events = self._start_memory_operations(memory_operations)
    #     # if events:
    #     #     for event in events:
    #     #         event.wait()

    async def execute_generating(
            self,
            container: Container,
            memory_operations: Optional[MemoryOperations], seq_metas: List[SeqGeneratingMetaForExecutor]):
        if memory_operations is not None:
            raise NotImplementedError
            # await container.step("memory_operations", self._start_memory_operations, memory_operations=memory_operations)

        next_tokens: List[int] = []
        positions: List[int] = []
        slots: List[int] = []
        context_lens: List[int] = []
        block_tables: List[List[int]] = []
        for meta in seq_metas:
            next_tokens.append(meta.token)
            positions.append(meta.position)
            context_lens.append(meta.context_len)
            slots.append(meta.slot)
            block_tables.append(meta.block_table)
        with container.mode("inference"):
            def gen_input(block_tables: List[List[int]], context_lens: List[int], positions: List[int], slots: List[int], next_tokens: List[int]):
                cache_provider, translator = self.cache_engine.get_cache_provider(list(set([
                    b
                    for bs in block_tables
                    for b in bs
                ])))
                block_tables = [
                    [translator.translate_block(b) for b in bs] for bs in block_tables
                ]
                slots = [translator.translate_slot(s) for s in slots]
                block_tables = pad(block_tables, 0)

                block_tables_t = torch.tensor(block_tables, dtype=torch.int32)
                context_lens_t = torch.tensor(context_lens, dtype=torch.int32)
                positions_t = torch.tensor(positions, dtype=torch.int32)
                slot_mapping_t = torch.tensor(slots, dtype=torch.int64)
                input_tokens_tensor = torch.tensor(next_tokens, dtype=torch.int32)
                return block_tables_t, context_lens_t, positions_t, slot_mapping_t, input_tokens_tensor, cache_provider

            block_tables_t, context_lens_t, positions_t, slot_mapping_t, input_tokens_tensor, cache_provider = \
                await container.step("gen_input", gen_input,
                                     block_tables=block_tables, context_lens=context_lens, positions=positions, slots=slots, next_tokens=next_tokens)

            hidden_states = await self.model_head.infer_to_embedding_tokens(container, input_tokens_tensor=input_tokens_tensor, dtype=torch.float16)
            hidden_states = hidden_states.unsqueeze(1)
            # max_context_len = max(context_lens) + 1
            # # pad block table
            # max_block_table_len = max([len(t) for t in block_tables])
            # padded_block_table = [
            #     t + [0] * (max_block_table_len - len(t)) for t in block_tables]

            # to torch.Tensor
            # input_tokens_tensor = torch.tensor(
            #     input_tokens, device=self.device, dtype=torch.int32)
            # next_input_embedding = self.model_head.to_embedding_tokens(
            #     input_tokens_tensor, torch.float16).unsqueeze(1)
            # position_ids_tensor: torch.Tensor = torch.tensor(
            #     input_positions, device=self.device, dtype=torch.int32)  # type:ignore

            hidden_states = await self.all_decoders.multi_seqs_generating_forward(
                container=container,
                hidden_states=hidden_states,
                position_ids=positions_t,  # [8],device, int32
                context_len_max=max(context_lens),  # 9
                context_lens=context_lens_t,  # [9],device, int32
                block_tables=block_tables_t,  # [[]], device, int32
                slot_mapping=slot_mapping_t,  # [], device int64
                layers_cache_provider=cache_provider,
            )

            logits = await self.model_tail.to_logits(container, hidden_states)
            # generating_complex = self.cache_engine.make_generating_cache_with_sequences_info_provider(
            #     slot_mapping, context_lens, block_tables
            # )
            # slots_tensor = torch.tensor(
            #     slot_mapping, dtype=torch.long, device=self.device)
            # context_lens_tensor = torch.tensor(
            #     context_lens, device=self.device, dtype=torch.int)
            # block_tables_tensor = torch.tensor(
            #     padded_block_table, dtype=torch.int, device=self.device)
            # hidden_states = self.all_decoders.multi_seqs_generating_forward(
            #     hidden_states=next_input_embedding, position_ids=position_ids_tensor,
            #     generating_complex=generating_complex
            # )

            # execute model
            # hidden_states = self.all_decoders.infer_decoding_forward_paged_attention(
            #     hidden_states=next_input_embedding, position_ids=position_ids_tensor,
            #     context_lens=context_lens_tensor, max_context_len=max_context_len,
            #     block_tables=block_tables_tensor, slots_tensor=slots_tensor,
            #     cache_engine=self.cache_engine, events=events
            # )
            # logits = self.model_tail.to_logits(hidden_states.float())
            return logits.squeeze(1)

    async def infer_generating(
        self,
        container: Container,
        memory_operations: Optional[MemoryOperations],
        seq_group_metas: List[Tuple[SeqGroupSelectedSeqIDs, GeneratingSeqGroupMeta]]
    ) -> List[List[SequenceSampleOutput]]:
        generating_seqs: List[SeqGeneratingMetaForExecutor] = []
        sampling_metas: List[Tuple[SeqGroupSelectedSeqIDs,
                                   SeqGroupMetaForSampler]] = []
        for (seq_group_seq_ids, generating_seq_group_meta) in seq_group_metas:
            generating_seq_group_meta: GeneratingSeqGroupMeta
            for seq_meta in generating_seq_group_meta.executer_meta:
                generating_seqs.append(seq_meta)
                # print(seq_meta.__dict__)
            sampling_metas.append(
                (seq_group_seq_ids, generating_seq_group_meta.sampler_meta))
        logits = await self.execute_generating(container=container, memory_operations=memory_operations, seq_metas=generating_seqs)
        sampler_output = await container.step(
            "generate_sampling", self.do_sampling,
            logits=logits, metas=sampling_metas, is_prompt=False)
        return sampler_output
