import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TypeVar, cast
from operators.cache_engine.common import BlockManager, CacheEngine
from operators.cache_engine.scalable import ScalableCacheEngine
from operators.container.container import Container
from operators.assemble.decoders.common import SimpleLayersPrefillCacheProvider
from operators.assemble.decoders.real_decoders_group import DecodersGroup
from operators.core.lora.layers_lora_provider import SimpleLayersLoraHookProvider
from operators.core.lora.render import LoraRenderConfig
from operators.core.tokenizer.common import SequenceLengthData
from operators import flags
from . import flags as mod_flags
from operators.helpers.cuda_utils import tmp_stream
from ..common.brain import *


@dataclass
class SeqComplex:
    block_manager: BlockManager
    seq: SequenceLengthData
    user_model: Union[None, List[int], Tuple[LoraRenderConfig, List[int]]]
    occupied_blocks: List[int] = cast(List[int], field(default_factory=list))
    occupied_slots: List[int] = cast(List[int], field(default_factory=list))

    block_mapping: Dict[int, int] = cast(
        Dict[int, int], field(default_factory=dict))
    slot_mapping: Dict[int, int] = cast(
        Dict[int, int], field(default_factory=dict))

    def append_slot(self) -> int:
        current_slots = len(self.occupied_slots)
        num_blocks_req = (
            current_slots+1+(flags.BLOCK_SIZE-1))//flags.BLOCK_SIZE
        if num_blocks_req > len(self.occupied_blocks):
            b = self.block_manager.allocate_block()
            self.occupied_blocks.append(b)
            self.block_mapping[len(self.block_mapping)] = b
        b = self.occupied_blocks[-1]
        slot_offset = current_slots % flags.BLOCK_SIZE
        slot = b*flags.BLOCK_SIZE+slot_offset
        self.occupied_slots.append(slot)
        self.slot_mapping[len(self.slot_mapping)] = slot
        return slot

    @property
    def user_model_related_blocks(self) -> List[int]:
        if self.user_model is None:
            return []
        elif isinstance(self.user_model, tuple):
            return self.user_model[1]
        else:
            return self.user_model

    @property
    def user_model_mode(self) -> Optional[LoraRenderConfig]:
        if self.user_model is None:
            return None
        elif isinstance(self.user_model, tuple):
            return self.user_model[0]
        else:
            return None

    @property
    def ptuning_user_blocks(self) -> Optional[List[int]]:
        if self.user_model is None or isinstance(self.user_model, tuple):
            return None
        else:
            return self.user_model

    @property
    def current_slots_count(self) -> int:
        return len(self.occupied_slots)

    @property
    def current_blocks_count(self) -> int:
        return len(self.occupied_blocks)

    @property
    def generate_req_block(self) -> bool:
        current_slots = len(self.occupied_slots)
        num_blocks_req = (
            current_slots+1+(flags.BLOCK_SIZE-1))//flags.BLOCK_SIZE
        return num_blocks_req > len(self.occupied_blocks)

    @property
    def prefill_req_blocks(self) -> int:
        return (self.seq.prompt_len+flags.BLOCK_SIZE-1)//flags.BLOCK_SIZE


E = TypeVar('E')
CPU = torch.device("cpu")


def pad(mutli_lists: List[List[E]], default: E) -> List[List[E]]:
    pad_len = max([len(l) for l in mutli_lists])
    ls = [l+[default]*(pad_len-len(l))
          for l in mutli_lists]
    return ls


async def run_prefilling(
    operates: List[Tuple[OperateDoPrefill, SeqComplex]],
    container: Container,
    cache_engine: CacheEngine,
    decoders: DecodersGroup
):
    prompt_lens: List[int] = []
    seqeezed_generating_len: List[int] = []
    for (o, sc) in operates:
        actual_prompt_len = o.carrier.tensor.size(0)
        expected_prompt_len = sc.seq.prompt_len
        seqeezed_generating_len.append(actual_prompt_len-expected_prompt_len)
        prompt_lens.append(actual_prompt_len)

    input_lens = prompt_lens
    slot_mapping: List[int] = []
    ranges: List[Tuple[int, int]] = []
    offset = 0

    related_blocks: List[int] = []
    for prompt_len, (o, sc) in zip(prompt_lens, operates):
        for _ in range(prompt_len):
            slot = sc.append_slot()
            slot_mapping.append(slot)
            related_blocks.append(slot//flags.BLOCK_SIZE)
        related_blocks.extend(sc.user_model_related_blocks)
        ranges.append((offset, offset + prompt_len))
        offset += prompt_len

    related_blocks = list(set(related_blocks))

    input_positions = [
        i for prompt_len in prompt_lens for i in range(prompt_len)]
    input_hidden_states: List[torch.Tensor] = []
    for (o, sc) in operates:
        if o.cpu_in:
            input_hidden_states.append(await container.data_cpu_to_device(o.carrier.tensor))
        else:
            input_hidden_states.append(o.carrier.tensor)

    def gen_input(
        input_hidden_states: List[torch.Tensor],
        slot_mapping: List[int],
        input_positions: List[int],
        user_models: List[Union[None, List[int], Tuple[LoraRenderConfig, List[int]]]]
    ):
        cache_provider, translator = cache_engine.get_cache_provider(related_blocks)
        slot_mapping = [translator.translate_slot(s) for s in slot_mapping]
        ptunings = [
            [translator.translate_block(b) for b in p]
            if isinstance(p, list)
            else None
            for p in user_models
        ]
        loras = [
            (l[0], [translator.translate_block(b) for b in l[1]])
            if isinstance(l, tuple)
            else None
            for l in user_models
        ]
        hidden_states = torch.cat(input_hidden_states)
        slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int64)
        input_positions_tensor = torch.tensor(
            input_positions, dtype=torch.int32)
        ptuning_provider = SimpleLayersPrefillCacheProvider(
            seq_ptunings=ptunings, cache_provider=cache_provider
        )
        lora_provider = SimpleLayersLoraHookProvider(
            seq_loras=loras,
            seq_lens=input_lens,
            cache_provider=cache_provider
        )
        return hidden_states, slot_mapping_tensor, input_positions_tensor, ptuning_provider, lora_provider

    hidden_states, slot_mapping_tensor, input_positions_tensor, ptuning_provider, lora_provider = await container.step(
        "gen_input",
        gen_input,
        input_hidden_states=input_hidden_states,
        slot_mapping=slot_mapping,
        input_positions=input_positions,
        user_models=[sc.user_model for (_, sc) in operates]
    )

    hidden_states = await decoders.multi_seqs_prefill_forward(
        container=container,
        hidden_states=hidden_states, position_ids=input_positions_tensor, input_lens=input_lens,
        layers_cache_provider=ptuning_provider,
        slot_mapping=slot_mapping_tensor,
        layers_lora_hook_provider=lora_provider
    )
    outs: List[torch.Tensor] = []
    for i, (o, _) in enumerate(operates):
        t = hidden_states[ranges[i][0]:ranges[i][1]]
        if o.tail_in:
            t = t[-1]
        if o.cpu_out:
            outs.append(await container.data_device_to_cpu(t, wait=False))
        else:
            outs.append(t)

    read_only_blocks: List[int] = []
    for (_, sc) in operates:
        read_only_blocks += sc.occupied_blocks[:-1]
    cache_engine.set_blocks_readonly(read_only_blocks)

    sp = None
    if any(o.cpu_out for (o, _) in operates):
        sp = container.sync_point()
    for i, (o, sc) in enumerate(operates):
        squeeze_len = seqeezed_generating_len[i]
        sc.seq.output_count += 1+squeeze_len
        if o.cpu_out:
            o.set_result(DeviceTensorCarrier(outs[i], no_gc=o.carrier.no_gc), sp)
        else:
            o.set_result(DeviceTensorCarrier(outs[i], no_gc=o.carrier.no_gc))
        o.carrier.free()


def check_evil_fast(operates: List[Tuple[OperateDoGenerating, SeqComplex]]):
    start = -1
    curr = -1
    for (op, sc) in operates:
        t = op.carrier.tensor
        if not isinstance(op.carrier.tensor, tuple):
            return False
        _, _, _start, _num, _this = t
        if _num != len(operates):
            return False
        if start == -1:
            start = _start
            curr = _start
        elif start != _start:
            return False
        if _this != curr:
            return False
        else:
            curr += 1
    return True


_evil_fast_alloc = 0


async def run_generating(
    operates: List[Tuple[OperateDoGenerating, SeqComplex]],
    container: Container,
    cache_engine: CacheEngine,
    decoders: DecodersGroup
):
    prompt_lens = [sc.seq.prompt_len for (_, sc) in operates]
    generating_lens = [sc.seq.output_count for (_, sc) in operates]
    # latest generated token (this input token) has not already been in block cache
    cached_lens = [p+g-1 for (p, g) in zip(prompt_lens, generating_lens)]
    positions = cached_lens
    context_lens = [
        (len(sc.ptuning_user_blocks or [])*flags.BLOCK_SIZE)+position+1
        for position, (_, sc)
        in zip(positions, operates)
    ]  # when attention, this input token is already been cached
    slots = [sc.append_slot() for (_, sc) in operates]
    block_tables = [
        (sc.ptuning_user_blocks or []) + sc.occupied_blocks
        for (_, sc) in operates
    ]
    loras = [(sc.user_model if isinstance(sc.user_model, tuple) else None) for (_, sc) in operates]

    related_blocks = list(set([
        b
        for (_, sc) in operates
        for b in (sc.user_model_related_blocks or []) + sc.occupied_blocks
    ]))

    context_lens_max = max(context_lens)

    def gen_hidden_state(next_tokens: List[torch.Tensor]):
        return torch.stack(next_tokens).unsqueeze_(1)

    def gen_aux_input(context_lens: List[int], positions: List[int]):
        device = container.device
        context_lens_t = torch.tensor(context_lens, dtype=torch.int32, device=CPU).to(device, non_blocking=True)
        positions_t = torch.tensor(positions, dtype=torch.int32, device=CPU).to(device, non_blocking=True)
        return context_lens_t, positions_t

    is_evil_fast = check_evil_fast(operates)
    aux = None
    if is_evil_fast:
        hidden_states, aux = operates[0][0].carrier.tensor[0]
    else:
        input_hidden_states: List[torch.Tensor] = []
        for (o, _) in operates:
            if o.cpu_in:
                input_hidden_states.append(await container.data_cpu_to_device(o.carrier.tensor))
            else:
                t = o.carrier.tensor
                if isinstance(o.carrier.tensor, tuple):
                    t = t[1]()  # type: ignore
                input_hidden_states.append(t)
        hidden_states = await container.step(
            "gen_hidden_state", gen_hidden_state,
            next_tokens=input_hidden_states,
        )
    if aux is None:
        context_lens_t, positions_t = \
            await container.step(
                "gen_aux_input", gen_aux_input,
                context_lens=context_lens,
                positions=positions,
            )
        aux = (context_lens_t, positions_t)
    else:
        context_lens_t, positions_t = aux

    def gen_cache_input(
        block_tables: List[List[int]],
        slots: List[int],
        related_blocks: List[int],
        loras: List[Optional[Tuple[LoraRenderConfig, List[int]]]]
    ):
        device = container.device
        any_lora = any([l is not None for l in loras])
        if mod_flags.OUTSIDE_CACHE_INFO_MOVE and not any_lora:
            _tmp_stream = tmp_stream.create_or_reuse_tmp_stream("bypass-cache-info-load", device)
            _tmp_stream.wait_stream(container.stream)
            torch.cuda.set_stream(_tmp_stream)

        cache_provider, translator = \
            cache_engine.get_cache_provider(related_blocks)
        block_tables = [
            [translator.translate_block(b) for b in bs] for bs in block_tables
        ]
        slots = [translator.translate_slot(s) for s in slots]
        block_tables = pad(block_tables, 0)
        loras = [
            (l[0], [translator.translate_block(b) for b in l[1]])
            if l is not None else None
            for l in loras
        ]
        lora_provider = SimpleLayersLoraHookProvider(
            seq_loras=loras,
            seq_lens=[1]*len(slots),
            cache_provider=cache_provider
        )
        block_tables_t = torch.tensor(block_tables, dtype=torch.int32, device=CPU).to(device, non_blocking=True)
        slot_mapping_t = torch.tensor(slots, dtype=torch.int64, device=CPU).to(device, non_blocking=True)
        if mod_flags.OUTSIDE_CACHE_INFO_MOVE and not any_lora:
            cache_engine.record_stream(_tmp_stream)
            torch.cuda.set_stream(container.stream)
        return block_tables_t, slot_mapping_t, cache_provider, lora_provider

    block_tables_t, slot_mapping_t, cache_provider, lora_provider = \
        await container.step(
            "gen_cache_input", gen_cache_input,
            block_tables=block_tables,
            slots=slots,
            related_blocks=related_blocks,
            loras=loras
        )

    hidden_states = await decoders.multi_seqs_generating_forward(
        container=container,
        hidden_states=hidden_states,
        position_ids=positions_t,  # [8],device, int32
        layers_cache_provider=cache_provider,
        context_len_max=context_lens_max,  # 9
        context_lens=context_lens_t,  # [9],device, int32
        block_tables=block_tables_t,  # [[]], device, int32
        slot_mapping=slot_mapping_t,  # [], device int64
        layers_lora_hook_provider=lora_provider,
    )

    read_only_blocks: List[int] = []
    for (_, sc) in operates:
        read_only_blocks.extend(sc.occupied_blocks[-3:-1])
        sc.seq.output_count += 1
    cache_engine.set_blocks_readonly(read_only_blocks)

    # check evil fast
    if mod_flags.ALLOW_2WAY_DISPATCH and all((not op.cpu_out) for op, _ in operates):
        # can evil fast
        direct_result = (hidden_states, aux)
        start = _evil_fast_alloc
        num = len(operates)

        def make_evil_result(i: int):
            global _evil_fast_alloc

            def lazy():
                return hidden_states[i].view(-1)
            this = _evil_fast_alloc
            _evil_fast_alloc += 1
            return (direct_result if i == 0 else None), lazy, start, num, this
        for i, (o, _) in enumerate(operates):
            o.set_result(DeviceTensorCarrier(make_evil_result(i), no_gc=o.carrier.no_gc))  # type: ignore
            o.carrier.free()
    else:
        outs: List[torch.Tensor] = [
            await container.data_device_to_cpu(hidden_states[i].view(-1), wait=False)
            if o.cpu_out else
            hidden_states[i].view(-1)
            for i, (o, _) in enumerate(operates)
        ]
        sp = container.sync_point()
        for i, (o, sc) in enumerate(operates):
            o.set_result(DeviceTensorCarrier(outs[i], no_gc=o.carrier.no_gc), sp if o.cpu_out else None)
            o.carrier.free()


async def put_blocks(
    operate: OperatePutBlocks,
    seq_complex: SeqComplex,
    cache_engine: CacheEngine,
    container: Container,
):
    local_blocks = [seq_complex.block_mapping[b] for b in operate.seq_blocks]
    await container.step("put_blocks", cache_engine.put_blocks, target_blocks=local_blocks, kv_cache=operate.kv_cache)
    if len(local_blocks) > 0:
        # could be last block of prefill
        cache_engine.set_blocks_readonly(local_blocks[:-1])
    if not isinstance(cache_engine, ScalableCacheEngine):
        # no actual copy
        sp = container.sync_point()
        operate.set_result(None, sp)
    else:
        operate.set_result(None)


async def get_blocks(
    operate: OperateGetBlocks,
    seq_complex: SeqComplex,
    cache_engine: CacheEngine,
    container: Container,
):
    local_blocks = [seq_complex.block_mapping[b] for b in operate.seq_blocks]
    bs = await container.step("get_blocks", cache_engine.get_blocks, target_blocks=local_blocks)
    sp = container.sync_point()
    operate.set_result(bs, sp)
