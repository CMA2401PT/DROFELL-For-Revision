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
from . import head, tail
from operators.helpers.cuda_utils import tmp_stream
from ..common.brain import *
from .decode_graph import (
    DecodeGraphRunner,
    GraphKey,
    HeadDecodeGraphRunner,
    HiddenInput,
    TailDecodeGraphRunner,
)


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
    if any(op.recovery_decode for op, _ in operates):
        if not all(op.recovery_decode for op, _ in operates):
            raise RuntimeError(
                "regular prefill and recovery decode cannot share a batch")
        return await run_recovery_decode_chunks(
            operates=operates,
            container=container,
            cache_engine=cache_engine,
            decoders=decoders,
        )
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


async def run_recovery_decode_chunks(
    operates: List[Tuple[OperateDoPrefill, SeqComplex]],
    container: Container,
    cache_engine: CacheEngine,
    decoders: DecodersGroup,
) -> None:
    """Decode several consecutive known tokens per sequence in one eager call.

    Each token is represented as one paged-attention batch row. Rows belonging
    to the same logical sequence share a block table but use monotonically
    increasing context lengths, so each row can see the restored prefix and
    only its causal predecessors from this chunk.
    """

    chunk_lens = [op.carrier.shape[0] for op, _ in operates]
    if any(length <= 1 for length in chunk_lens):
        raise ValueError("recovery decode chunks must contain multiple tokens")
    if any(op.tail_in for op, _ in operates):
        raise ValueError("recovery decode chunks must preserve every output row")

    positions: List[int] = []
    context_lens: List[int] = []
    slots: List[int] = []
    ranges: List[Tuple[int, int]] = []
    block_tables: List[List[int]] = []
    token_loras: List[Optional[Tuple[LoraRenderConfig, List[int]]]] = []
    related_blocks: List[int] = []
    offset = 0

    for chunk_len, (_, sc) in zip(chunk_lens, operates):
        prefix_slots = sc.current_slots_count
        ptuning_blocks = sc.ptuning_user_blocks or []
        ptuning_slots = len(ptuning_blocks) * flags.BLOCK_SIZE
        chunk_slots = [sc.append_slot() for _ in range(chunk_len)]
        positions.extend(range(prefix_slots, prefix_slots + chunk_len))
        context_lens.extend(
            ptuning_slots + prefix_slots + token_i + 1
            for token_i in range(chunk_len)
        )
        slots.extend(chunk_slots)
        table = ptuning_blocks + sc.occupied_blocks
        block_tables.extend([table] * chunk_len)
        lora = sc.user_model if isinstance(sc.user_model, tuple) else None
        token_loras.extend([lora] * chunk_len)
        related_blocks.extend(sc.user_model_related_blocks)
        related_blocks.extend(sc.occupied_blocks)
        ranges.append((offset, offset + chunk_len))
        offset += chunk_len

    input_hidden_states: List[torch.Tensor] = []
    for op, _ in operates:
        hidden = op.carrier.tensor
        if op.cpu_in:
            hidden = await container.data_cpu_to_device(hidden)
        input_hidden_states.append(hidden)

    def prepare_inputs(
        input_hidden_states: List[torch.Tensor],
        block_tables: List[List[int]],
        slots: List[int],
        positions: List[int],
        context_lens: List[int],
        related_blocks: List[int],
        token_loras: List[Optional[Tuple[LoraRenderConfig, List[int]]]],
    ):
        cache_provider, translator = cache_engine.get_cache_provider(
            list(set(related_blocks))
        )
        translated_tables = [
            [translator.translate_block(block) for block in table]
            for table in block_tables
        ]
        translated_tables = pad(translated_tables, 0)
        translated_slots = [
            translator.translate_slot(slot) for slot in slots
        ]
        translated_loras = [
            (
                lora[0],
                [translator.translate_block(block) for block in lora[1]],
            )
            if lora is not None else None
            for lora in token_loras
        ]
        hidden_states = torch.cat(input_hidden_states).unsqueeze(1)
        device = container.device
        position_ids_dtype = \
            decoders.decoders[0].rotary_emb.position_ids_dtype
        positions_t = torch.tensor(
            positions, dtype=position_ids_dtype, device="cpu"
        ).to(device, non_blocking=True)
        context_lens_t = torch.tensor(
            context_lens, dtype=torch.int32, device="cpu"
        ).to(device, non_blocking=True)
        block_tables_t = torch.tensor(
            translated_tables, dtype=torch.int32, device="cpu"
        ).to(device, non_blocking=True)
        slot_mapping_t = torch.tensor(
            translated_slots, dtype=torch.int64, device="cpu"
        ).to(device, non_blocking=True)
        lora_provider = SimpleLayersLoraHookProvider(
            seq_loras=translated_loras,
            seq_lens=[1] * len(translated_slots),
            cache_provider=cache_provider,
        )
        return (
            hidden_states,
            positions_t,
            context_lens_t,
            block_tables_t,
            slot_mapping_t,
            cache_provider,
            lora_provider,
        )

    (
        hidden_states,
        positions_t,
        context_lens_t,
        block_tables_t,
        slot_mapping_t,
        cache_provider,
        lora_provider,
    ) = await container.step(
        "prepare_recovery_decode_chunk",
        prepare_inputs,
        input_hidden_states=input_hidden_states,
        block_tables=block_tables,
        slots=slots,
        positions=positions,
        context_lens=context_lens,
        related_blocks=related_blocks,
        token_loras=token_loras,
    )

    hidden_states = await decoders.multi_seqs_generating_forward(
        container=container,
        hidden_states=hidden_states,
        position_ids=positions_t,
        layers_cache_provider=cache_provider,
        context_len_max=max(context_lens),
        context_lens=context_lens_t,
        block_tables=block_tables_t,
        slot_mapping=slot_mapping_t,
        layers_lora_hook_provider=lora_provider,
    )
    hidden_states = hidden_states.squeeze(1)

    read_only_blocks: List[int] = []
    for _, sc in operates:
        read_only_blocks.extend(sc.occupied_blocks[:-1])
    cache_engine.set_blocks_readonly(list(set(read_only_blocks)))

    cpu_outputs: List[torch.Tensor] = []
    for (start, end), (op, _) in zip(ranges, operates):
        output = hidden_states[start:end]
        if op.cpu_out:
            output = await container.data_device_to_cpu(output, wait=False)
        cpu_outputs.append(output)
    sync_point = container.sync_point() \
        if any(op.cpu_out for op, _ in operates) else None

    for chunk_len, output, (op, sc) in zip(
        chunk_lens, cpu_outputs, operates
    ):
        sc.seq.output_count += chunk_len
        no_gc = op.carrier.no_gc
        op.set_result(
            DeviceTensorCarrier(output, no_gc),
            sync_point if op.cpu_out else None,
        )
        op.carrier.free()


def check_evil_fast(operates: List[Tuple[OperateDoGenerating, SeqComplex]]):
    start = -1
    curr = -1
    for (op, sc) in operates:
        if op.carrier is None:
            return False
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


async def _prepare_generating_inputs(
    operates: List[Tuple[OperateDoGenerating, SeqComplex]],
    container: Container,
    model_heads: Dict[str, ModelHead],
    head_decode_graph_runner: HeadDecodeGraphRunner,
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """Resolve carriers and execute embedded head stages.

    Head tasks are grouped by base-model binding. The second return value is a
    directly usable batched tensor for the common homogeneous all-head case;
    heterogeneous batches return ``None`` and are stacked by the caller.
    """
    input_hidden_states: List[Optional[torch.Tensor]] = [
        None for _ in operates
    ]
    head_groups: Dict[str, List[Tuple[int, DecodeHeadTask]]] = {}
    for i, (operate, _) in enumerate(operates):
        if operate.head_task is not None:
            head_groups.setdefault(operate.head_task.binding, []).append(
                (i, operate.head_task)
            )
            continue
        assert operate.carrier is not None
        if operate.cpu_in:
            input_hidden_states[i] = await container.data_cpu_to_device(
                operate.carrier.tensor
            )
        else:
            hidden_state = operate.carrier.tensor
            if isinstance(hidden_state, tuple):
                hidden_state = hidden_state[1]()  # type: ignore
            input_hidden_states[i] = hidden_state

    direct_head_hidden_states = None
    for head_binding, group in head_groups.items():
        tasks = [task for _, task in group]
        for task in tasks:
            if task.begin_metric is not None:
                task.begin_metric()
        head_graph_fallback = head_decode_graph_runner.fallback_reason(
            len(tasks)
        )
        if head_graph_fallback is None:
            grouped_hidden_states = await head_decode_graph_runner.run(
                key=(head_binding, len(tasks)),
                model_head=model_heads[head_binding],
                token_ids=[task.token for task in tasks],
                container=container,
            )
        else:
            head_decode_graph_runner.record_fallback(head_graph_fallback)
            grouped_hidden_states = await head.decode_tokens_to_hidden_states(
                model_head=model_heads[head_binding],
                container=container,
                tokens=[task.token for task in tasks],
            )
        if len(head_groups) == 1 and len(group) == len(operates):
            direct_head_hidden_states = grouped_hidden_states
        for group_index, (operate_index, task) in enumerate(group):
            input_hidden_states[operate_index] = grouped_hidden_states[
                group_index
            ].view(-1)
            if task.finish_metric is not None:
                task.finish_metric()

    assert all(hidden is not None for hidden in input_hidden_states)
    return cast(List[torch.Tensor], input_hidden_states), \
        direct_head_hidden_states


async def _run_embedded_decode_tails(
    operates: List[Tuple[OperateDoGenerating, SeqComplex]],
    hidden_states: torch.Tensor,
    container: Container,
    model_tails: Dict[str, ModelTail],
    tail_decode_graph_runner: TailDecodeGraphRunner,
) -> Dict[int, Tuple[int, Dict[int, float]]]:
    """Execute embedded tail stages and index results by parent operation."""
    tail_groups: Dict[str, List[Tuple[int, DecodeTailTask]]] = {}
    for i, (operate, _) in enumerate(operates):
        if operate.tail_task is not None:
            tail_groups.setdefault(operate.tail_task.binding, []).append(
                (i, operate.tail_task)
            )

    outputs_by_operate: Dict[int, Tuple[int, Dict[int, float]]] = {}
    for tail_binding, group in tail_groups.items():
        tasks = [task for _, task in group]
        grouped_hidden_states: HiddenInput
        if len(tail_groups) == 1 and len(group) == len(operates):
            grouped_hidden_states = hidden_states.squeeze(1)
        else:
            grouped_hidden_states = [
                hidden_states[i].view(-1) for i, _ in group
            ]
        use_greedy_fast_path = tail.uses_greedy_no_logprobs([
            task.sampling_params for task in tasks
        ])
        tail_graph_fallback = tail_decode_graph_runner.fallback_reason(
            batch_size=len(tasks),
            use_greedy_fast_path=use_greedy_fast_path,
        )
        if tail_graph_fallback is None:
            for task in tasks:
                if task.begin_metric is not None:
                    task.begin_metric()
            next_tokens = await tail_decode_graph_runner.run(
                key=(tail_binding, len(tasks)),
                model_tail=model_tails[tail_binding],
                hidden_states=grouped_hidden_states,
                container=container,
            )
            outputs = []
            for task, next_token in zip(tasks, next_tokens):
                task.seq_core.append_token_id(next_token)
                outputs.append((next_token, {}))
                if task.finish_metric is not None:
                    task.finish_metric()
        else:
            tail_decode_graph_runner.record_fallback(tail_graph_fallback)
            outputs = await tail.decode_hidden_states_to_sampling_outputs(
                model_tail=model_tails[tail_binding],
                container=container,
                hidden_states=grouped_hidden_states,
                tasks=tasks,
            )
        for (operate_index, _), output in zip(group, outputs):
            outputs_by_operate[operate_index] = output
    return outputs_by_operate


def _uniform_decode_graph_stages(
    operates: List[Tuple[OperateDoGenerating, SeqComplex]],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return uniform head/tail bindings or a structural fallback reason."""
    head_bindings = [
        op.head_task.binding if op.head_task is not None else None
        for op, _ in operates
    ]
    tail_bindings = [
        op.tail_task.binding if op.tail_task is not None else None
        for op, _ in operates
    ]
    if any(binding != head_bindings[0] for binding in head_bindings[1:]):
        return None, None, "mixed_head"
    if any(binding != tail_bindings[0] for binding in tail_bindings[1:]):
        return None, None, "mixed_tail"
    return head_bindings[0], tail_bindings[0], None


async def run_generating(
    operates: List[Tuple[OperateDoGenerating, SeqComplex]],
    container: Container,
    cache_engine: CacheEngine,
    decoders: DecodersGroup,
    decode_graph_runner: DecodeGraphRunner,
    decode_graph_binding: str,
    decoder_backend: str,
    model_heads: Dict[str, ModelHead],
    model_tails: Dict[str, ModelTail],
    head_decode_graph_runner: HeadDecodeGraphRunner,
    tail_decode_graph_runner: TailDecodeGraphRunner,
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
    has_fused_stages = any(
        op.head_task is not None or op.tail_task is not None
        for op, _ in operates
    )
    head_binding, tail_binding, graph_fallback = \
        _uniform_decode_graph_stages(operates)
    decoder_graph_fallback = decode_graph_runner.fallback_reason(
        batch_size=len(operates),
        context_len_max=context_lens_max,
        has_lora=any(lora is not None for lora in loras),
        decoder_backend=decoder_backend,
    )
    if graph_fallback is None:
        graph_fallback = decoder_graph_fallback
    if graph_fallback is None and tail_binding is not None:
        tail_tasks = [
            op.tail_task for op, _ in operates
            if op.tail_task is not None
        ]
        if not tail.uses_greedy_no_logprobs([
            task.sampling_params for task in tail_tasks
        ]):
            graph_fallback = "sampling"

    exact_graph_key = GraphKey(
        decoder_binding=decode_graph_binding,
        batch_size=len(operates),
        cache_engine_id=id(cache_engine),
        head_binding=head_binding,
        tail_binding=tail_binding,
    )
    if graph_fallback is None \
            and not decode_graph_runner.has_graph(exact_graph_key):
        graph_fallback = "variant_not_captured"
    use_exact_graph = graph_fallback is None
    if graph_fallback is not None:
        decode_graph_runner.record_fallback(graph_fallback)

    decoder_only_graph_key = GraphKey(
        decoder_binding=decode_graph_binding,
        batch_size=len(operates),
        cache_engine_id=id(cache_engine),
    )
    use_staged_decoder_graph = (
        not use_exact_graph
        and decoder_graph_fallback is None
        and decode_graph_runner.has_graph(decoder_only_graph_key)
    )
    use_decoder_graph = use_exact_graph or use_staged_decoder_graph

    def gen_hidden_state(next_tokens: List[torch.Tensor]):
        return torch.stack(next_tokens).unsqueeze_(1)

    def gen_aux_input(context_lens: List[int], positions: List[int]):
        device = container.device
        context_lens_t = torch.tensor(context_lens, dtype=torch.int32, device=CPU).to(device, non_blocking=True)
        position_ids_dtype = decoders.decoders[0].rotary_emb.position_ids_dtype
        positions_t = torch.tensor(positions, dtype=position_ids_dtype, device=CPU).to(device, non_blocking=True)
        return context_lens_t, positions_t

    is_evil_fast = not has_fused_stages and check_evil_fast(operates)
    aux = None
    graph_hidden_states: Optional[HiddenInput]
    graph_token_ids: Optional[List[int]] = None
    direct_head_hidden_states = None
    if use_exact_graph and head_binding is not None:
        head_tasks = [
            op.head_task for op, _ in operates
            if op.head_task is not None
        ]
        for task in head_tasks:
            if task.begin_metric is not None:
                task.begin_metric()
        graph_token_ids = [task.token for task in head_tasks]
        graph_hidden_states = None
    elif is_evil_fast:
        assert operates[0][0].carrier is not None
        hidden_states, aux = operates[0][0].carrier.tensor[0]
        graph_hidden_states = hidden_states
    else:
        concrete_input_hidden_states, direct_head_hidden_states = \
            await _prepare_generating_inputs(
                operates=operates,
                container=container,
                model_heads=model_heads,
                head_decode_graph_runner=head_decode_graph_runner,
            )
        graph_hidden_states = concrete_input_hidden_states
        if not use_decoder_graph:
            if direct_head_hidden_states is not None:
                hidden_states = direct_head_hidden_states.unsqueeze(1)
            else:
                hidden_states = await container.step(
                    "gen_hidden_state", gen_hidden_state,
                    next_tokens=concrete_input_hidden_states,
                )
    if not use_decoder_graph and aux is None:
        context_lens_t, positions_t = \
            await container.step(
                "gen_aux_input", gen_aux_input,
                context_lens=context_lens,
                positions=positions,
            )
        aux = (context_lens_t, positions_t)
    elif not use_decoder_graph:
        context_lens_t, positions_t = aux

    graph_result = None
    if use_decoder_graph:
        def prepare_graph_cache_input(
            block_tables: List[List[int]],
            slots: List[int],
            related_blocks: List[int],
        ):
            cache_provider, translator = cache_engine.get_cache_provider(
                related_blocks
            )
            translated_tables = [
                [translator.translate_block(block) for block in table]
                for table in block_tables
            ]
            translated_slots = [
                translator.translate_slot(slot) for slot in slots
            ]
            return translated_tables, translated_slots, cache_provider

        translated_tables, translated_slots, cache_provider = \
            await container.step(
                "prepare_decode_graph_input",
                prepare_graph_cache_input,
                block_tables=block_tables,
                slots=slots,
                related_blocks=related_blocks,
            )
        graph_tail_tasks = [
            op.tail_task for op, _ in operates
            if op.tail_task is not None
        ] if use_exact_graph and tail_binding is not None else []
        for task in graph_tail_tasks:
            if task.begin_metric is not None:
                task.begin_metric()
        graph_result = await decode_graph_runner.run(
            key=(exact_graph_key if use_exact_graph
                 else decoder_only_graph_key),
            container=container,
            decoders=decoders,
            cache_provider=cache_provider,
            hidden_states=graph_hidden_states,
            token_ids=graph_token_ids,
            block_tables=translated_tables,
            context_lens=context_lens,
            positions=positions,
            slots=translated_slots,
            model_head=(
                model_heads[head_binding]
                if use_exact_graph and head_binding is not None else None
            ),
            model_tail=(
                model_tails[tail_binding]
                if use_exact_graph and tail_binding is not None else None
            ),
        )
        hidden_states = graph_result.hidden_states
        if use_exact_graph and head_binding is not None:
            for task in head_tasks:
                if task.finish_metric is not None:
                    task.finish_metric()
    else:

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
            position_ids=positions_t,
            layers_cache_provider=cache_provider,
            context_len_max=context_lens_max,
            context_lens=context_lens_t,
            block_tables=block_tables_t,
            slot_mapping=slot_mapping_t,
            layers_lora_hook_provider=lora_provider,
        )

    read_only_blocks: List[int] = []
    for (_, sc) in operates:
        read_only_blocks.extend(sc.occupied_blocks[-3:-1])
        sc.seq.output_count += 1
    cache_engine.set_blocks_readonly(read_only_blocks)

    if use_exact_graph and tail_binding is not None:
        assert graph_result is not None
        if graph_result.token_ids is None:
            raise RuntimeError("tail-fused graph returned no token IDs")
        if len(graph_result.token_ids) != len(operates):
            raise RuntimeError("tail-fused graph returned wrong batch size")
        tail_outputs = {}
        for i, ((operate, _), next_token) in enumerate(zip(
            operates, graph_result.token_ids
        )):
            task = operate.tail_task
            assert task is not None
            task.seq_core.append_token_id(next_token)
            if task.finish_metric is not None:
                task.finish_metric()
            tail_outputs[i] = (next_token, {})
    else:
        tail_outputs = await _run_embedded_decode_tails(
            operates=operates,
            hidden_states=hidden_states,
            container=container,
            model_tails=model_tails,
            tail_decode_graph_runner=tail_decode_graph_runner,
        )

    # check evil fast
    if mod_flags.ALLOW_2WAY_DISPATCH \
            and not tail_outputs \
            and all((not op.cpu_out) for op, _ in operates):
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
            no_gc = None if o.carrier is None else o.carrier.no_gc
            o.set_result(DecodeGeneratingResult(
                carrier=DeviceTensorCarrier(make_evil_result(i), no_gc=no_gc)  # type: ignore
            ))
            if o.carrier is not None:
                o.carrier.free()
    else:
        outs: List[Optional[torch.Tensor]] = [None for _ in operates]
        for i, (o, _) in enumerate(operates):
            if i in tail_outputs:
                continue
            outs[i] = (
                await container.data_device_to_cpu(
                    hidden_states[i].view(-1), wait=False
                )
                if o.cpu_out else hidden_states[i].view(-1)
            )
        sp = container.sync_point() if any(
            o.cpu_out and i not in tail_outputs
            for i, (o, _) in enumerate(operates)
        ) else None
        for i, (o, sc) in enumerate(operates):
            if i in tail_outputs:
                result = DecodeGeneratingResult(
                    sampling_output=tail_outputs[i]
                )
            else:
                assert outs[i] is not None
                no_gc = None if o.carrier is None else o.carrier.no_gc
                result = DecodeGeneratingResult(
                    carrier=DeviceTensorCarrier(outs[i], no_gc=no_gc)
                )
            o.set_result(
                result,
                sp if o.cpu_out and i not in tail_outputs else None,
            )
            if o.carrier is not None:
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
