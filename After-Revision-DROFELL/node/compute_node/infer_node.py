import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union, cast

from dealers.common.base_types import DeviceTensorCarrier
from dealers.common.brain import UserModelHandle
from dealers.common.provider import DecodersInferBindingShim, DecodersShim, HeadInferShim, Provider, TailInferBindingShim
from operators import flags
import extend_asyncio as asyncio
from operators.core.lora.render import LoraRenderConfig
from .infer_meta import InferTask, UserModelDataName, InferInit, PutBlocks


@dataclass(frozen=True)
class _DecodeFusionPlan:
    """A contiguous decode segment delegated to one decoder shim."""

    decoder: DecodersInferBindingShim
    head: Optional[HeadInferShim] = None
    tail: Optional[TailInferBindingShim] = None

    @property
    def span(self) -> int:
        return 1 + int(self.head is not None) + int(self.tail is not None)


class Node:
    def __init__(
        self,
        block_retrieve: bool,
        provider: Provider,
        put_resources: Callable[[str, PutBlocks], Coroutine[Any, Any, None]],
        # infer_task_dispatch: Callable[
        #     [InferTask, Callable[[], None]],
        #     Coroutine[Any, Any, None]
        # ],
        # infer_task_discard: Callable[
        #     [InferTask, Callable[[], None]],
        #     Coroutine[Any, Any, None]
        # ],
    ) -> None:
        self._put_resources = put_resources
        # self._infer_task_dispatch = infer_task_dispatch
        # self._infer_task_discard = infer_task_discard
        self.provider = provider
        self._remove_barrier: Callable[[str], None] = self.provider.brain.remove_barrier
        self._add_barrier: Callable[[str], None] = self.provider.brain.add_barrier

        self._caches: Dict[str, Any] = {}
        self.user_models: List[str] = []
        self._user_models_datas: List[Tuple[str, str]] = []
        self._initing_caches: Dict[str, asyncio.Future[None]] = {}
        self._block_retrieve = block_retrieve
        self.dropped: bool = False
        self._retrieve_postpone = 8
        self._removed = False

    # def drop(self):
    #     # disable communication and stop all task asap for some simulation,
    #     # as on real world this will cause fully datch
    #     # but it is also possible that packet will come after drop becase of latency
    #     self._dropped = True

    #     async def stub_req_resource(name: str) -> Any:
    #         return None

    #     async def stub_put_resource(name: str, res: Any) -> Any:
    #         return None

    #     async def stub_infer_task_dispatch(task: InferTask) -> None:
    #         return None
    #     self._req_resources = stub_req_resource
    #     self._put_resources = stub_put_resource
    #     self.infer_task_dispatch = stub_infer_task_dispatch

    async def _get_user_model(
        self,
        user_model_name: str,
        blocks: List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]],
        cfg: Optional[LoraRenderConfig]
    ):
        if user_model_name in self._caches:
            return self._caches[user_model_name]
        unique_name, base = UserModelDataName.split_full_name(user_model_name)
        decoders_name = base.__str__()
        decoder = cast(DecodersShim, self.provider[decoders_name])
        handle = await decoder.put_user_model(unique_name, blocks, cfg)
        self._user_models_datas.append((decoders_name, unique_name))
        self._caches[user_model_name] = handle
        return handle

    # async def _res_expand(self, res_name: str, res: Any) -> Tuple[str, Any]:
    #     if UserModelDataName.is_full_res_name(res_name):
    #         _, base = UserModelDataName.split_full_name(res_name)
    #         blocks, cfg = res
    #         decoders_name = base.__str__()
    #         decoder = cast(DecodersShim, self._provider[decoders_name])
    #         handle = await decoder.put_user_model(blocks, cfg)
    #         return res_name, handle
    #     return res_name, res

    # async def _req_resources_with_cache(self, res_name: str) -> Any:
    #     res = self._caches.get(res_name)
    #     if res is not None:
    #         return res
    #     res = await self._req_resources(res_name)
    #     res_name, res = await self._res_expand(res_name, res)
    #     self._caches[res_name] = res
    #     return res

    def _warp_on_block_change(self, binding: DecodersInferBindingShim, put_blocks_key: str, dst: str, _last_ready_blocks: int):
        # print(f"{dst} -> last ready blocks: {_last_ready_blocks}")
        last_ready_blocks = _last_ready_blocks
        prefill_slots = -1

        def on_block_change(shim: DecodersInferBindingShim):
            nonlocal last_ready_blocks, prefill_slots
            if shim.is_first_generating:
                prefill_slots = shim.slots_count
            if (shim.slots_count-self._retrieve_postpone) % flags.BLOCK_SIZE != 0 or \
                    (shim.slots_count-self._retrieve_postpone) <= prefill_slots:
                return
            ready_blocks = shim.ready_blocks
            target_blocks = [b_i for b_i in range(last_ready_blocks, ready_blocks)]
            if len(target_blocks) == 0:
                return

            async def retrieve(target_blocks: List[int]):
                nonlocal last_ready_blocks
                layers_blocks = await shim.get_blocks(target_blocks)
                # layers_blocks_carrier = [
                #     [
                #         (DeviceTensorCarrier(k, None), DeviceTensorCarrier(v, None))
                #         for (k, v) in blocks
                #     ]
                #     for blocks in layers_blocks
                # ]
                last_ready_blocks = ready_blocks
                await self._put_resources(put_blocks_key, PutBlocks(dst, target_blocks, layers_blocks))
            asyncio.create_task(retrieve(target_blocks))
        binding.on_before_generating = on_block_change

    async def infer_init(self, config: InferInit) -> Union[None, List[Tuple[int, DeviceTensorCarrier]], DeviceTensorCarrier]:
        self.user_models = config.user_models
        # mark initing state:

        def mark_init_state(name: str):
            assert not name in self._initing_caches, name
            assert not name in self._caches, name
            self._initing_caches[name] = asyncio.Future()

        def finish_init(name: str, res: Any):
            assert name in self._initing_caches, name
            self._initing_caches[name].set_result(None)
            del self._initing_caches[name]
            self._caches[name] = res
        if config.head is not None:
            mark_init_state(config.head.dst)
        if config.tail is not None:
            mark_init_state(config.tail.dst)
        for _cfg in config.bindings:
            mark_init_state(_cfg.dst)

        if config.head is not None:
            _cfg = config.head
            adjacent_decoder = (
                config.bindings[0].src if config.bindings else None
            )
            _infer_head = await self.provider.get_model_head_dealer(
                _cfg.src,
                adjacent_decoder=adjacent_decoder,
            ).create_head_infer_binding(
                init_step=_cfg.init_step
            )
            finish_init(config.head.dst, _infer_head)
        if config.tail is not None:
            _cfg = config.tail
            adjacent_decoder = (
                config.bindings[-1].src if config.bindings else None
            )
            _infer_tail = await self.provider.get_model_tail_dealer(
                _cfg.src,
                adjacent_decoder=adjacent_decoder,
            ).create_infer_binding(
                seq_id=_cfg.seq_id, init_step=_cfg.step, seq_core=_cfg.seq_core, sampling_params=_cfg.sampling_params
            )
            finish_init(_cfg.dst, _infer_tail)
        for _cfg in config.bindings:
            _decoders_group = cast(DecodersShim, self.provider[_cfg.src])
            handle = None
            if _cfg.user_model is not None:
                name, blocks, ucfg = _cfg.user_model
                handle = cast(UserModelHandle, await self._get_user_model(name, blocks, ucfg))
            _infer_decoders_binding = await _decoders_group.create_infer_binding(
                seq_id=_cfg.seq_id, init_step=_cfg.init_step,
                seq_length_data=_cfg.length_info.clone(),
                user_model_handle=handle,
                allocate_slots=_cfg.allocate_slots,
                name=_cfg.dst,
            )
            if _cfg.init_blocks is not None:
                await _infer_decoders_binding.put_blocks(
                    seq_blocks=[i for i in range(_cfg.received_blocks)],
                    kv_cache=_cfg.init_blocks,
                )
            finish_init(_cfg.dst, _infer_decoders_binding)
            last_ready_blocks = _cfg.received_blocks
            put_blocks_key = _cfg.dst+"-blocks"
            dst = _cfg.dst
            if self._block_retrieve:
                self._warp_on_block_change(_infer_decoders_binding, put_blocks_key, dst, last_ready_blocks)

    # async def on_infer_task(self, infer_task: InferTask):
    #     context_id = infer_task.unique_id_str
    #     self._add_barrier(context_id)
    #     updated_infer_task = await self.infer(infer_task)
    #     if updated_infer_task is None or self.dropped:
    #         await self._infer_task_discard(
    #             infer_task,  # infer task is in unknown state
    #             lambda: self._remove_barrier(context_id)
    #         )
    #     else:
    #         await self._infer_task_dispatch(
    #             updated_infer_task,
    #             lambda: self._remove_barrier(context_id)
    #         )

    def _on_drop(self, infer_task: InferTask):
        # print(f"drop: {infer_task}")
        assert infer_task.is_cpu

    def _decode_plan_from_head(
        self,
        infer_task: InferTask,
        head: HeadInferShim,
    ) -> Optional[_DecodeFusionPlan]:
        """Plan head+decoder(+tail) without creating or enqueueing work.

        Only ready, contiguous shims owned by the same Brain can be delegated
        together. A target still being initialized follows the standalone path
        for this round; decoder groups are never merged with one another.
        """
        if infer_task.is_prefill or infer_task.navi.is_last:
            return None
        next_target, next_is_head, next_is_tail = infer_task.navi.target(1)
        next_res = self._caches.get(next_target)
        if next_is_head or next_is_tail \
                or not isinstance(next_res, DecodersInferBindingShim) \
                or next_res.brain is not head.brain:
            return None
        tail = None
        if infer_task.navi.current + 2 < len(infer_task.navi.user_models):
            tail_target, _, tail_is_tail = infer_task.navi.target(2)
            tail_res = self._caches.get(tail_target)
            if tail_is_tail \
                    and isinstance(tail_res, TailInferBindingShim) \
                    and tail_res.brain is next_res.brain:
                tail = tail_res
        return _DecodeFusionPlan(decoder=next_res, head=head, tail=tail)

    def _decode_tail_after_decoder(
        self,
        infer_task: InferTask,
        decoder: DecodersInferBindingShim,
    ) -> Optional[TailInferBindingShim]:
        """Return the ready same-Brain tail immediately after decoder."""
        if infer_task.is_prefill or infer_task.navi.is_last:
            return None
        next_target, _, next_is_tail = infer_task.navi.target(1)
        next_res = self._caches.get(next_target)
        if next_is_tail \
                and isinstance(next_res, TailInferBindingShim) \
                and next_res.brain is decoder.brain:
            return next_res
        return None

    def _should_leave_after(self, infer_task: InferTask, span: int) -> bool:
        """Whether the carrier after a local contiguous segment must leave."""
        end = infer_task.navi.current + span - 1
        if end == len(infer_task.navi.user_models) - 1:
            return True
        next_target, _, _ = infer_task.navi.target(span)
        return next_target not in self._caches \
            and next_target not in self._initing_caches

    async def infer(
        self,
        infer_task: InferTask,
        add_barrier: Callable[[], None],
        remove_barrier: Callable[[], None],
        metrics: Optional[List[Tuple[str, int]]] = None,
    ):
        input = infer_task.input
        _inp = None
        step = infer_task.step
        _last_ptr = infer_task.navi.current-1
        context_id = infer_task.unique_id_str

        def metric(name: str):
            if metrics is not None:
                metrics.append((name, time.time_ns()))

        def metric_pair(name: str):
            if metrics is None:
                return None, None

            def _start():
                assert metrics is not None
                metrics.append(("exec_"+name, time.time_ns()))

            def _end():
                assert metrics is not None
                metrics.append(("done_"+name, time.time_ns()))
            return _start, _end

        async def wait_decode_shim(name: str, shim: Any) -> bool:
            """Wait for one decode shim without transferring task ownership."""
            metric("wait_"+name)
            return await shim.can_continue_or_reoder(
                step,
                False,
                on_reorder_start=remove_barrier,
                on_reorder_end=add_barrier,
            )

        while True:
            assert infer_task.navi.current != _last_ptr
            _last_ptr = infer_task.navi.current
            target, is_head, is_tail = infer_task.navi.target(0)
            # print(f"{infer_task.step} is_prefill={infer_task.is_prefill} ->{target}")
            is_last = infer_task.navi.is_last
            should_leave = is_tail or is_last
            next_is_tail = False
            if not is_last:
                next_target, _, next_is_tail = infer_task.navi.target(1)
                should_leave = should_leave or ((next_target not in self._caches) and (next_target not in self._initing_caches))
            if target in self._initing_caches:
                remove_barrier()
                fut = self._initing_caches[target]
                await fut
                add_barrier()
            if self.dropped:
                return
            res = self._caches[target]
            if is_head:
                _h = cast(HeadInferShim, res)
                if infer_task.is_recovery_decode:
                    tokens = cast(List[int], input)
                    if len(tokens) != infer_task.recovery_decode_count:
                        raise ValueError(
                            "recovery head token count does not match chunk")
                    metric("wait_head")
                    can_continue = await _h.can_continue_or_reoder(
                        step,
                        False,
                        on_reorder_start=remove_barrier,
                        on_reorder_end=add_barrier,
                    )
                    if not can_continue:
                        infer_task.input = tokens
                        self._on_drop(infer_task)
                        return
                    if self.dropped:
                        return
                    _s, _e = metric_pair("head")
                    _inp = await _h.recovery_decode(
                        context_id=context_id,
                        step=step,
                        tokens=tokens,
                        cpu_out=should_leave,
                        on_sync_begin=remove_barrier,
                        on_sync_done=add_barrier,
                        begin_metric=_s,
                        finish_metric=_e,
                    )
                    metric("async_head")
                    infer_task.navi.next()
                    if should_leave:
                        infer_task.input = _inp
                        return infer_task
                    continue
                fusion = self._decode_plan_from_head(infer_task, _h)
                if fusion is not None:
                    raw_token = input
                    if isinstance(raw_token, list):
                        assert len(raw_token) == 1
                        token = raw_token[0]
                    else:
                        token = cast(int, raw_token)
                    can_continue = await wait_decode_shim("head", _h)
                    if can_continue:
                        can_continue = await wait_decode_shim(
                            "decoder", fusion.decoder
                        )
                    if can_continue and fusion.tail is not None:
                        can_continue = await wait_decode_shim(
                            "tail", fusion.tail
                        )
                    if not can_continue:
                        infer_task.input = token
                        self._on_drop(infer_task)
                        return
                    if self.dropped:
                        return
                    fused_should_leave = fusion.tail is not None or \
                        self._should_leave_after(infer_task, fusion.span)
                    head_metric = metric_pair("head")
                    decoder_metric = metric_pair("decoder")
                    tail_metric = metric_pair("tail") \
                        if fusion.tail is not None else None
                    result = await fusion.decoder.generating(
                        context_id=context_id,
                        step=step,
                        carrier=None,
                        cpu_in=False,
                        cpu_out=fused_should_leave and fusion.tail is None,
                        head=_h,
                        head_token=token,
                        tail=fusion.tail,
                        head_metric=head_metric,
                        tail_metric=tail_metric,
                        on_sync_begin=remove_barrier,
                        on_sync_done=add_barrier,
                        begin_metric=decoder_metric[0],
                        finish_metric=decoder_metric[1],
                    )
                    metric("async_head")
                    metric("async_decoder")
                    for _ in range(fusion.span):
                        infer_task.navi.next()
                    if fusion.tail is not None:
                        metric("async_tail")
                        assert result.sampling_output is not None
                        del infer_task.input
                        infer_task.output = result.sampling_output
                        return infer_task
                    assert result.carrier is not None
                    _inp = result.carrier
                    if fused_should_leave:
                        infer_task.input = _inp
                        return infer_task
                    continue
                if infer_task.is_prefill:
                    _inp = cast(List[int], input)
                else:
                    _inp = [cast(int, input)]
                metric("wait_head")
                can_continue = await _h.can_continue_or_reoder(infer_task.step, infer_task.is_prefill, on_reorder_start=remove_barrier, on_reorder_end=add_barrier)
                if not can_continue:
                    infer_task.input = _inp
                    self._on_drop(infer_task)
                    return
                if self.dropped:
                    return
                _s, _e = metric_pair("head")
                _inp = await _h.infer_tensor_to_embedding_tokens(
                    context_id=context_id, step=infer_task.step,
                    tokens=_inp, is_prefill=infer_task.is_prefill, cpu_out=should_leave,
                    on_sync_begin=remove_barrier, on_sync_done=add_barrier,
                    begin_metric=_s, finish_metric=_e
                )
                metric("async_head")
                infer_task.navi.next()
                if should_leave:
                    infer_task.input = _inp
                    return infer_task
            elif is_tail:
                _t = cast(TailInferBindingShim, res)
                _cpu_in = False
                if _inp is None:
                    _cpu_in = True
                    _inp = cast(DeviceTensorCarrier, input)
                if infer_task.is_recovery_replay:
                    metric("wait_tail")
                    can_continue = await _t.can_continue_or_reoder(
                        step,
                        infer_task.is_prefill,
                        on_reorder_start=remove_barrier,
                        on_reorder_end=add_barrier,
                    )
                    if not can_continue:
                        infer_task.input = _inp
                        self._on_drop(infer_task)
                        return
                    if self.dropped:
                        return
                    output_token_ids = infer_task.recovery_output_token_ids
                    if output_token_ids is None:
                        raise ValueError(
                            "recovery tail requires confirmed output tokens")
                    _t.recovery_replay(
                        step=step,
                        carrier=_inp,
                        output_token_ids=output_token_ids,
                        is_prefill=infer_task.is_prefill,
                    )
                    metric("async_tail")
                    infer_task.navi.next()
                    # Tail recovery consumes and frees the hidden-state
                    # carrier.  Do not leave that empty carrier attached to
                    # the completed task: the task is forwarded to the Agent
                    # for shadow-state bookkeeping and must remain serializable.
                    del infer_task.input
                    return infer_task
                metric("wait_tail")
                can_continue = await _t.can_continue_or_reoder(infer_task.step, infer_task.is_prefill, on_reorder_start=remove_barrier, on_reorder_end=add_barrier)
                if not can_continue:
                    infer_task.input = _inp
                    self._on_drop(infer_task)
                    return
                if self.dropped:
                    return
                _s, _e = metric_pair("tail")
                token, logprobs = await _t.sampling(
                    context_id=context_id, step=step,
                    carrier=_inp, is_prefill=infer_task.is_prefill, cpu_in=_cpu_in,
                    on_sync_begin=remove_barrier, on_sync_done=add_barrier,
                    begin_metric=_s, finish_metric=_e
                )
                metric("async_tail")
                infer_task.navi.next()
                del infer_task.input
                infer_task.output = (token, logprobs)
                return infer_task
            else:
                _b = cast(DecodersInferBindingShim, res)
                _cpu_in = False
                if _inp is None:
                    _cpu_in = True
                    _inp = cast(DeviceTensorCarrier, input)
                if infer_task.is_recovery_decode:
                    if _inp.shape[0] != infer_task.recovery_decode_count:
                        raise ValueError(
                            "recovery decoder rows do not match chunk")
                    metric("wait_decoder")
                    can_continue = await _b.can_continue_or_reoder(
                        step,
                        False,
                        on_reorder_start=remove_barrier,
                        on_reorder_end=add_barrier,
                    )
                    if not can_continue:
                        infer_task.input = _inp
                        self._on_drop(infer_task)
                        return
                    if self.dropped:
                        return
                    _s, _e = metric_pair("decoder")
                    _inp = await _b.recovery_decode(
                        context_id=context_id,
                        step=step,
                        carrier=_inp,
                        cpu_in=_cpu_in,
                        cpu_out=should_leave,
                        on_sync_begin=remove_barrier,
                        on_sync_done=add_barrier,
                        begin_metric=_s,
                        finish_metric=_e,
                    )
                    metric("async_decoder")
                    infer_task.navi.next()
                    if should_leave:
                        infer_task.input = _inp
                        return infer_task
                    continue
                metric("wait_decoder")
                can_continue = await _b.can_continue_or_reoder(infer_task.step, infer_task.is_prefill, on_reorder_start=remove_barrier, on_reorder_end=add_barrier)
                if not can_continue:
                    infer_task.input = _inp
                    self._on_drop(infer_task)
                    return
                if self.dropped:
                    return
                # print(f"run {infer_task.step} is_prefill={infer_task.is_prefill} ->{target}")
                _s, _e = metric_pair("decoder")
                if infer_task.is_prefill:
                    _inp = await _b.prefill(
                        context_id=context_id, step=step,
                        carrier=_inp, cpu_in=_cpu_in, cpu_out=should_leave, tail_in=next_is_tail,
                        on_sync_begin=remove_barrier, on_sync_done=add_barrier,
                        begin_metric=_s, finish_metric=_e
                    )
                else:
                    fused_tail = self._decode_tail_after_decoder(
                        infer_task, _b
                    )
                    if fused_tail is not None:
                        can_continue = await wait_decode_shim(
                            "tail", fused_tail
                        )
                        if not can_continue:
                            infer_task.input = _inp
                            self._on_drop(infer_task)
                            return
                        if self.dropped:
                            return
                    tail_metric = metric_pair("tail") \
                        if fused_tail is not None else None
                    result = await _b.generating(
                        context_id=context_id, step=step,
                        carrier=_inp, cpu_in=_cpu_in,
                        cpu_out=should_leave and fused_tail is None,
                        tail=fused_tail,
                        tail_metric=tail_metric,
                        on_sync_begin=remove_barrier, on_sync_done=add_barrier,
                        begin_metric=_s, finish_metric=_e
                    )
                    if fused_tail is not None:
                        metric("async_decoder")
                        metric("async_tail")
                        infer_task.navi.next()
                        infer_task.navi.next()
                        assert result.sampling_output is not None
                        del infer_task.input
                        infer_task.output = result.sampling_output
                        return infer_task
                    assert result.carrier is not None
                    _inp = result.carrier
                # print(f"done {infer_task.step} is_prefill={infer_task.is_prefill} ->{target}")
                metric("async_decoder")
                infer_task.navi.next()
                if should_leave:
                    infer_task.input = _inp
                    return infer_task

    def remove(self):
        if self._removed:
            return
        self.dropped = True
        for decoders_name, unique_name in self._user_models_datas:
            decoder = cast(DecodersShim, self.provider[decoders_name])
            decoder.remove_user_model(unique_name)
        self._user_models_datas = []
        for c in self._caches:
            self._caches[c] = None
        self._caches = {}
        self._removed = True

    def __del__(self):
        assert self._removed
