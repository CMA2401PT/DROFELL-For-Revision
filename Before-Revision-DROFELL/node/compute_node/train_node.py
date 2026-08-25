from collections import defaultdict
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, cast

import torch
from dealers.common.base_types import DeviceTensorCarrier
from dealers.common.provider import DecodersShim, HeadDealerShim, HeadTrainShim, Provider, TailDealerShim
from dealers.common.base_types import DeviceTensorCarrier
from dealers.common.brain import TrainForwardMeta
from dealers.common.provider import DecodersShim, DecodersTrainBindingShim, HeadDealerShim, TailDealerShim, TailTrainBindingShim
import extend_asyncio as asyncio
from .train_meta import TrainTask, ModelCKPT, TrainInit, TrainHeadInput, TrainTailInput


class Node:
    def __init__(
        self,
        ckpt_period: int,
        provider: Provider,
        put_resources: Callable[[str, ModelCKPT], Coroutine[Any, Any, None]],
        # train_task_dispatch: Callable[
        #     [TrainTask, Callable[[], None]],
        #     Coroutine[Any, Any, None]
        # ],
        # train_task_discard: Callable[
        #     [TrainTask, Callable[[], None]],
        #     Coroutine[Any, Any, None]
        # ]
    ) -> None:
        self._put_resources = put_resources
        # self._train_task_dispatch = train_task_dispatch
        # self._train_task_discard = train_task_discard
        self.provider = provider
        self.user_models: List[str] = []
        self._forward_sequence_len: int = 0
        self._backward_sequence_len: Optional[int] = None
        self._remove_barrier: Callable[[str], None] = self.provider.brain.remove_barrier
        self._add_barrier: Callable[[str], None] = self.provider.brain.add_barrier

        self._caches: Dict[str, Any] = {}
        self._initing_caches: Dict[str, asyncio.Future[None]] = {}
        self._ckpt_period = ckpt_period
        self._pending_updates: Dict[int, List[Tuple[str, ModelCKPT]]] = defaultdict(list)
        self.dropped: bool = False
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

    #     async def stub_train_task_dispatch(task: TrainTask) -> None:
    #         return None
    #     self._req_resources = stub_req_resource
    #     self._put_resources = stub_put_resource
    #     self.train_task_dispatch = stub_train_task_dispatch

    def model_sequence_len(self, is_backward: bool):
        if not is_backward:
            return self._forward_sequence_len
        if is_backward:
            assert self._backward_sequence_len is not None
            return self._backward_sequence_len

    def _wrap_on_should_update(self, put_ckpt_key: str, dst: str):
        def on_should_update(shim: DecodersTrainBindingShim):
            async def _do_update():
                await shim.update()
                # print(f"{dst} version comes to {shim.version}")
                if self._ckpt_period <= 0:
                    return
                if shim.version % self._ckpt_period != 0:
                    return
                # = req_version of not handled input
                current_version = shim.version
                states = await shim.dump_train_states()
                ckpt = ModelCKPT(dst, current_version, states)
                # reorder and make sure the ckpt is sent later than task,
                # thus keep send state is also recovered
                self._pending_updates[current_version].append((put_ckpt_key, ckpt))
                # await self.put_resources(put_ckpt_key, ckpt)
            asyncio.create_task(_do_update())
        return on_should_update

    def _wrap_on_head_tail_update_complete(self, put_ckpt_key: str, dst: str):
        def on_should_update(current_version: int):
            async def _do_update():
                ckpt = ModelCKPT(dst, current_version, {})
                self._pending_updates[current_version].append((put_ckpt_key, ckpt))
                # await self.put_resources(put_ckpt_key, ckpt)
            asyncio.create_task(_do_update())
        return on_should_update

    async def train_init(self, config: TrainInit):
        self.user_models = config.user_models
        self._forward_sequence_len = config.num_steps(is_backward=False)
        if config.tail is None:
            self._backward_sequence_len = config.num_steps(is_backward=True)

        print(f"initing {config.model_versions}")
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
            _head_shim = cast(HeadDealerShim, self.provider[config.head.src])
            _put_ckpt_key, _dst = _cfg.dst+"-ckpt", _cfg.dst
            _on_update_complete = self._wrap_on_head_tail_update_complete(_put_ckpt_key, _dst)
            finish_init(
                name=_cfg.dst,
                res=await _head_shim.create_head_train_binding(
                    init_version=config.head.version,
                    on_update_complete=_on_update_complete
                )
            )
        if config.tail is not None:
            _cfg = config.tail
            _tail_shim = cast(TailDealerShim, self.provider[_cfg.src])
            _put_ckpt_key, _dst = _cfg.dst+"-ckpt", _cfg.dst
            _on_update_complete = self._wrap_on_head_tail_update_complete(_put_ckpt_key, _dst)
            finish_init(
                name=_cfg.dst,
                res=await _tail_shim.create_train_binding(
                    init_version=_cfg.version,
                    impl=_cfg.impl,
                    on_update_complete=_on_update_complete
                )
            )
        for _cfg in config.bindings:
            _decoders_group = cast(DecodersShim, self.provider[_cfg.src])
            _put_ckpt_key = _cfg.dst+"-ckpt"
            _dst = _cfg.dst
            on_should_update = self._wrap_on_should_update(_put_ckpt_key, _dst)
            _train_decoders_binding = await _decoders_group.create_train_binding(
                version=_cfg.version,
                user_model_config=_cfg.user_model_config.clone(),
                optimizer_config=_cfg.optimizer_config,
                scheduler_config=_cfg.scheduler_config,
                data_init_method=_cfg.data_init_method,
                on_should_update=on_should_update,
                name=_dst,
            )
            if _cfg.recover_state_dict is not None:
                await _train_decoders_binding.load_train_states(_cfg.recover_state_dict)
            finish_init(_cfg.dst, _train_decoders_binding)
        print(f"init complete: {config.model_versions}")

    # async def on_train_task(self, train_task: TrainTask):
    #     self._add_barrier(train_task.minibatch_id)
    #     updated_train_task = await self.train(train_task)
    #     if updated_train_task is None or self.dropped:
    #         # task dropped
    #         await self._train_task_discard(
    #             train_task,  # infer task is in unknown state
    #             lambda: self._remove_barrier(train_task.minibatch_id),
    #         )
    #         return
    #     await self._train_task_dispatch(
    #         updated_train_task,
    #         lambda: self._remove_barrier(train_task.minibatch_id),
    #     )

    async def try_send_model_update(self, version: int):
        # make sure state is send later than previous tasks!
        current_sent_version = version
        minimal_version = sorted(self._pending_updates.keys())
        if len(minimal_version) > 0 and minimal_version[0] <= current_sent_version:
            version = minimal_version[0]
            for key, ckpt in self._pending_updates[version]:
                await self._put_resources(key, ckpt)
            del self._pending_updates[version]

    def _on_drop(self, train_task: TrainTask):
        assert train_task.is_cpu
        # print(f"task: {train_task} drop on compute node!")

    async def train(
        self,
        train_task: TrainTask,
        add_barrier: Callable[[], None],
        remove_barrier: Callable[[], None],
        metrics: Optional[List[Tuple[str, int]]] = None,
    ):
        input = train_task.input
        _inp = None
        req_version = train_task.request_version
        context_id = train_task.unique_id_str
        full_batch = train_task.batch_ids
        # print(f"compute node receive train task:\n\t {train_task}")

        _last_ptr = train_task.navi.current-1

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

        while True:
            assert train_task.navi.current != _last_ptr
            _last_ptr = train_task.navi.current
            target, is_forward_binding, is_backward_binding, is_head, is_tail = train_task.navi.target(0)
            is_last = train_task.navi.is_last
            should_leave = train_task.navi.is_last_backward_binding or is_last
            if not should_leave:
                next_target, _, _, _, _ = train_task.navi.target(1)
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
                _h = cast(HeadTrainShim, res)
                _inp = cast(TrainHeadInput, input)
                _pos_ids, _input_lens, _label_ids = _inp.position_ids, _inp.input_lens, _inp.label_ids
                _dtype = torch.bfloat16 if _inp.is_bfloat16 else torch.float16
                metric("wait_head")
                can_continue = await _h.can_continue_or_reoder(req_version=req_version, minibatch_id=context_id,
                                                               full_batch=full_batch, on_reoders_start=remove_barrier, on_reoders_done=add_barrier)
                if not can_continue:
                    train_task.input = _inp
                    self._on_drop(train_task)
                    return
                if self.dropped:
                    return
                _s, _e = metric_pair("head")
                _inp = await _h.train_tensor_to_embedding_tokens(
                    req_version=req_version, minibatch_id=context_id,
                    hidden_states=_inp.input_ids_tensor, dtype=_dtype, cpu_out=should_leave,
                    on_sync_begin=remove_barrier, on_sync_done=add_barrier,
                    begin_metric=_s, finish_metric=_e
                )
                _inp = TrainForwardMeta(
                    sample_batch_id=context_id,
                    hidden_states=_inp,
                    position_ids=_pos_ids,
                    device_pos_idx=None,
                    input_lens=_input_lens,
                    label_ids=_label_ids
                )
                metric("async_head")
                train_task.navi.next()
                if should_leave:
                    train_task.input = _inp
                    return train_task
            elif is_tail:
                _t = cast(TailTrainBindingShim, res)
                _cpu_in_out = False
                if _inp is None:
                    _cpu_in_out = True
                    _inp = cast(TrainTailInput, input)
                else:
                    assert isinstance(_inp, TrainTailInput)
                metric("wait_tail")
                can_continue = await _t.can_continue_or_reoder(req_version=req_version, minibatch_id=context_id,
                                                               full_batch=full_batch, on_reoders_start=remove_barrier, on_reoders_done=add_barrier)
                if not can_continue:
                    train_task.input = _inp
                    self._on_drop(train_task)
                    return
                if self.dropped:
                    return
                _s, _e = metric_pair("tail")
                loss, grad = await _t.compute_logits_loss_and_grad(
                    req_version=req_version, minibatch_id=context_id,
                    carrier=_inp.carrier, label_ids=_inp.label_ids_tensor, cpu_in_out=_cpu_in_out,
                    on_sync_begin=remove_barrier, on_sync_done=add_barrier,
                    begin_metric=_s, finish_metric=_e
                )
                _inp = grad
                train_task.loss = loss
                metric("async_tail")
                train_task.navi.next()
                if should_leave:
                    train_task.input = _inp
                    return train_task
            elif is_forward_binding:
                _b = cast(DecodersTrainBindingShim, res)
                _cpu_in = False
                if _inp is None:
                    _cpu_in = True
                    _inp = cast(TrainForwardMeta, input)
                else:
                    assert isinstance(_inp, TrainForwardMeta)
                metric("wait_forward")
                can_continue = await _b.can_continue_or_reoder(req_version=req_version, minibatch_id=context_id,
                                                               full_batch=full_batch, is_forward=True, on_reoders_start=remove_barrier, on_reoders_done=add_barrier)
                if not can_continue:
                    train_task.input = _inp
                    self._on_drop(train_task)
                    return
                if self.dropped:
                    return
                _is_first, _is_last = train_task.navi.is_first_or_last_forward_binding
                _label_ids = _inp.label_ids
                _s, _e = metric_pair("forward")
                _inp = await _b.forward(
                    req_version=req_version,
                    meta=_inp, cpu_in=_cpu_in, cpu_out=should_leave, is_first=_is_first, is_last=_is_last,
                    on_sync_begin=remove_barrier, on_sync_done=add_barrier,
                    begin_metric=_s, finish_metric=_e,
                )
                metric("async_forward")
                train_task.navi.next()
                if _is_last:
                    assert isinstance(_inp, DeviceTensorCarrier)
                    _inp = TrainTailInput(_inp, _label_ids)
                if should_leave:
                    train_task.input = _inp
                    return train_task
            elif is_backward_binding:
                _b = cast(DecodersTrainBindingShim, res)
                _cpu_in = False
                if _inp is None:
                    _cpu_in = True
                    _inp = cast(DeviceTensorCarrier, input)
                else:
                    assert isinstance(_inp, DeviceTensorCarrier)
                metric("wait_backward")
                can_continue = await _b.can_continue_or_reoder(req_version=req_version, minibatch_id=context_id,
                                                               full_batch=full_batch, is_forward=False, on_reoders_start=remove_barrier, on_reoders_done=add_barrier)
                if not can_continue:
                    train_task.input = _inp
                    self._on_drop(train_task)
                    return
                if self.dropped:
                    return
                _is_last = train_task.navi.is_last_backward_binding
                _s, _e = metric_pair("backward")
                _inp = await _b.backward(
                    req_version=req_version, context_id=context_id,
                    grad=_inp, cpu_in=_cpu_in, cpu_out=should_leave, has_next=not _is_last,
                    on_sync_begin=remove_barrier, on_sync_done=add_barrier,
                    begin_metric=_s, finish_metric=_e
                )
                metric("async_backward")
                train_task.navi.next()
                if _is_last:
                    assert _inp is None
                    should_leave = True
                if should_leave:
                    train_task.input = _inp
                    return train_task

    def remove(self):
        if self._removed:
            return
        self.dropped = True
        for c in self._caches:
            self._caches[c] = None
        self._caches = {}
        self._removed = True

    def __del__(self):
        assert self._removed
