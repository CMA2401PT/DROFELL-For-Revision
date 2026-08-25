from typing import Any, List, Optional, Dict, Set, Tuple
from typing import Any
from dataclasses import dataclass
from dataset.common import IGNORE_PAD_TOKEN_ID_IN_LABEL, ProcessedRecord
from operators import flags
from operators.core.head.real import ModelHead
from operators.core.ptuning.common import PtuningConfig

from operators.core.tokenizer.common import SequenceCore, SequenceLengthData
from operators.core.sampler.common import SamplingParams
import extend_asyncio as asyncio
from .base_types import *
from .brain import *
from .reorder import TrainTaskReorderAndDropCtrl, InferTaskReorderAndDropCtrl


@dataclass
class Sample:
    is_bfloat16: bool
    input_lens: List[int]
    position_ids_tensor: torch.Tensor
    input_ids_tensor: torch.Tensor
    label_ids_tensor: torch.Tensor

    @staticmethod
    def create_from_records(samples: List[ProcessedRecord], is_bfloat16: bool):
        input_lens = [len(sample.input_ids) for sample in samples]
        return Sample(
            is_bfloat16=is_bfloat16,
            input_lens=input_lens,
            position_ids_tensor=torch.tensor(data=[i for input_len in input_lens for i in range(input_len)], dtype=torch.int32, device="cpu", requires_grad=False),
            input_ids_tensor=torch.tensor([e for sample in samples for e in sample.input_ids], dtype=torch.int32, device="cpu", requires_grad=False),
            label_ids_tensor=torch.tensor([e for sample in samples for e in sample.output_ids], dtype=torch.long, device="cpu", requires_grad=False),
        )


class Shim:
    def __init__(self, binding: str, brain: Brain) -> None:
        self.binding = binding
        self.brain = brain


MetricFn = Optional[Callable[[], None]]


class HeadInferShim(Shim):
    def __init__(self, binding: str, brain: Brain, init_step: int) -> None:
        super().__init__(binding, brain)
        self.ctrl = InferTaskReorderAndDropCtrl(init_step=init_step)

    def can_continue_or_reoder(self, step: int, is_prefill: bool, on_reorder_start: Callable[[], None], on_reorder_end: Callable[[], None]):
        return self.ctrl.can_continue_or_reoder(step, is_prefill, on_reorder_start=on_reorder_start, on_reorder_end=on_reorder_end)

    async def infer_tensor_to_embedding_tokens(
            self, context_id: str, tokens: List[int], is_prefill: bool, step: int,
            cpu_out: bool,
            on_sync_begin: Callable[[], None], on_sync_done: Callable[[], None],
            begin_metric: MetricFn = None, finish_metric: MetricFn = None,
            **kwargs: Any):
        assert self.ctrl.can_operate(step, is_prefill)
        self.ctrl.set_running()

        op = OperateInferTensorToEmbeddingTokens(
            binding=self.binding, fut=asyncio.Future(), sync=None, extra=kwargs,
            tokens=tokens, is_prefill=is_prefill,
            context_id=context_id, cpu_in=False, cpu_out=cpu_out,
            begin_metric=begin_metric, finish_metric=finish_metric)
        self.brain.operates_infer_tensor_to_embedding_tokens.append(op)
        ret = await op.get_result(on_sync_begin, on_sync_done)
        self.ctrl.add_complete_task(step, is_prefill)
        return ret


class HeadTrainShim(Shim):
    def __init__(self, binding: str, brain: Brain, init_version: int, on_update_complete: Callable[[int], None]) -> None:
        super().__init__(binding, brain)
        self.ctrl = TrainTaskReorderAndDropCtrl(init_version, False)
        self.on_update_complete = on_update_complete

    def can_continue_or_reoder(self, req_version: int, minibatch_id: str, full_batch: Union[Set[str], List[str]],
                               on_reoders_start: Callable[[], None] = lambda: None, on_reoders_done: Callable[[], None] = lambda: None):
        return self.ctrl.can_continue_or_reoder(req_version, minibatch_id, full_batch, False, on_reoders_start, on_reoders_done)

    async def train_tensor_to_embedding_tokens(
            self, req_version: int, minibatch_id: SampleBatchID,
            hidden_states: torch.Tensor, dtype: torch.dtype,
            cpu_out: bool,
            on_sync_begin: Callable[[], None], on_sync_done: Callable[[], None],
            begin_metric: MetricFn = None, finish_metric: MetricFn = None, **kwargs: Any):
        assert self.ctrl.can_operate(req_version, minibatch_id)
        self.ctrl.set_running(req_version, minibatch_id)
        op = OperateTrainTensorToEmbeddingTokens(
            binding=self.binding, fut=asyncio.Future(), sync=None, extra=kwargs,
            hidden_states=hidden_states, dtype=dtype,
            context_id=minibatch_id, cpu_in=False, cpu_out=cpu_out,
            begin_metric=begin_metric, finish_metric=finish_metric)
        self.brain.operates_train_tensor_to_embedding_tokens.append(op)
        ret = await op.get_result(on_sync_begin, on_sync_done)
        self.ctrl.add_complete_task(req_version, minibatch_id)
        if self.ctrl.should_update:
            self.ctrl.version_update()
            self.on_update_complete(self.ctrl.version)
        return ret

    # async def forward_with_wrap(
    #         self, batch_id: SampleBatchID, sample: Sample,
    #         cpu_out: bool, begin_metric: MetricFn = None, finish_metric: MetricFn = None, **kwargs: Any):
    #     '''
    #     helper function which simplify input/output of head/decoders dealer
    #     '''
    #     dtype = torch.bfloat16 if sample.is_bfloat16 else torch.float16
    #     cpu_pos_idx = sample.position_ids_tensor
    #     hidden_states, gpu_pos_idx = await self.train_tensor_to_embedding_tokens(
    #         batch_id=batch_id,
    #         hidden_states=sample.input_ids_tensor,
    #         pos_idx=sample.position_ids_tensor,
    #         dtype=dtype,
    #         cpu_out=cpu_out,
    #         begin_metric=begin_metric, finish_metric=finish_metric,
    #         **kwargs
    #     )

    #     meta = TrainForwardMeta(
    #         sample_batch_id=batch_id,
    #         hidden_states=DeviceTensorCarrier(hidden_states, None),
    #         cpu_pos_idx=cpu_pos_idx,
    #         gpu_pos_idx=gpu_pos_idx,
    #         input_lens=sample.input_lens,
    #         label_ids_tensor=sample.label_ids_tensor
    #     )
    #     return meta


class HeadDealerShim(Shim):
    def __init__(self, binding: str, brain: Brain) -> None:
        self.binding = binding
        self.brain = brain
        super().__init__(binding, brain)

    @property
    def underlay_model(self) -> ModelHead:
        return self.brain.get_underlay_model_head(self.binding)

    async def create_head_infer_binding(self, init_step: int):
        return HeadInferShim(self.binding, self.brain, init_step)

    async def create_head_train_binding(self, init_version: int, on_update_complete: Callable[[int], None]):
        return HeadTrainShim(self.binding, self.brain, init_version, on_update_complete)

    def __del__(self):
        self.brain.remove_model_head(binding=self.binding)


class TailInferBindingShim(Shim):
    def __init__(self, seq_id: str, init_step: int, binding: str, seq_core: SequenceCore, sampling_params: SamplingParams, brain: Brain) -> None:
        super().__init__(binding, brain)
        self.seq_core = seq_core
        self.sampling_params = sampling_params
        self.seq_id = seq_id
        self.ctrl = InferTaskReorderAndDropCtrl(init_step=init_step)

    def can_continue_or_reoder(self, step: int, is_prefill: bool, on_reorder_start: Callable[[], None], on_reorder_end: Callable[[], None]):
        return self.ctrl.can_continue_or_reoder(step=step, is_prefill=is_prefill, on_reorder_start=on_reorder_start, on_reorder_end=on_reorder_end)

    async def sampling(
            self, context_id: str, step: int, carrier: DeviceTensorCarrier, cpu_in: bool, is_prefill: bool,
            on_sync_begin: Callable[[], None], on_sync_done: Callable[[], None],
            begin_metric: MetricFn = None, finish_metric: MetricFn = None,
            **kwargs: Any):
        '''
        step=0: prefill, step=1,2,3... generating
        '''
        assert self.ctrl.can_operate(step, is_prefill=is_prefill)
        self.ctrl.set_running()
        op = OperateHiddenStateToSamplingOutput(
            binding=self.binding, fut=asyncio.Future(), sync=None, extra=kwargs,
            seq_core=self.seq_core, sampling_params=self.sampling_params, carrier=carrier,
            context_id=context_id, cpu_in=cpu_in, cpu_out=False,
            begin_metric=begin_metric, finish_metric=finish_metric)
        self.brain.operates_hidden_state_to_sampling_output.append(op)
        ret = await op.get_result(on_sync_begin, on_sync_done)
        if not is_prefill:
            assert step+1 == len(self.seq_core.output_token_ids), f"{step}+1!={len(self.seq_core.output_token_ids)}"
        self.ctrl.add_complete_task(step, is_prefill=is_prefill)
        return ret


class TailTrainBindingShim(Shim):
    def __init__(self, binding: str, criterion: torch.nn.Module, brain: Brain, init_version: int, on_update_complete: Callable[[int], None]) -> None:
        super().__init__(binding, brain)
        self.criterion = criterion
        self.ctrl = TrainTaskReorderAndDropCtrl(init_version, False)
        self.on_update_complete = on_update_complete

    def can_continue_or_reoder(self, req_version: int, minibatch_id: str, full_batch: Union[Set[str], List[str]],
                               on_reoders_start: Callable[[], None] = lambda: None, on_reoders_done: Callable[[], None] = lambda: None):
        return self.ctrl.can_continue_or_reoder(req_version, minibatch_id, full_batch, False, on_reoders_start, on_reoders_done)

    async def compute_logits_loss_and_grad(
            self, req_version: int, minibatch_id: SampleBatchID, carrier: DeviceTensorCarrier, label_ids: torch.Tensor,
            cpu_in_out: bool,
            on_sync_begin: Callable[[], None], on_sync_done: Callable[[], None],
            begin_metric: MetricFn = None, finish_metric: MetricFn = None, **kwargs: Any):
        assert self.ctrl.can_operate(req_version, minibatch_id)
        self.ctrl.set_running(req_version, minibatch_id)
        op = OperateComputeLogitsLossAndGrad(
            binding=self.binding, fut=asyncio.Future(), sync=None, extra=kwargs,
            criterion=self.criterion, carrier=carrier, label_ids=label_ids,
            context_id=minibatch_id, cpu_in=cpu_in_out, cpu_out=cpu_in_out,
            begin_metric=begin_metric, finish_metric=finish_metric)
        self.brain.operates_compute_logits_loss_and_grad.append(op)
        ret = await op.get_result(on_sync_begin, on_sync_done)
        self.ctrl.add_complete_task(req_version, minibatch_id)
        if self.ctrl.should_update:
            self.ctrl.version_update()
            self.on_update_complete(self.ctrl.version)
        return ret


class TailDealerShim(Shim):
    def __init__(self, binding: str, brain: Brain) -> None:
        super().__init__(binding, brain)

    @property
    def underlay_model(self) -> ModelTail:
        return self.brain.get_underlay_model_tail(self.binding)

    async def create_infer_binding(self, seq_id: str, init_step: int, seq_core: SequenceCore, sampling_params: SamplingParams):
        return TailInferBindingShim(seq_id=seq_id, init_step=init_step, binding=self.binding, seq_core=seq_core, sampling_params=sampling_params, brain=self.brain)

    async def create_train_binding(self, init_version: int, on_update_complete: Callable[[int], None], impl: str = ""):
        criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_PAD_TOKEN_ID_IN_LABEL)
        return TailTrainBindingShim(binding=self.binding, criterion=criterion, brain=self.brain, init_version=init_version, on_update_complete=on_update_complete)

    def __del__(self):
        self.brain.remove_model_tail(binding=self.binding)


class DecodersTrainBindingShim(Shim):
    def __init__(self, name: str, init_version: int, binding: str, brain: Brain, num_layers: int, on_should_update: Callable[["DecodersTrainBindingShim"], None]) -> None:
        super().__init__(binding, brain)
        self.num_layers = num_layers
        self.ctrl = TrainTaskReorderAndDropCtrl(init_version, with_fwd_and_bwd=True)
        self.on_should_update: Callable[["DecodersTrainBindingShim"], None] = on_should_update
        self.name = name

    @property
    def version_ctrl(self):
        return self.ctrl

    @property
    def version(self):
        return self.ctrl.version

    def can_continue_or_reoder(self, req_version: int, minibatch_id: str, full_batch: Union[Set[str], List[str]], is_forward: bool,
                               on_reoders_start: Callable[[], None] = lambda: None, on_reoders_done: Callable[[], None] = lambda: None):
        return self.ctrl.can_continue_or_reoder(req_version, minibatch_id, full_batch, is_forward, on_reoders_start, on_reoders_done)

    async def forward(
            self, req_version: int, meta: TrainForwardMeta, is_last: bool, is_first: bool,
            cpu_in: bool, cpu_out: bool,
            on_sync_begin: Callable[[], None], on_sync_done: Callable[[], None],
            begin_metric: MetricFn = None, finish_metric: MetricFn = None,
            **kwargs: Any):
        context_id = meta.sample_batch_id
        assert self.ctrl.can_operate(req_version=req_version, minibatch_id=context_id, is_forward=True)
        self.ctrl.set_running(req_version=req_version, minibatch_id=context_id, is_forward=True)
        op = OperateForward(binding=self.binding, fut=asyncio.Future(), sync=None, extra=kwargs,
                            meta=meta, is_last=is_last, is_first=is_first,
                            context_id=meta.sample_batch_id, cpu_in=cpu_in, cpu_out=cpu_out,
                            begin_metric=begin_metric, finish_metric=finish_metric)
        self.brain.operates_forward.append(op)
        meta = await op.get_result(on_sync_begin, on_sync_done)
        self.ctrl.add_complete_task(req_version=req_version, minibatch_id=context_id, is_forward=True)
        if is_last:
            assert meta.hidden_states is not None
            return meta.hidden_states
        else:
            return meta

    async def backward(
            self, req_version: int, context_id: SampleBatchID, grad: DeviceTensorCarrier, has_next: bool,
            cpu_in: bool, cpu_out: bool,
            on_sync_begin: Callable[[], None], on_sync_done: Callable[[], None],
            begin_metric: MetricFn = None, finish_metric: MetricFn = None, **kwargs: Any):
        assert self.ctrl.can_operate(req_version=req_version, minibatch_id=context_id, is_forward=False)
        self.ctrl.set_running(req_version=req_version, minibatch_id=context_id, is_forward=False)
        op = OperateBackward(binding=self.binding, fut=asyncio.Future(), sync=None, extra=kwargs,
                             grad=grad, has_next=has_next,
                             context_id=context_id,
                             cpu_in=cpu_in, cpu_out=cpu_out, begin_metric=begin_metric, finish_metric=finish_metric)
        self.brain.operates_backward.append(op)
        ret = await op.get_result(on_sync_begin, on_sync_done)
        self.ctrl.add_complete_task(req_version=req_version, minibatch_id=context_id, is_forward=False)
        if self.ctrl.should_update:
            # print("should update")
            self.on_should_update(self)
        # else:
            # print("no update")
        return ret

    async def update(self, **kwargs: Any):
        op = OperateWeightUpdate(binding=self.binding, fut=asyncio.Future(), sync=None, extra=kwargs)
        self.brain.operates_weight_update.append(op)
        await op.get_result(lambda: None, lambda: None)
        self.ctrl.version_update()

    async def dump_train_states(self):
        op = OperateDumpTrainStates(binding=self.binding, fut=asyncio.Future(), sync=None, extra={})
        self.brain.operates_dump_train_states.append(op)
        return await op.get_result(lambda: None, lambda: None)

    async def load_train_states(self,  states: Dict[str, Any]):
        op = OperateLoadTrainStates(binding=self.binding, fut=asyncio.Future(), sync=None, extra={},
                                    states=states)
        self.brain.operates_load_train_states.append(op)
        return await op.get_result(lambda: None, lambda: None)

    def __del__(self):
        self.brain.remove_decoders_train_binding(binding=self.binding)


class DecodersInferBindingShim(Shim):
    def __init__(self, name: str, seq_id: str, init_step: int, init_slot_count: int,
                 binding: str, brain: Brain,
                 num_layers: int, user_model_handle: Optional[UserModelHandle]) -> None:
        super().__init__(binding, brain)
        self.num_layers = num_layers
        self.user_model_handle = user_model_handle
        self.seq_id = seq_id
        self.ctrl = InferTaskReorderAndDropCtrl(init_step=init_step)
        self.slots_count = init_slot_count
        self.is_first_generating: bool = True
        self.on_ready_blocks_change: Callable[["DecodersInferBindingShim"], None] = lambda _: None
        self.on_before_generating: Callable[["DecodersInferBindingShim"], None] = lambda _: None
        self.on_after_generating: Callable[["DecodersInferBindingShim"], None] = lambda _: None
        self.name = name

    @property
    def occupied_blocks(self):
        return [i for i in range(self.num_blocks_occupied)]

    @property
    def num_blocks_occupied(self):
        return (self.slots_count+(flags.BLOCK_SIZE-1))//flags.BLOCK_SIZE

    async def put_blocks(self, seq_blocks: List[int], kv_cache: List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]]):
        op = OperatePutBlocks(binding=self.binding, fut=asyncio.Future(), sync=None, extra={},
                              seq_blocks=seq_blocks, kv_cache=kv_cache)
        self.brain.operates_put_blocks.append(op)
        return await op.get_result(lambda: None, lambda: None)

    async def get_blocks(self, seq_blocks: List[int]):
        op = OperateGetBlocks(binding=self.binding, fut=asyncio.Future(), sync=None, extra={},
                              seq_blocks=seq_blocks)
        self.brain.operates_get_blocks.append(op)
        return await op.get_result(lambda: None, lambda: None)

    def can_continue_or_reoder(self, step: int, is_prefill: bool, on_reorder_start: Callable[[], None], on_reorder_end: Callable[[], None]):
        return self.ctrl.can_continue_or_reoder(step=step, is_prefill=is_prefill, on_reorder_start=on_reorder_start, on_reorder_end=on_reorder_end)

    @property
    def ready_blocks(self):
        return self.slots_count//flags.BLOCK_SIZE

    async def generating(
            self, context_id: str, step: int, carrier: DeviceTensorCarrier, tail_in: bool,
            cpu_in: bool, cpu_out: bool,
            on_sync_begin: Callable[[], None], on_sync_done: Callable[[], None],
            begin_metric: MetricFn = None, finish_metric: MetricFn = None, **kwargs: Any):
        assert self.ctrl.can_operate(step, is_prefill=False)
        self.ctrl.set_running()
        self.on_before_generating(self)
        if (self.slots_count % flags.BLOCK_SIZE == 0) or \
                self.is_first_generating:
            self.on_ready_blocks_change(self)
        self.is_first_generating = False
        op = OperateDoGenerating(binding=self.binding, fut=asyncio.Future(), sync=None, extra=kwargs,
                                 carrier=carrier, tail_in=tail_in,
                                 context_id=context_id, cpu_in=cpu_in, cpu_out=cpu_out, begin_metric=begin_metric, finish_metric=finish_metric)
        self.brain.operates_do_generating.append(op)
        ret = await op.get_result(on_sync_begin, on_sync_done)
        self.slots_count += 1
        self.on_after_generating(self)
        self.ctrl.add_complete_task(step, False)
        return ret

    async def prefill(
            self, context_id: str, step: int,  carrier: DeviceTensorCarrier, tail_in: bool,
            cpu_in: bool, cpu_out: bool,
            on_sync_begin: Callable[[], None], on_sync_done: Callable[[], None],
            begin_metric: MetricFn = None, finish_metric: MetricFn = None, **kwargs: Any):
        assert self.ctrl.can_operate(step, is_prefill=True)
        self.ctrl.set_running()
        input_len = carrier.shape[0]
        op = OperateDoPrefill(binding=self.binding, fut=asyncio.Future(), sync=None, extra=kwargs,
                              carrier=carrier, tail_in=tail_in,
                              context_id=context_id, cpu_in=cpu_in, cpu_out=cpu_out, begin_metric=begin_metric, finish_metric=finish_metric)
        self.brain.operates_do_prefill.append(op)
        ret = await op.get_result(on_sync_begin, on_sync_done)
        self.slots_count = input_len
        self.ctrl.add_complete_task(step, True)
        return ret

    def __del__(self):
        self.brain.remove_decoders_infer_binding(binding=self.binding)


class DecodersShim(Shim):
    def __init__(self, binding: str, brain: Brain, num_layers: int) -> None:
        super().__init__(binding, brain)
        self.num_layers = num_layers
        self.decoder_config = self.underlay_model.decoder_config
        self.user_model_handles: Dict[str, Tuple[int, UserModelHandle]] = {}
        self.user_model_waitor: Dict[str, asyncio.Future[None]] = {}

    @property
    def underlay_model(self) -> DecodersGroup:
        return self.brain.get_underlay_decoders(self.binding)

    async def create_infer_binding(self,
                                   seq_id: str,
                                   init_step: int,
                                   seq_length_data: SequenceLengthData,
                                   user_model_handle: Optional[UserModelHandle],
                                   allocate_slots: bool,
                                   name: str,
                                   ):
        user_model = None
        if user_model_handle is not None:
            user_model = user_model_handle.blocks
            if user_model_handle.lora_render_cfg is not None:
                user_model = (user_model_handle.lora_render_cfg, user_model)

        op = OperateCreateDecodersInferBinding(
            binding=self.binding, fut=asyncio.Future(), sync=None, extra={},
            seq_length_data=seq_length_data,
            user_model=user_model,
            allocate_slots=allocate_slots,
            binding_name=name
        )
        self.brain.operates_create_decoders_infer_binding.append(op)
        binding = await op.get_result(lambda: None, lambda: None)
        init_slot_count = 0 if not allocate_slots else seq_length_data.num_slots
        return DecodersInferBindingShim(
            name=name, seq_id=seq_id, init_step=init_step, init_slot_count=init_slot_count,
            binding=binding, brain=self.brain, num_layers=self.num_layers, user_model_handle=user_model_handle)

    async def create_train_binding(self,
                                   version: int,
                                   user_model_config: Union[PtuningConfig, LoraConfig],
                                   optimizer_config: OptimizerConfig,
                                   scheduler_config: Optional[SchedulerConfig],
                                   data_init_method: Optional[UserModelDataInitConfig],
                                   on_should_update: Callable[["DecodersTrainBindingShim"], None],
                                   name: str,
                                   ):
        op = OperateCreateDecodersTrainBinding(binding=self.binding, fut=asyncio.Future(), sync=None, extra={},
                                               user_model_config=user_model_config, optimizer_config=optimizer_config,
                                               scheduler_config=scheduler_config, data_init_method=data_init_method, binding_name=name)
        self.brain.operates_create_decoders_train_binding.append(op)
        binding = await op.get_result(lambda: None, lambda: None)
        return DecodersTrainBindingShim(
            name=name, init_version=version, binding=binding, brain=self.brain,
            num_layers=self.num_layers, on_should_update=on_should_update)

    async def put_user_model(
        self,
        unique_name: str,
        user_model_blocks: List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]],
        lora_or_ptuning: Optional[LoraRenderConfig]
    ):
        if unique_name in self.user_model_waitor:
            await self.user_model_waitor[unique_name]
        if unique_name in self.user_model_handles:
            ref_count, handle = self.user_model_handles[unique_name]
            self.user_model_handles[unique_name] = (ref_count+1, handle)
            return handle

        op = OperatePutUserModel(
            binding=self.binding,
            fut=asyncio.Future(), sync=None, extra={},
            user_model_blocks=user_model_blocks,
            lora_render_cfg=lora_or_ptuning
        )
        self.brain.operates_put_user_model.append(op)
        fut: asyncio.Future[None] = asyncio.Future()
        self.user_model_waitor[unique_name] = fut
        handle = await op.get_result(lambda: None, lambda: None)
        del self.user_model_waitor[unique_name]
        fut.set_result(None)
        self.user_model_handles[unique_name] = (1, handle)
        return handle

    def remove_user_model(self, unique_name: str):
        ref_count, handle = self.user_model_handles[unique_name]
        ref_count -= 1
        if ref_count == 0:
            handle.free()
            del self.user_model_handles[unique_name]
            return
        self.user_model_handles[unique_name] = (ref_count, handle)

    def __del__(self):
        assert len(self.user_model_handles) == 0
        self.brain.remove_model_decoders(binding=self.binding)


class Provider:
    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self.caches: Dict[str, Any] = {}
        self.restriction: Optional[Set[str]] = None

    def remove(self):
        self.caches.clear()

    def __getitem__(self, key: str) -> Union[HeadDealerShim, TailDealerShim, DecodersShim]:
        if self.restriction is not None:
            assert key in self.restriction
        cache = self.caches.get(key)
        if cache is not None:
            return cache
        describe = ModelDescribe.from_str(describe_str=key)
        if describe.is_head:
            return self.get_model_head_dealer(describe_str=key)
        elif describe.is_tail:
            return self.get_model_tail_dealer(describe_str=key)
        else:
            return self.get_model_decoders_dealer(describe_str=key)

    def get_model(self, describe_str: str):
        return self.__getitem__(describe_str)

    def get_model_head_dealer(self, describe_str: str) -> HeadDealerShim:
        cache = self.caches.get(describe_str)
        if cache is not None:
            return cache
        describe = ModelDescribe.from_str(describe_str)
        assert describe.is_head
        binding = self.brain.make_model_head(base_model=describe.base_model)
        head_shim = HeadDealerShim(binding, self.brain)
        self.caches[describe_str] = head_shim
        return head_shim

    def get_model_tail_dealer(self,  describe_str: str) -> TailDealerShim:
        cache = self.caches.get(describe_str)
        if cache is not None:
            return cache
        describe = ModelDescribe.from_str(describe_str)
        assert describe.is_tail
        binding = self.brain.make_model_tail(base_model=describe.base_model)
        tail_shim = TailDealerShim(binding, self.brain)
        self.caches[describe_str] = tail_shim
        return tail_shim

    def get_model_decoders_dealer(self, describe_str: str) -> DecodersShim:
        cache = self.caches.get(describe_str)
        if cache is not None:
            return cache
        describe = ModelDescribe.from_str(describe_str)
        assert describe.is_decoders
        binding = self.brain.make_model_decoders(base_model=describe.base_model, decoders_range=describe.decoders_range)
        num_layers = describe.decoders_range[1]-describe.decoders_range[0]
        decoders_shim = DecodersShim(binding, self.brain, num_layers)
        self.caches[describe_str] = decoders_shim
        return decoders_shim
