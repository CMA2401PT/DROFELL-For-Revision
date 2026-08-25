from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
import time
from typing import Any, Awaitable, Callable, Generic, List, Literal, Optional, Dict, Tuple, TypeVar

from utils.format import format_size
from utils.runtime_metrics import record_sampled_event

from .barrier import FreeStateOpMantainer
from operators.core.lora.common import LoraConfig
from operators.core.lora.render import LoraRenderConfig
from operators.core.tail.real import ModelTail
from operators.assemble.decoders.real_decoders_group import DecodersGroup
from operators.core.head.real import ModelHead
from operators.assemble.decoders.real import DecodersImplement, make_decoders_from_files
from operators.helpers.model_lib.common import ModelsLib
from operators.core.tokenizer.common import SequenceCore, SequenceLengthData
from operators.core.sampler.common import SamplingParams
from operators.core.ptuning.common import PtuningConfig
import extend_asyncio as asyncio

from .base_types import *


RetType = TypeVar("RetType")


@dataclass
class UserModelDataInitConfig:
    init_method: Literal["none", "from_ckpt", "reinit"] = "reinit"
    # same keys as decoders' count
    ckpt_bytes: Optional[Dict[int, bytes]] = None


@dataclass
class OptimizerConfig:
    optimizer_name: str = "AdamW"
    lr: float = 1e-2
    eps: float = 1e-4


@dataclass
class SchedulerConfig:
    scheduler_name: str = "MultiStepLR"
    scheduler_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "milestones": [2000, 5000, 10000, 15000, 20000, 25000],
        "gamma": 0.5,
        "last_epoch": -1
    })


@dataclass
class TrainForwardMeta:
    sample_batch_id: SampleBatchID
    hidden_states: Optional[DeviceTensorCarrier]
    # cpu_pos_idx: Optional[torch.Tensor]
    position_ids: Optional[List[int]]
    device_pos_idx: Optional[torch.Tensor]
    input_lens: List[int]
    label_ids: List[int]

    @property
    def cpu_pos_idx(self):
        assert self.position_ids is not None
        return torch.tensor(data=self.position_ids, dtype=torch.int32, device="cpu", requires_grad=False)

    @property
    def is_cpu(self):
        return self.hidden_states is None or self.hidden_states.is_cpu

    def shallow_copy(self):
        return TrainForwardMeta(
            sample_batch_id=self.sample_batch_id,
            hidden_states=self.hidden_states.shallow_copy(
            ) if self.hidden_states is not None else None,
            position_ids=self.position_ids,
            device_pos_idx=self.device_pos_idx,
            input_lens=self.input_lens,
            label_ids=self.label_ids,
        )

    def debug_check_cpu(self):
        if self.hidden_states is not None:
            assert self.hidden_states.is_cpu
        assert self.device_pos_idx is None


@dataclass
class Operate(Generic[RetType]):
    binding: str
    extra: Optional[Dict[str, Any]]
    fut: asyncio.Future[RetType]
    sync: Optional[Awaitable[None]]

    def on_start(self):
        self.start_called = True

    def on_finish(self):
        self.finish_called = True

    def set_result(self, ret: RetType, sp: Optional[Awaitable[None]] = None):
        self.ret = ret
        self.fut.set_result(ret)
        self.sync = sp

    @property
    def is_sync_result(self):
        return self.sync is not None

    async def get_result(
        self, on_sync_begin: Callable[[], None], on_sync_done: Callable[[], None]
    ):
        ret = await self.fut
        if self.sync is not None:
            on_sync_begin()
            sync_started_ns = time.perf_counter_ns()
            try:
                await self.sync
            finally:
                record_sampled_event(
                    "sync", "operate_sync_wait",
                    operation=type(self).__name__,
                    binding=str(self.binding),
                    context_id=getattr(self, "context_id", None),
                    cpu_in=getattr(self, "cpu_in", None),
                    cpu_out=getattr(self, "cpu_out", None),
                    duration_cpu_ns=(
                        time.perf_counter_ns() - sync_started_ns
                    ),
                )
            on_sync_done()
        return ret

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['fut']
        del state['sync']
        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)
        self.start_called = True
        self.finish_called = True

    # def __del__(self):
    #     assert self.start_called == True
    #     assert self.finish_called == True

    def desc(self) -> Tuple[str, ...]:
        return ("operate", self.binding)


OperateType = TypeVar("OperateType", bound=Operate[Any])


@dataclass
class ComputeOperate(Operate[RetType]):
    context_id: str
    cpu_in: bool  # not all operate has cpu_in
    cpu_out: bool  # not all operate has cpu_out
    begin_metric: Optional[Callable[[], None]]
    finish_metric: Optional[Callable[[], None]]

    def on_start(self):
        self.start_called = True
        if self.begin_metric is not None:
            self.begin_metric()

    def on_finish(self):
        self.finish_called = True
        if self.finish_metric is not None:
            self.finish_metric()

    def set_result(self, ret: RetType, sp: Optional[Awaitable[None]] = None):
        self.ret = ret
        self.finish_called = True
        self.fut.set_result(ret)
        self.sync = sp

    def desc(self) -> Tuple[str, ...]:
        return ("compute_operate", self.binding, self.context_id)


@dataclass
class OperateInferTensorToEmbeddingTokens(ComputeOperate[DeviceTensorCarrier]):
    tokens: List[int]
    cpu_out: bool
    is_prefill: bool
    context_id: str
    recovery_decode: bool = False

    def desc(self):
        return ("infer_head", self.binding, self.context_id)


@dataclass
class OperateTrainTensorToEmbeddingTokens(ComputeOperate[DeviceTensorCarrier]):
    hidden_states: torch.Tensor
    dtype: torch.dtype

    def desc(self):
        return ("train_head", self.binding, self.context_id)


@dataclass
class OperateHiddenStateToSamplingOutput(ComputeOperate[Tuple[int, Dict[int, float]]]):
    seq_core: SequenceCore
    sampling_params: SamplingParams
    carrier: DeviceTensorCarrier
    is_prefill: bool

    def desc(self):
        return ("infer_tail", self.binding, self.context_id)


@dataclass
class OperateComputeLogitsLossAndGrad(ComputeOperate[Tuple[float, DeviceTensorCarrier]]):
    carrier: DeviceTensorCarrier
    criterion: torch.nn.Module
    label_ids: torch.Tensor

    def desc(self):
        return ("train_tail", self.binding, self.context_id)


@dataclass
class OperateForward(ComputeOperate[TrainForwardMeta]):
    meta: TrainForwardMeta
    is_last: bool
    is_first: bool

    def desc(self):
        return ("forward", self.binding, self.context_id)


@dataclass
class OperateBackward(ComputeOperate[Optional[DeviceTensorCarrier]]):
    grad: DeviceTensorCarrier
    has_next: bool

    def desc(self):
        return ("backward", self.binding, self.context_id)


@dataclass
class OperateWeightUpdate(Operate[None]):

    def desc(self):
        return ("weight_update", self.binding)


@dataclass
class OperateDumpTrainStates(Operate[Dict[str, Any]]):
    def desc(self):
        return ("dump_state", self.binding)


@dataclass
class OperateLoadTrainStates(Operate[None]):
    states: Dict[str, Any]

    def desc(self):
        return ("load_state", self.binding)


@dataclass
class UserModelHandle:
    blocks: List[int]
    lora_render_cfg: Optional[LoraRenderConfig]
    _free: Callable[[], None]
    _freed: bool = False

    def free(self):
        if self._freed:
            return
        self._free()
        self._freed = True

    def __del__(self):
        assert self._freed

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_free']
        del state['_freed']
        return state

    def __setstate__(self, state: Dict[str, Any]):
        state['_free'] = lambda: None
        state['_freed'] = True
        self.__dict__.update(state)


@dataclass
class OperatePutUserModel(Operate[UserModelHandle]):
    user_model_blocks: List[List[Tuple[DeviceTensorCarrier,
                                       DeviceTensorCarrier]]]
    lora_render_cfg: Optional[LoraRenderConfig]

    def desc(self):
        return ("put_user_model", self.binding)


@dataclass
class OperatePutBlocks(Operate[None]):
    seq_blocks: List[int]
    kv_cache: List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]]

    def desc(self):
        return ("put_blocks", self.binding)


@dataclass
class OperateGetBlocks(Operate[List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]]]):
    seq_blocks: List[int]

    def desc(self):
        return ("get_blocks", self.binding)


@dataclass
class DecodeHeadTask:
    """Head work owned by one parent decoder generating operation."""

    binding: str
    token: int
    begin_metric: Optional[Callable[[], None]] = None
    finish_metric: Optional[Callable[[], None]] = None


@dataclass
class DecodeTailTask:
    """Tail work owned by one parent decoder generating operation."""

    binding: str
    seq_core: SequenceCore
    sampling_params: SamplingParams
    begin_metric: Optional[Callable[[], None]] = None
    finish_metric: Optional[Callable[[], None]] = None


@dataclass
class DecodeGeneratingResult:
    """A generating operation returns either a carrier or sampled output."""

    carrier: Optional[DeviceTensorCarrier] = None
    sampling_output: Optional[Tuple[int, Dict[int, float]]] = None

    def __post_init__(self):
        if (self.carrier is None) == (self.sampling_output is None):
            raise ValueError(
                "decode result must contain exactly one output kind"
            )


@dataclass
class OperateDoGenerating(ComputeOperate[DecodeGeneratingResult]):
    """One decoder queue item with optional staged or fused head/tail work."""

    carrier: Optional[DeviceTensorCarrier]
    head_task: Optional[DecodeHeadTask]
    tail_task: Optional[DecodeTailTask]

    def __post_init__(self):
        if (self.carrier is None) != (self.head_task is not None):
            raise ValueError(
                "decode generating requires either carrier or head task"
            )
        if self.head_task is not None and self.cpu_in:
            raise ValueError("embedded head has no external carrier input")
        if self.tail_task is not None and self.cpu_out:
            raise ValueError("embedded tail returns sampling output")

    def desc(self):
        return ("generating", self.binding)


@dataclass
class OperateDoPrefill(ComputeOperate[DeviceTensorCarrier]):
    carrier: DeviceTensorCarrier
    tail_in: bool
    recovery_decode: bool = False

    def desc(self):
        return ("prefill", self.binding)


@dataclass
class OperateCreateDecodersInferBinding(Operate[str]):
    seq_length_data: SequenceLengthData
    user_model: Union[None, List[int], Tuple[LoraRenderConfig, List[int]]]
    allocate_slots: bool
    binding_name: str

    def desc(self):
        return ("create_infer_binding", self.binding)


@dataclass
class OperateCreateDecodersTrainBinding(Operate[str]):
    user_model_config: Union[PtuningConfig, LoraConfig]
    optimizer_config: OptimizerConfig
    scheduler_config: Optional[SchedulerConfig]
    data_init_method: Optional[UserModelDataInitConfig]
    binding_name: str

    def desc(self):
        return ("create_train_binding", self.binding)


class ListWithHook(Generic[OperateType]):
    def __init__(self):
        self.list: List[OperateType] = []
        self.hook: Callable[[OperateType], OperateType] = lambda op: op
        self.compare_fn: Optional[Callable[[
            OperateType], Union[int, str]]] = None
        # self.locked = False
        # self.put_later: List[OperateType] = []

    def __len__(self):
        return self.list.__len__()

    def _insort(self, op: OperateType):
        assert self.compare_fn is not None
        lo, hi = 0, len(self.list)
        while lo < hi:
            mid = (lo+hi)//2
            # type:ignore
            if self.compare_fn(op) < self.compare_fn(self.list[mid]):
                hi = mid
            else:
                lo = mid+1
        self.list.insert(lo, op)

    def append(self, op: OperateType):
        # if self.locked:
        #     self.put_later.append(op)
        #     return
        op = self.hook(op)
        if self.compare_fn is None:
            self.list.append(op)
        else:
            self._insort(op)

    def pop(self, idx: int):
        return self.list.pop(idx)

    # def lock(self):
    #     self.locked = True

    # def unlock(self):
    #     self.locked = False
    #     for op in self.put_later:
    #         self.append(op)
    #     self.put_later.clear()


@dataclass
class Profile:
    perdecoder_forward_tps: int = 0
    perdecoder_frowarad_backward_tps: int = 0
    perdecoder_generating_tps: int = 0
    perdecoder_memory: int = 0

    def __str__(self) -> str:
        s = "" +\
            f"Memory: {format_size(self.perdecoder_memory)}\n" +\
            f"Foward TPS: {self.perdecoder_forward_tps}\n" +\
            f"Foward+Backwad TPS: {self.perdecoder_forward_tps}\n" +\
            f"Generating TPS: {self.perdecoder_generating_tps}"
        return s


class Brain(ABC):
    def __init__(
        self,
        models_path: str,
        device: torch.device,
        *,
        max_seq_len: int,
        max_concurrency: int,
        decoder_implement: DecodersImplement,
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        # aux create
        self.operates_create_decoders_infer_binding: ListWithHook[OperateCreateDecodersInferBinding] = ListWithHook(
        )
        self.operates_create_decoders_train_binding: ListWithHook[OperateCreateDecodersTrainBinding] = ListWithHook(
        )

        # data movement
        self.operates_put_user_model: ListWithHook[OperatePutUserModel] = ListWithHook(
        )
        self.operates_put_blocks: ListWithHook[OperatePutBlocks] = ListWithHook(
        )
        self.operates_get_blocks: ListWithHook[OperateGetBlocks] = ListWithHook(
        )
        self.operates_dump_train_states: ListWithHook[OperateDumpTrainStates] = ListWithHook(
        )
        self.operates_load_train_states: ListWithHook[OperateLoadTrainStates] = ListWithHook(
        )

        # infer
        self.operates_infer_tensor_to_embedding_tokens: ListWithHook[OperateInferTensorToEmbeddingTokens] = ListWithHook(
        )
        self.operates_hidden_state_to_sampling_output: ListWithHook[OperateHiddenStateToSamplingOutput] = ListWithHook(
        )
        self.operates_do_generating: ListWithHook[OperateDoGenerating] = ListWithHook(
        )
        self.operates_do_prefill: ListWithHook[OperateDoPrefill] = ListWithHook(
        )

        # train
        self.operates_train_tensor_to_embedding_tokens: ListWithHook[OperateTrainTensorToEmbeddingTokens] = ListWithHook(
        )
        self.operates_compute_logits_loss_and_grad: ListWithHook[OperateComputeLogitsLossAndGrad] = ListWithHook(
        )
        self.operates_forward: ListWithHook[OperateForward] = ListWithHook()
        self.operates_backward: ListWithHook[OperateBackward] = ListWithHook()
        self.operates_weight_update: ListWithHook[OperateWeightUpdate] = ListWithHook(
        )

        # gather
        self._wait_operate_fut: Optional[asyncio.Future[None]] = None
        self.all_queues: List[Tuple[str, ListWithHook[Any]]] = []
        for k, v in self.__dict__.items():
            if k.startswith("operates_"):
                queue: ListWithHook[Any] = v
                queue.hook = self._awake
                self.all_queues.append((k.removeprefix("operates_"), queue))

        # underlay model
        self._models = ModelsLib(lib_path=models_path)
        self.device = device
        self.max_seq_len = max_seq_len
        self.max_concurrency = max_concurrency
        self.decoder_implement = decoder_implement
        self.cache_kernel_implement: Literal['vllm',
                                             'dual_table'] = "dual_table"
        self._base_heads: Dict[str, ModelHead] = {}
        self._base_tails: Dict[str, ModelTail] = {}
        self._base_decoders: Dict[str, DecodersGroup] = {}

        # barrier
        self.barrier_maintainer = FreeStateOpMantainer()

    def query_extra_info(self, key: str) -> Any:
        return None

    @property
    @abstractmethod
    def memory(self) -> Tuple[int, int]:
        """
        return: (total_memory, base_model_memory)
        """
        raise NotImplementedError

    async def gen_profile(
        self,
        model_name: str,
        train_kwargs: Optional[Dict[str, Any]] = None,
        generating_kwargs: Optional[Dict[str, Any]] = None
    ) -> Profile:
        raise NotImplementedError

    def _awake(self, op: OperateType) -> OperateType:
        if isinstance(op, ComputeOperate):
            self.barrier_maintainer.add_managed_op(op.context_id)
        if self._wait_operate_fut is None:
            return op
        fut = self._wait_operate_fut
        self._wait_operate_fut = None
        fut.set_result(None)
        return op

    def make_model_head(
        self,
        base_model: str,
        adjacent_decoder: Optional[str] = None,
    ) -> str:
        name = ModelDescribe.new(base_model, is_head=True).__str__()
        if name in self._base_heads:
            return name
        model_info = self._models.models_storage[base_model]
        with torch.device(self.device):
            model_head = ModelHead(
                config=model_info.config.model_head_config,
                dtype=torch.float16,
            )
            model_head.load_from_data_dir(model_info.model_head_path)
        self._base_heads[name] = model_head
        return name

    def remove_model_head(self, binding: str):
        del self._base_heads[binding]

    def get_underlay_model_head(self, name: str) -> ModelHead:
        return self._base_heads[name]

    def make_model_tail(
        self,
        base_model: str,
        adjacent_decoder: Optional[str] = None,
    ) -> str:
        name = ModelDescribe.new(base_model, is_tail=True).__str__()
        if name in self._base_tails:
            return name
        model_info = self._models.models_storage[base_model]
        with torch.device(self.device):
            model_tail = ModelTail(
                config=model_info.config.model_tail_config,
                dtype=torch.float16,
            )
            model_tail.load_from_data_dir(model_info.model_tail_path)
        self._base_tails[name] = model_tail
        return name

    def remove_model_tail(self, binding: str):
        del self._base_tails[binding]

    def get_underlay_model_tail(self, name: str) -> ModelTail:
        return self._base_tails[name]

    def make_model_decoders(self, base_model: str, decoders_range: Tuple[int, int]) -> Any:
        name = ModelDescribe.new(
            base_model, start=decoders_range[0], end=decoders_range[1]).__str__()
        if name in self._base_decoders:
            return self._base_decoders[name]
        print(f"start load decoders: {name}")
        st = time.time()

        model_info = self._models.models_storage[base_model]
        model_config = copy.copy(model_info.config)
        model_config.max_seq_len = self.max_seq_len
        start_layer, end_layer = decoders_range
        with torch.device(self.device):
            decoders: DecodersGroup = make_decoders_from_files(
                implement=self.decoder_implement,
                config=model_config.quant_decoder_config,
                state_files=[
                    model_info.decoder_path(i)
                    for i in range(start_layer, end_layer)
                ],
            )
        self._base_decoders[name] = decoders
        print(f"done load decoders: {name} in {time.time()-st:.2f} seconds")
        return name

    def remove_model_decoders(self, binding: str):
        del self._base_decoders[binding]

    def get_underlay_decoders(self, name: str) -> DecodersGroup:
        return self._base_decoders[name]

    def _is_empty(self) -> bool:
        for _, q in self.all_queues:
            if len(q.list) != 0:
                return False
        return True

    async def _wait(self):
        assert self._wait_operate_fut is None
        fut: asyncio.Future[None] = asyncio.Future()
        self._wait_operate_fut = fut
        await fut

    async def wait_task(self):
        if self._is_empty():
            await self._wait()

    @abstractmethod
    def remove_decoders_infer_binding(self, binding: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove_decoders_train_binding(self, binding: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def swapout(self, target: str, **kwargs: Any) -> None:
        raise NotImplementedError

    def add_barrier(self, context_id: str) -> None:
        self.barrier_maintainer.add_free_op(context_id)

    def remove_barrier(self, context_id: str) -> None:
        self.barrier_maintainer.remove_free_op(context_id)

    @abstractmethod
    def capture_decode_graph(self) -> None:
        raise NotImplementedError
