from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict, Tuple
from dataclasses import dataclass
from dataset.common import ProcessedRecord
from operators.core.sampler.common import SamplingParams
from operators.core.tokenizer.common import SequenceCore, SequenceLengthData
from .base_types import *
from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal, Optional


@dataclass
class UserModelDataInitConfig:
    init_method: Literal["none", "from_ckpt", "xavier_uniform_"] = "xavier_uniform_"
    # same keys as decoders' count
    ckpt_bytes: Optional[Dict[int, bytes]] = None


@dataclass
class OptimizerConfig:
    optimizer_name: str = "AdamW"
    lr: float = 1e-2
    eps: float = 5e-3


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
    hidden_states: Optional[torch.Tensor]
    cpu_pos_idx: Optional[torch.Tensor]
    gpu_pos_idx: Optional[torch.Tensor]
    input_lens: List[int]
    label_ids_tensor: torch.Tensor


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


class HeadInferDealer(ABC):
    @abstractmethod
    async def infer_tensor_to_embedding_tokens(self, tokens: List[int], cpu_out: bool, is_prefill: bool, seq_id: str, **kwargs: Any) -> DeviceTensorCarrier:
        raise NotImplementedError


class HeadTrainDealer(ABC):
    @abstractmethod
    async def train_tensor_to_embedding_tokens(self, batch_id: SampleBatchID, hidden_states: torch.Tensor, pos_idx: torch.Tensor, dtype: torch.dtype, cpu_out: bool, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError

    async def forward_with_wrap(self, batch_id: SampleBatchID, sample: Sample, cpu_out: bool, **kwargs: Any):
        '''
        helper function which simplify input/output of head/decoders dealer
        '''
        dtype = torch.bfloat16 if sample.is_bfloat16 else torch.float16
        cpu_pos_idx = sample.position_ids_tensor
        gpu_pos_idx: Optional[torch.Tensor] = None
        hidden_states, gpu_pos_idx = await self.train_tensor_to_embedding_tokens(
            batch_id=batch_id,
            hidden_states=sample.input_ids_tensor,
            pos_idx=sample.position_ids_tensor,
            dtype=dtype,
            cpu_out=cpu_out,
            **kwargs
        )
        meta = TrainForwardMeta(
            sample_batch_id=batch_id,
            hidden_states=hidden_states,
            cpu_pos_idx=cpu_pos_idx,
            gpu_pos_idx=gpu_pos_idx,
            input_lens=sample.input_lens,
            label_ids_tensor=sample.label_ids_tensor
        )
        return meta


class TailInferBinding(ABC):
    @abstractmethod
    async def hidden_state_to_sampling_output(self, carrier: DeviceTensorCarrier, cpu_in: bool, seq_id: str, **kwargs: Any) -> Tuple[int, Dict[int, float]]:
        raise NotImplementedError


class TailInferRoot(ABC):
    @abstractmethod
    def create_binding(self, seq_core: SequenceCore, sampling_params: SamplingParams) -> TailInferBinding:
        raise NotImplementedError


class TailTrainBinding:
    @abstractmethod
    async def compute_logits_loss_and_grad(self, batch_id: SampleBatchID, carrier: DeviceTensorCarrier, label_ids: torch.Tensor, cpu_in_out: bool, **kwargs: Any) -> Tuple[float, DeviceTensorCarrier]:
        raise NotImplementedError


class DecodersInferBinding(ABC):
    @abstractmethod
    def allocate_slots(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def do_generating(self, carrier: DeviceTensorCarrier, cpu_in: bool, cpu_out: bool, tail_in: bool, seq_id: str, **kwargs: Any) -> DeviceTensorCarrier:
        raise NotImplementedError

    @abstractmethod
    async def do_prefill(self, carrier: DeviceTensorCarrier, cpu_in: bool, cpu_out: bool, tail_in: bool, seq_id: str, **kwargs: Any) -> DeviceTensorCarrier:
        raise NotImplementedError

    @abstractmethod
    async def put_blocks(self, seq_blocks: List[int], kv_cache: List[Tuple[List[int], List[int]]]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_blocks(self,  seq_blocks: List[int], kv_cache: List[Tuple[List[int], List[int]]]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_blocks_with_create(self, seq_blocks: List[int]) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError


class DecodersInferRoot(ABC):
    @property
    @abstractmethod
    def num_layers(self) -> int:
        raise NotImplementedError

    @abstractmethod
    async def put_ptuning(self, kv_cache: List[Tuple[List[int], List[int]]]) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def create_binding(self,
                       seq_length_data: SequenceLengthData,
                       ptuning_blocks: Optional[List[int]],
                       ) -> DecodersInferBinding:
        raise NotImplementedError


class DecodersTrainBinding(ABC):
    @abstractmethod
    async def forward(self, meta: TrainForwardMeta, cpu_in: bool, cpu_out: bool, is_last: bool, is_first: bool, **kwargs: Any) -> Optional[DeviceTensorCarrier]:
        raise NotImplementedError

    @abstractmethod
    async def backward(self, forward_ctx_id: SampleBatchID, grad: DeviceTensorCarrier, cpu_in: bool, cpu_out: bool, has_next: bool, **kwargs: Any) -> Optional[DeviceTensorCarrier]:
        raise NotImplementedError

    @abstractmethod
    async def weight_update(self, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    async def dump_train_states(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def load_train_states(self, states: Dict[str, Any]) -> None:
        raise NotImplementedError
