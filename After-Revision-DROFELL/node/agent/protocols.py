from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple, Union

from dealers.common.brain import (
    OptimizerConfig,
    SchedulerConfig,
    UserModelDataInitConfig,
)
from node.compute_node.infer_meta import UserModelDataProvider
from node.compute_node.train_meta import PossibleTrainInput
from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig
from operators.core.sampler.common import SamplingParams
from operators.core.tokenizer.common import SequenceCore


API_AGENT_INSPECT_FASTEST_PATH = "agent.inspect_fastest_path"
API_AGENT_INSPECT_FASTEST_N_PATH = "agent.inspect_fastest_n_path"
API_AGENT_START_INFER = "agent.start_infer"
API_AGENT_CANCEL_INFER = "agent.cancel_infer"
API_AGENT_INFER_EVENT = "agent.infer_event"
API_AGENT_CREATE_TRAIN = "agent.create_train"
API_AGENT_TRAIN_STEP = "agent.train_step"
API_AGENT_REMOVE_TRAIN = "agent.remove_train"


@dataclass(frozen=True)
class RequestResponse:
    ok: bool
    value: Any = None
    error: Optional[str] = None

    @classmethod
    def success(cls, value: Any) -> "RequestResponse":
        return cls(ok=True, value=value)

    @classmethod
    def failure(cls, error: BaseException) -> "RequestResponse":
        return cls(
            ok=False,
            error=f"{type(error).__name__}: {error}",
        )


@dataclass(frozen=True)
class InspectFastestPathRequest:
    type: Literal["train", "infer"]
    base_models: List[str]
    providers: Optional[List[List[str]]]


@dataclass(frozen=True)
class InspectFastestNPathRequest:
    type: Literal["train", "infer"]
    base_models: List[str]
    allow_reuse: bool
    max_n: int
    providers: Optional[List[List[str]]]


@dataclass(frozen=True)
class StartInferRequest:
    seq_id: str
    base_models: List[str]
    seq_core: SequenceCore
    sampling_params: SamplingParams
    user_model_data: Optional[Tuple[str, UserModelDataProvider]]
    keypoint_providers: List[Tuple[str, int, int]]
    retrieve: bool
    inp: Any
    is_prefill: bool
    detail_trace: bool


@dataclass(frozen=True)
class InferEvent:
    seq_id: str
    step: int
    stopped: bool = False
    next_token: Optional[int] = None
    logprobs: Any = None
    detail_trace: Any = None
    error: Optional[str] = None


@dataclass(frozen=True)
class CreateTrainRequest:
    control_id: str
    base_models: List[str]
    user_model_config: Union[PtuningConfig, LoraConfig]
    optimizer_config: OptimizerConfig
    scheduler_config: Optional[SchedulerConfig]
    user_model_init_config: UserModelDataInitConfig
    keypoint_providers: List[Tuple[str, int, int]]
    ckpt_period: int


@dataclass(frozen=True)
class TrainStepRequest:
    control_id: str
    req_version: int
    full_batch_ids: List[str]
    minibatch_id: str
    sample: PossibleTrainInput
    detail_trace: bool


@dataclass(frozen=True)
class TrainTaskResult:
    loss: float


@dataclass(frozen=True)
class TrainStepResult:
    task: TrainTaskResult
    detail_trace: Any = None
