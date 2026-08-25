from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import extend_asyncio as asyncio

from dealers.common.brain import (
    OptimizerConfig,
    SchedulerConfig,
    UserModelDataInitConfig,
)
from node.agent.agent import AgentNode
from node.agent.protocols import (
    API_AGENT_CANCEL_INFER,
    API_AGENT_CREATE_TRAIN,
    API_AGENT_INFER_EVENT,
    API_AGENT_INSPECT_FASTEST_N_PATH,
    API_AGENT_INSPECT_FASTEST_PATH,
    API_AGENT_REMOVE_TRAIN,
    API_AGENT_START_INFER,
    API_AGENT_TRAIN_STEP,
    CreateTrainRequest,
    InferEvent,
    InspectFastestNPathRequest,
    InspectFastestPathRequest,
    RequestResponse,
    StartInferRequest,
    TrainStepRequest,
    TrainStepResult,
)
from node.base import BaseNode
from node.compute_node.infer_meta import UserModelDataProvider
from node.compute_node.train_meta import PossibleTrainInput
from node.infer.infer_ctrl import InferControl
from node.train.train_ctrl import TrainControl
from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig
from operators.core.sampler.common import SamplingParams
from operators.core.tokenizer.common import SequenceCore


class RemoteRequestError(RuntimeError):
    pass


class RemoteInferControl:
    def __init__(
        self,
        request_node: "RequestNode",
        seq_id: str,
        base_models: List[str],
        seq_core: SequenceCore,
        sampling_params: SamplingParams,
        user_model_data: Optional[Tuple[str, UserModelDataProvider]],
        keypoint_providers: List[Tuple[str, int, int]],
        retrieve: bool,
    ) -> None:
        self.ctrl_id = seq_id
        self.request_node = request_node
        self._base_models = base_models
        self._seq_core = seq_core
        self._sampling_params = sampling_params
        self._user_model_data = user_model_data
        self._keypoint_providers = keypoint_providers
        self._retrieve = retrieve
        self._events: asyncio.Queue[InferEvent] = asyncio.Queue()
        self._started = False
        self._closed = False
        self._detail_trace = False
        self._next_step = 0
        self._expected_input: Optional[int] = None

    @property
    def next_step(self):
        return self._next_step

    def put_event(self, event: InferEvent) -> None:
        if not self._closed:
            self._events.put_nowait(event)

    async def infer_step(
        self,
        inp: Any,
        is_prefill: bool,
        detail_trace: bool = False,
    ) -> Any:
        if self._closed:
            raise RuntimeError(f"inference control {self.ctrl_id!r} is closed")
        if not self._started:
            if not is_prefill:
                raise ValueError("the first remote infer_step must be prefill")
            await self.request_node.ensure_remote_agent()
            self._detail_trace = detail_trace
            self.request_node.base_node.send(
                (
                    self.request_node.remote_agent_node,
                    API_AGENT_START_INFER,
                    StartInferRequest(
                        seq_id=self.ctrl_id,
                        base_models=self._base_models,
                        seq_core=self._seq_core,
                        sampling_params=self._sampling_params,
                        user_model_data=self._user_model_data,
                        keypoint_providers=self._keypoint_providers,
                        retrieve=self._retrieve,
                        inp=inp,
                        is_prefill=True,
                        detail_trace=detail_trace,
                    ),
                )
            )
            self._started = True
        else:
            if is_prefill:
                raise ValueError(
                    "remote inference prefill was already started")
            if detail_trace != self._detail_trace:
                raise ValueError(
                    "detail_trace cannot change during remote inference"
                )
            if (
                self._expected_input is not None
                and isinstance(inp, int)
                and inp != self._expected_input
            ):
                raise ValueError(
                    "remote Agent already continued with the previously "
                    "returned token"
                )

        event = await self._events.get()
        if event.error is not None:
            self._closed = True
            raise RemoteRequestError(event.error)
        if event.step != self._next_step:
            self._closed = True
            raise RuntimeError(
                f"remote inference event out of order: expected "
                f"{self._next_step}, received {event.step}"
            )
        if event.next_token is None:
            self._closed = True
            raise RuntimeError("remote inference event has no token")
        self._next_step += 1
        self._expected_input = event.next_token
        if event.stopped:
            self._closed = True
        return (
            event.stopped,
            event.next_token,
            event.logprobs,
            event.detail_trace,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._started:
            self.request_node.base_node.send(
                (
                    self.request_node.remote_agent_node,
                    API_AGENT_CANCEL_INFER,
                    self.ctrl_id,
                ),
                True,
            )


InferControlLike = Union[InferControl, RemoteInferControl]


class RemoteTrainControl:
    def __init__(
        self,
        request_node: "RequestNode",
        control_id: str,
        base_models: List[str],
        user_model_config: Union[PtuningConfig, LoraConfig],
        optimizer_config: OptimizerConfig,
        scheduler_config: Optional[SchedulerConfig],
        user_model_init_config: UserModelDataInitConfig,
        keypoint_providers: List[Tuple[str, int, int]],
        ckpt_period: int,
    ) -> None:
        self.ctrl_id = control_id
        self.request_node = request_node
        self._create_request = CreateTrainRequest(
            control_id=control_id,
            base_models=base_models,
            user_model_config=user_model_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            user_model_init_config=user_model_init_config,
            keypoint_providers=keypoint_providers,
            ckpt_period=ckpt_period,
        )
        self._create_task: Optional[asyncio.Task[Any]] = None
        self._created = False
        self._closed = False
        self._disconnect_error: Optional[str] = None

    async def _create(self) -> None:
        await self.request_node._request(
            API_AGENT_CREATE_TRAIN,
            self._create_request,
        )
        self._created = True

    async def _ensure_created(self) -> None:
        if self._closed:
            message = self._disconnect_error or (
                f"training control {self.ctrl_id!r} is closed"
            )
            raise RuntimeError(message)
        if self._create_task is None:
            self._create_task = asyncio.create_task(self._create())
        await self._create_task

    async def train_step(
        self,
        req_version: int,
        full_batch_ids: List[str],
        minibatch_id: str,
        sample: PossibleTrainInput,
    ) -> float:
        result = await self.train_step_with_detail(
            req_version=req_version,
            full_batch_ids=full_batch_ids,
            minibatch_id=minibatch_id,
            sample=sample,
            detail_trace=False,
        )
        return result.task.loss

    async def train_step_with_detail(
        self,
        req_version: int,
        full_batch_ids: List[str],
        minibatch_id: str,
        sample: PossibleTrainInput,
        detail_trace: bool = False,
    ) -> TrainStepResult:
        await self._ensure_created()
        result = await self.request_node._request(
            API_AGENT_TRAIN_STEP,
            TrainStepRequest(
                control_id=self.ctrl_id,
                req_version=req_version,
                full_batch_ids=full_batch_ids,
                minibatch_id=minibatch_id,
                sample=sample,
                detail_trace=detail_trace,
            ),
        )
        if not isinstance(result, TrainStepResult):
            raise TypeError(
                f"invalid Agent training result: {type(result).__name__}"
            )
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._created:
            self.request_node.base_node.send(
                (
                    self.request_node.remote_agent_node,
                    API_AGENT_REMOVE_TRAIN,
                    self.ctrl_id,
                ),
                True,
            )
            return
        if self._create_task is None:
            return

        async def remove_after_create() -> None:
            try:
                await self._create_task
            except Exception:
                return
            self.request_node.base_node.send(
                (
                    self.request_node.remote_agent_node,
                    API_AGENT_REMOVE_TRAIN,
                    self.ctrl_id,
                ),
                True,
            )

        asyncio.create_task(remove_after_create())

    def disconnect(self, agent_node: str) -> None:
        self._disconnect_error = f"Agent {agent_node!r} disconnected"
        self._closed = True


TrainControlLike = Union[TrainControl, RemoteTrainControl]


class RequestNode:
    def __init__(
        self,
        base_node: BaseNode,
        agent_node: Optional[str] = None,
        path_search_redundancy: int = 1,
    ) -> None:
        self.base_node = base_node
        self._agent_node = agent_node
        self._local_agent = (
            AgentNode(
                base_node,
                path_search_redundancy=path_search_redundancy,
            )
            if agent_node is None
            else None
        )
        self._remote_infer_controls: Dict[str, RemoteInferControl] = {}
        self._remote_train_controls: Dict[str, RemoteTrainControl] = {}
        if self._local_agent is None:
            if API_AGENT_INFER_EVENT in self.base_node.apis:
                raise ValueError("Agent infer event API is already registered")
            self.base_node.apis[API_AGENT_INFER_EVENT] = self._on_infer_event
            self.base_node.link_change_cbs.append(self._on_link_change)

    @property
    def remote_agent_node(self) -> str:
        assert self._agent_node is not None
        return self._agent_node

    @property
    def is_remote(self) -> bool:
        return self._local_agent is None

    async def ensure_remote_agent(self) -> None:
        if self._agent_node is None:
            raise RuntimeError("RequestNode has no remote Agent")
        if not await self.base_node.try_connect(self._agent_node):
            raise ConnectionError(
                f"cannot connect Request node to Agent {self._agent_node!r}"
            )

    async def _request(self, api: str, payload: Any) -> Any:
        await self.ensure_remote_agent()
        assert self._agent_node is not None
        response = await self.base_node.rpc_with_args(
            node=self._agent_node,
            api=api,
            args=payload,
            T=RequestResponse,
        )
        if response is None:
            raise ConnectionError(f"Agent {self._agent_node!r} disconnected")
        if not response.ok:
            raise RemoteRequestError(response.error or "unknown Agent error")
        return response.value

    async def inspect_fastest_path(
        self,
        type: Literal["train", "infer"],
        base_models: List[str],
        providers: Optional[List[List[str]]] = None,
    ) -> Any:
        if self._local_agent is not None:
            return await self._local_agent.inspect_fastest_path(
                type=type,
                base_models=base_models,
                providers=providers,
            )
        return await self._request(
            API_AGENT_INSPECT_FASTEST_PATH,
            InspectFastestPathRequest(type, base_models, providers),
        )

    async def inspect_fastest_n_path(
        self,
        type: Literal["train", "infer"],
        base_models: List[str],
        allow_reuse: bool,
        max_n: int = 1,
        providers: Optional[List[List[str]]] = None,
    ) -> Any:
        if self._local_agent is not None:
            return await self._local_agent.inspect_fastest_n_path(
                type=type,
                base_models=base_models,
                allow_reuse=allow_reuse,
                max_n=max_n,
                providers=providers,
            )
        return await self._request(
            API_AGENT_INSPECT_FASTEST_N_PATH,
            InspectFastestNPathRequest(
                type,
                base_models,
                allow_reuse,
                max_n,
                providers,
            ),
        )

    def create_infer_control(
        self,
        seq_id: str,
        base_models: List[str],
        seq_core: SequenceCore,
        sampling_params: SamplingParams,
        user_model_data: Optional[Tuple[str, UserModelDataProvider]],
        keypoint_providers: List[Tuple[str, int, int]],
        retrieve: bool,
    ) -> InferControlLike:
        if self._local_agent is not None:
            return self._local_agent.create_infer_control(
                seq_id=seq_id,
                base_models=base_models,
                seq_core=seq_core,
                sampling_params=sampling_params,
                user_model_data=user_model_data,
                keypoint_providers=keypoint_providers,
                retrieve=retrieve,
            )
        if seq_id in self._remote_infer_controls:
            raise ValueError(f"inference control already exists: {seq_id}")
        control = RemoteInferControl(
            request_node=self,
            seq_id=seq_id,
            base_models=base_models,
            seq_core=seq_core,
            sampling_params=sampling_params,
            user_model_data=user_model_data,
            keypoint_providers=keypoint_providers,
            retrieve=retrieve,
        )
        self._remote_infer_controls[seq_id] = control
        return control

    def remove_infer_control(self, seq_id: str) -> None:
        if self._local_agent is not None:
            self._local_agent.remove_infer_control(seq_id)
            return
        control = self._remote_infer_controls.pop(seq_id)
        control.close()

    def create_train_control(
        self,
        user_model_name: str,
        base_models: List[str],
        user_model_config: Union[PtuningConfig, LoraConfig],
        optimizer_config: OptimizerConfig,
        scheduler_config: Optional[SchedulerConfig],
        user_model_init_config: UserModelDataInitConfig,
        keypoint_providers: List[Tuple[str, int, int]],
        ckpt_period: int,
    ) -> TrainControlLike:
        if self._local_agent is not None:
            return self._local_agent.create_train_control(
                user_model_name=user_model_name,
                base_models=base_models,
                user_model_config=user_model_config,
                optimizer_config=optimizer_config,
                scheduler_config=scheduler_config,
                user_model_init_config=user_model_init_config,
                keypoint_providers=keypoint_providers,
                ckpt_period=ckpt_period,
            )
        if user_model_name in self._remote_train_controls:
            raise ValueError(
                f"training control already exists: {user_model_name}"
            )
        control = RemoteTrainControl(
            request_node=self,
            control_id=user_model_name,
            base_models=base_models,
            user_model_config=user_model_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            user_model_init_config=user_model_init_config,
            keypoint_providers=keypoint_providers,
            ckpt_period=ckpt_period,
        )
        self._remote_train_controls[user_model_name] = control
        return control

    def remove_train_control(self, user_model_name: str) -> None:
        if self._local_agent is not None:
            self._local_agent.remove_train_control(user_model_name)
            return
        control = self._remote_train_controls.pop(user_model_name)
        control.close()

    def _on_infer_event(self, source: str, event: InferEvent) -> None:
        if source != self._agent_node:
            self.base_node.logger.warn(
                f"discard infer event from unexpected Agent {source!r}"
            )
            return
        control = self._remote_infer_controls.get(event.seq_id)
        if control is None:
            self.base_node.logger.warn(
                f"discard infer event for unknown control {event.seq_id!r}"
            )
            return
        control.put_event(event)

    def _on_link_change(
        self,
        node: str,
        drop: bool,
        _dsts: set[str],
    ) -> None:
        if not drop or node != self._agent_node:
            return
        for control in self._remote_infer_controls.values():
            control.put_event(
                InferEvent(
                    seq_id=control.ctrl_id,
                    step=control.next_step,
                    error=f"Agent {node!r} disconnected",
                )
            )
        for control in self._remote_train_controls.values():
            control.disconnect(node)
