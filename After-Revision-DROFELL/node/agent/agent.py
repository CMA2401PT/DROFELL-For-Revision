import inspect
from typing import Any, Awaitable, Callable, Dict, Set, Tuple
from uuid import uuid4

import extend_asyncio as asyncio

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
    TrainTaskResult,
)
from node.agent.control_plane import AgentControlPlane
from node.agent.infer import AgentInferController
from node.agent.train import AgentTrainController
from node.base import BaseNode
from node.dealer_node.protocols import (
    API_GroupedInferTask,
    API_InferTask,
    API_NodeSuicide,
    API_TrainTask,
)
from node.train.train_ctrl import TrainControl
from utils.runtime_metrics import record_event
from node.recovery_metrics import new_recovery_id
from node.train_recovery_metrics import new_train_recovery_id


RequestHandler = Callable[[str, Any], Awaitable[Any] | Any]
RemoteInferKey = Tuple[str, str]
RemoteTrainKey = Tuple[str, str]
RemoteTrainStepKey = Tuple[str, str, str]


class AgentNode:
    def __init__(
        self,
        base_node: BaseNode,
        path_search_redundancy: int = 1,
    ) -> None:
        self.base_node = base_node
        self.control_plane = AgentControlPlane(
            base_node=base_node,
            path_search_redundancy=path_search_redundancy,
        )
        self.inspect_fastest_path = self.control_plane.inspect_fastest_path
        self.inspect_fastest_n_path = self.control_plane.inspect_fastest_n_path
        self.infer_controller = AgentInferController(self.control_plane)
        self.create_infer_control = self.infer_controller.create_infer_control
        self.remove_infer_control = self.control_plane.remove_infer_control
        self.train_controller = AgentTrainController(self.control_plane)
        self.create_train_control = self.train_controller.create_train_control
        self.remove_train_control = self.control_plane.remove_train_control
        self._remote_infer_tasks: Dict[RemoteInferKey, asyncio.Task[Any]] = {}
        self._remote_train_controls: Dict[RemoteTrainKey, TrainControl] = {}
        self._remote_train_tasks: Dict[
            RemoteTrainStepKey, asyncio.Task[Any]
        ] = {}
        self.control_plane.base.apis[API_NodeSuicide] = self.on_suicide
        self.control_plane.base.apis[API_InferTask] = (
            self.infer_controller.on_infer_task
        )
        self.control_plane.base.apis[API_GroupedInferTask] = (
            self.infer_controller.on_grouped_infer_task
        )
        self.control_plane.base.apis[API_TrainTask] = (
            self.train_controller.on_train_task
        )
        self.control_plane.base.link_change_cbs.append(self.on_link_change)
        self._register_remote_apis()

    def _register_request_handler(
        self,
        api: str,
        handler: RequestHandler,
    ) -> None:
        if api in self.control_plane.base.apis:
            raise ValueError(f"Agent API is already registered: {api}")

        def receive(source: str, payload: tuple[int, Any]) -> None:
            ticket, request = payload

            async def run() -> None:
                try:
                    value = handler(source, request)
                    if inspect.isawaitable(value):
                        value = await value
                    response = RequestResponse.success(value)
                except Exception as error:
                    response = RequestResponse.failure(error)
                    self.control_plane.logger.error(
                        f"Agent API {api!r} from {source!r} failed: "
                        f"{response.error}"
                    )
                self.base_node.send(
                    (source, "resp", (ticket, response)),
                    True,
                )

            asyncio.create_task(run())

        self.control_plane.base.apis[api] = receive

    def _register_remote_apis(self) -> None:
        async def inspect_fastest_path(
            _source: str,
            request: InspectFastestPathRequest,
        ) -> Any:
            return await self.inspect_fastest_path(
                type=request.type,
                base_models=request.base_models,
                providers=request.providers,
            )

        async def inspect_fastest_n_path(
            _source: str,
            request: InspectFastestNPathRequest,
        ) -> Any:
            return await self.inspect_fastest_n_path(
                type=request.type,
                base_models=request.base_models,
                allow_reuse=request.allow_reuse,
                max_n=request.max_n,
                providers=request.providers,
            )

        self._register_request_handler(
            API_AGENT_INSPECT_FASTEST_PATH,
            inspect_fastest_path,
        )
        self._register_request_handler(
            API_AGENT_INSPECT_FASTEST_N_PATH,
            inspect_fastest_n_path,
        )
        self.control_plane.base.apis[API_AGENT_START_INFER] = (
            self._start_remote_infer
        )
        self.control_plane.base.apis[API_AGENT_CANCEL_INFER] = (
            self._cancel_remote_infer
        )
        self._register_request_handler(
            API_AGENT_CREATE_TRAIN,
            self._create_remote_train,
        )
        self.control_plane.base.apis[API_AGENT_TRAIN_STEP] = (
            self._start_remote_train_step
        )
        self.control_plane.base.apis[API_AGENT_REMOVE_TRAIN] = (
            self._remove_remote_train
        )

    def _send_infer_event(self, target: str, event: InferEvent) -> None:
        self.base_node.send((target, API_AGENT_INFER_EVENT, event), True)

    def _start_remote_infer(
        self,
        source: str,
        request: StartInferRequest,
    ) -> None:
        key = (source, request.seq_id)
        if key in self._remote_infer_tasks:
            self._send_infer_event(
                source,
                InferEvent(
                    seq_id=request.seq_id,
                    step=0,
                    error=f"inference session already exists: {request.seq_id}",
                ),
            )
            return

        async def run() -> None:
            control = None

            if not request.is_prefill:
                raise ValueError(
                    "the first remote infer_step must be prefill"
                )
            control_id = f"{request.seq_id}-{uuid4().hex[:8]}"
            if self.control_plane.infer_controls.get(control_id) is not None:
                raise ValueError(
                    f"Agent inference control already exists: {control_id}"
                )
            control = self.create_infer_control(
                seq_id=control_id,
                base_models=request.base_models,
                seq_core=request.seq_core,
                sampling_params=request.sampling_params,
                user_model_data=request.user_model_data,
                keypoint_providers=request.keypoint_providers,
                retrieve=request.retrieve,
            )
            inp = request.inp
            is_prefill = True
            step = 0
            while True:
                stopped, next_token, logprobs, detail_trace = (
                    await control.infer_step(
                        inp=inp,
                        is_prefill=is_prefill,
                        detail_trace=request.detail_trace,
                    )
                )
                self._send_infer_event(
                    source,
                    InferEvent(
                        seq_id=request.seq_id,
                        step=step,
                        stopped=stopped,
                        next_token=next_token,
                        logprobs=logprobs,
                        detail_trace=detail_trace,
                    ),
                )
                if stopped:
                    break
                inp = next_token
                is_prefill = False
                step += 1
            self._remote_infer_tasks.pop(key, None)
            if (
                control is not None
                and self.control_plane.infer_controls.get(control.ctrl_id)
                is not None
            ):
                self.remove_infer_control(control.ctrl_id)

        task = asyncio.create_task(run())
        self._remote_infer_tasks[key] = task

    def _cancel_remote_infer(self, source: str, seq_id: str) -> None:
        task = self._remote_infer_tasks.get((source, seq_id))
        if task is not None:
            task.cancel()

    def _create_remote_train(
        self,
        source: str,
        request: CreateTrainRequest,
    ) -> None:
        key = (source, request.control_id)
        if key in self._remote_train_controls:
            raise ValueError(
                f"training session already exists: {request.control_id}"
            )
        internal_id = f"{request.control_id}-{uuid4().hex[:8]}"
        control = self.create_train_control(
            user_model_name=internal_id,
            base_models=request.base_models,
            user_model_config=request.user_model_config,
            optimizer_config=request.optimizer_config,
            scheduler_config=request.scheduler_config,
            user_model_init_config=request.user_model_init_config,
            keypoint_providers=request.keypoint_providers,
            ckpt_period=request.ckpt_period,
        )
        self._remote_train_controls[key] = control

    def _start_remote_train_step(
        self,
        source: str,
        payload: tuple[int, TrainStepRequest],
    ) -> None:
        ticket, request = payload
        control_key = (source, request.control_id)
        step_key = (source, request.control_id, request.minibatch_id)

        async def run() -> None:
            try:
                control = self._remote_train_controls.get(control_key)
                if control is None:
                    raise KeyError(
                        f"unknown training session: {request.control_id}"
                    )
                result = await control.train_step_with_detail(
                    req_version=request.req_version,
                    full_batch_ids=request.full_batch_ids,
                    minibatch_id=request.minibatch_id,
                    sample=request.sample,
                    detail_trace=request.detail_trace,
                )
                loss = result.task.loss
                if loss is None:
                    raise RuntimeError(
                        f"training step {request.minibatch_id!r} has no loss"
                    )
                response = RequestResponse.success(
                    TrainStepResult(
                        task=TrainTaskResult(loss=loss),
                        detail_trace=result.detail_trace,
                    )
                )
            except Exception as error:
                response = RequestResponse.failure(error)
                self.control_plane.logger.error(
                    f"Agent training step from {source!r} failed: "
                    f"{response.error}"
                )
            finally:
                self._remote_train_tasks.pop(step_key, None)
            self.base_node.send(
                (source, "resp", (ticket, response)),
                True,
            )

        if step_key in self._remote_train_tasks:
            response = RequestResponse.failure(
                ValueError(
                    f"training minibatch already pending: "
                    f"{request.minibatch_id}"
                )
            )
            self.base_node.send(
                (source, "resp", (ticket, response)),
                True,
            )
            return
        task = asyncio.create_task(run())
        self._remote_train_tasks[step_key] = task

    def _remove_remote_train(self, source: str, control_id: str) -> None:
        key = (source, control_id)
        for step_key, task in tuple(self._remote_train_tasks.items()):
            if step_key[:2] == key:
                task.cancel()
        control = self._remote_train_controls.pop(key, None)
        if control is not None:
            self.remove_train_control(control.ctrl_id)

    def on_suicide(self, node: str, node_name: str) -> None:
        record_event(
            "failure_detected",
            dealer_node=node,
            compute_node=node_name,
            handler_node=self.base_node.node_name,
            recovery_scope="agent",
        )
        target = self.control_plane.train_controls.get_handle(node_name)
        if target is not None:
            self.control_plane.logger.warn(
                f"{node_name} @ {node} suicide, self healing"
            )
            train_control, handle = target
            recovery_id = new_train_recovery_id()
            record_event(
                "shadow_state_snapshot",
                recovery_id=recovery_id,
                state_type="train",
                control_id=train_control.ctrl_id,
                compute_node=node_name,
                metrics=handle.instance_data.metrics_snapshot(),
            )
            self.train_controller.recover_handles_by_train_node(
                train_control=train_control,
                effected_handle=handle,
                report_dealer_node=node,
                recovery_id=recovery_id,
            )
            return
        target = self.control_plane.infer_controls.get_handle(node_name)
        if target is not None:
            self.control_plane.logger.warn(
                f"{node_name} @ {node} suicide, self healing"
            )
            infer_control, handle = target
            recovery_id = new_recovery_id()
            record_event(
                "shadow_state_snapshot",
                recovery_id=recovery_id,
                state_type="infer",
                control_id=infer_control.ctrl_id,
                compute_node=node_name,
                metrics=handle.instance_data.metrics_snapshot(),
            )
            self.infer_controller.recover_handles_by_infer_node(
                infer_control=infer_control,
                effected_handle=handle,
                report_dealer_node=node,
                recovery_id=recovery_id,
            )

    def on_link_change(
        self,
        node: str,
        drop: bool,
        dsts: Set[str],
    ) -> None:
        if not drop:
            return
        for (source, _), task in tuple(self._remote_infer_tasks.items()):
            if source == node:
                task.cancel()
        remote_train_ids = [
            control_id
            for source, control_id in self._remote_train_controls
            if source == node
        ]
        for control_id in remote_train_ids:
            self._remove_remote_train(node, control_id)
        targets = self.control_plane.train_controls.get_handle_by_dealer(node)
        if targets is not None:
            for train_control, handles in targets:
                self.control_plane.logger.warn(
                    f"{node} dropped, {train_control.ctrl_id} effected, "
                    "self healing"
                )
                self.train_controller.recover_handles_by_dealer_node(
                    train_control=train_control,
                    effected_handles=handles,
                    dropped_node=node,
                )

        targets = self.control_plane.infer_controls.get_handle_by_dealer(node)
        if targets is not None:
            for infer_control, handles in targets:
                self.control_plane.logger.warn(
                    f"{node} dropped, {infer_control.ctrl_id} effected, "
                    "self healing"
                )
                self.infer_controller.recover_handles_by_dealer_node(
                    infer_control=infer_control,
                    effected_handles=handles,
                    dropped_node=node,
                )
