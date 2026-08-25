from typing import List, Optional, Tuple, Union
from uuid import uuid4

import extend_asyncio as asyncio
from dealers.common.brain import (OptimizerConfig, SchedulerConfig,
                                  UserModelDataInitConfig)
from .control_plane import AgentControlPlane
from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig

from ..compute_node.train_meta import TrainTask
from ..dealer_node.protocols import (API_TrainTask,
                                     MultiPathTrainTask)
from ..dealer_node.train_handle import TrainNodeHandle
from ..train.train_ctrl import TrainControl, TrainControlBase
from node.train_recovery_metrics import (
    mark_train_recovery,
    new_train_recovery_id,
    record_train_output,
)
from utils.runtime_metrics import record_event


class AgentTrainController:
    def __init__(
        self,
        base_node: AgentControlPlane,
    ) -> None:
        self.base = base_node
        # self.train_controls = TrainControlNode(self.base)
        # self.remove_train_control = self.train_controls.__delitem__
        # if API_NodeSuicide not in self.base.apis:
        #     self.base.apis[API_NodeSuicide] = self.on_node_suicide
        # else:
        #     orig_suicide = self.base.apis[API_NodeSuicide]

        #     def on_suicide(node: str, train_node_name: str):
        #         orig_suicide(node, train_node_name)
        #         self.on_node_suicide(node, train_node_name)
        #     self.base.apis[API_NodeSuicide] = on_suicide
        # self.base.apis[API_TrainTask] = self.on_train_task

        # self.path_search_redundancy = 1
        # self.base.link_change_cbs.append(self.on_link_change)
        # self._extra_info: Optional[ExtraInfo] = None

    def create_train_control(
        self,
        user_model_name: str, base_models: List[str],
        user_model_config: Union[PtuningConfig, LoraConfig],
        optimizer_config: OptimizerConfig,
        scheduler_config: Optional[SchedulerConfig],
        user_model_init_config: UserModelDataInitConfig,
        keypoint_providers: List[Tuple[str, int, int]],
        ckpt_period: int,

    ):
        ctrl = TrainControl(
            ctrl_host_name=self.base.node_name,
            ctrl_id=user_model_name,
            base_models=base_models,
            send=self.base.train_send,
            user_model_config=user_model_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            user_model_data_provider=(
                lambda _source, _destination: user_model_init_config
            ),
            user_model_prefix=user_model_name,
            is_root=True
        ).init_target_nodes(
            providers=keypoint_providers,
            ckpt_period=ckpt_period
        )
        self.base.train_controls[user_model_name] = ctrl
        return ctrl

    def recover_handles(
        self,
        # providers: Dict[str, List[str]],
        replacements: List[Tuple[
            TrainNodeHandle,
            Optional[List[Tuple[str, int, int]]]
        ]],
        train_control: TrainControlBase,
        recovery_id: str,
    ):
        # this replace breaker, as replacement is finished without async
        # thus node data will not change in replacement
        recover_tasks: List[Tuple[str, TrainTask]] = []
        record_event(
            "state_rebuild_start",
            recovery_id=recovery_id,
            state_type="train",
            control_id=train_control.ctrl_id,
            handle_count=len(replacements),
        )
        # Recovery replay counts are needed by the replacement Dealer at node
        # creation time. Compute them before creating replacement handles.
        recovery_parts = []
        for handle, new_edges in replacements:
            per_edge = []
            if new_edges is not None:
                for dealer_node, start, count in new_edges:
                    new_models = handle.models[start:start + count]
                    new_instance_data = handle.instance_data.get_part(new_models)
                    new_init_cfg, recover_extra = new_instance_data.gen_init_cfg()
                    per_edge.append((dealer_node, new_instance_data, new_init_cfg, recover_extra))
            recovery_parts.append((handle, new_edges, per_edge))

        for handle, new_edges, per_edge in recovery_parts:
            # base_models = [m for (m, u) in handle.models]
            # providers_list: List[List[str]] = [providers[m] for (m, u) in handle.models]
            # # new_path = paths_to_edge(gen_lowest_energy_paths(providers_list)[0])
            # self.base.logger.info(f"replacing: {handle}")
            # inspected = await self.inspect_fastest_path(base_models, providers_list)
            # if inspected is None:
            #     self.base.logger.warn(f"cannot find replacement of {handle} for train ctrl: {train_control.ctrl_id}, reduce reduncy")
            #     continue
            # new_edges = list(paths_to_edge(inspected.best))
            if new_edges is not None:
                new_handles: List[TrainNodeHandle] = []
                for dealer_node, new_instance_data, new_init_cfg, recover_extra in per_edge:
                    new_handle = train_control.init_train_node(
                        dealer=dealer_node,
                        ckpt_period=handle.ckpt_period,
                        cfg=new_init_cfg,
                        recovery_id=recovery_id,
                        recovery_task_count=len(recover_extra),
                    )
                    new_handle.instance_data = new_instance_data
                    new_handles.append(new_handle)
                    self.base.train_controls.reg_new_handle(new_handle, train_control)
                    for task in recover_extra:
                        recover_tasks.append((dealer_node, task))
                new_handles_info = "\n  ".join(str(h) for h in new_handles)
                self.base.logger.info(f"find replacement of {handle} for train ctrl: {train_control.ctrl_id}:\n  "+new_handles_info)
            else:
                self.base.logger.error(f"cannot find replacement of {handle} for train ctrl: {train_control.ctrl_id}, reduce reduncy")
            # do remove
            self.base.train_controls.remove_handle(handle.train_node_name)
            handle.drop(False)

        assert train_control.is_contigious(), f"cannot heal train path, current train path is: \n{train_control}"
        record_event(
            "state_rebuild_end",
            recovery_id=recovery_id,
            state_type="train",
            control_id=train_control.ctrl_id,
            replay_task_count=len(recover_tasks),
            replay_minibatch_count=len({
                task.minibatch_id for _, task in recover_tasks
            }),
        )
        record_event(
            "replacement_graph_installed",
            recovery_id=recovery_id,
            state_type="train",
            control_id=train_control.ctrl_id,
            replay_task_count=len(recover_tasks),
        )
        mark_train_recovery(
            train_control.ctrl_id,
            recovery_id,
            (task.minibatch_id for _, task in recover_tasks),
        )
        if recover_tasks:
            record_event(
                "replay_start",
                recovery_id=recovery_id,
                state_type="train",
                control_id=train_control.ctrl_id,
                replay_task_count=len(recover_tasks),
            )
        for task_index, (dealer_node, task) in enumerate(recover_tasks):
            packet_id = uuid4().hex
            wrapped_task = MultiPathTrainTask(
                task=task,
                resource_loc=train_control.res_location,
                trace=[],
                trace_invalid=True,
                control_id=train_control.ctrl_id,
                packet_id=packet_id,
                origin="recovery_replay",
                recovery_id=recovery_id,
                recovery_task_index=task_index,
                recovery_task_count=len(recover_tasks),
            )
            print(f"send recover delta: {wrapped_task} -> {dealer_node}")
            record_event(
                "recovery_packet_send_start",
                recovery_id=recovery_id,
                state_type="train",
                control_id=train_control.ctrl_id,
                packet_id=packet_id,
                packet_kind="recovery_replay",
                task_index=task_index,
                task_count=len(recover_tasks),
                minibatch_id=task.minibatch_id,
                request_version=task.request_version,
                target_dealer=dealer_node,
            )
            self.base.send((dealer_node, API_TrainTask, wrapped_task))
            record_event(
                "recovery_packet_enqueue_accepted",
                recovery_id=recovery_id,
                state_type="train",
                control_id=train_control.ctrl_id,
                packet_id=packet_id,
                packet_kind="recovery_replay",
                task_index=task_index,
                task_count=len(recover_tasks),
                minibatch_id=task.minibatch_id,
                request_version=task.request_version,
                target_dealer=dealer_node,
            )
        record_event(
            "replay_send_end",
            recovery_id=recovery_id,
            state_type="train",
            control_id=train_control.ctrl_id,
            replay_task_count=len(recover_tasks),
        )
        record_event(
            "promise_handoff_committed",
            recovery_id=recovery_id,
            state_type="train",
            control_id=train_control.ctrl_id,
            required_packet_count=len(recover_tasks) + sum(
                len(per_edge) for _, _, per_edge in recovery_parts
            ),
            replay_task_count=len(recover_tasks),
        )
        train_control.latest_handoff_id = recovery_id
        # self.base.logger.error("HEADLING COMPLETE!")
        # train_control.breaker_disable()

    def on_train_task(self, node: str, train_task: MultiPathTrainTask):
        # self.base.logger.error(f"RECV: {train_task}")
        control = self.base.train_controls.get(train_task.control_id or "")
        accepted_output = bool(
            train_task.task.navi.finished
            and control is not None
            and train_task.unique_id in control.result_waitor
        )
        record_event(
            "agent_train_packet_receive",
            recovery_id=train_task.recovery_id,
            logical_handoff_id=train_task.logical_handoff_id,
            control_id=train_task.control_id,
            dealer_node=node,
            packet_id=train_task.packet_id,
            origin=train_task.origin,
            minibatch_id=train_task.task.minibatch_id,
            request_version=train_task.task.request_version,
            navigation_finished=train_task.task.navi.finished,
        )
        train_task.record_step("root")
        if train_task.task.navi.finished:
            record_event(
                "train_service_output_visible",
                recovery_id=train_task.recovery_id,
                logical_handoff_id=train_task.logical_handoff_id,
                control_id=train_task.control_id,
                dealer_node=node,
                packet_id=train_task.packet_id,
                origin=train_task.origin,
                minibatch_id=train_task.task.minibatch_id,
                request_version=train_task.task.request_version,
                loss=train_task.task.loss,
                accepted=accepted_output,
            )
            record_train_output(
                train_task.task.navi.dir_user_model[0].split(":", 1)[0],
                train_task.task.minibatch_id,
                train_task.task.request_version,
                train_task.trace_invalid,
            )
        self.base.train_controls.on_train_task(node, train_task)

    def recover_handles_by_train_node(
        self,
        train_control: TrainControlBase,
        effected_handle: TrainNodeHandle,
        report_dealer_node: str,
        recovery_id: Optional[str] = None,
    ):
        async def coro():
            resolved_recovery_id = (
                recovery_id or new_train_recovery_id()
            )
            record_event(
                "replacement_lookup_start",
                recovery_id=resolved_recovery_id,
                state_type="train",
                control_id=train_control.ctrl_id,
                failed_dealer=report_dealer_node,
                models=effected_handle.base_models,
            )
            replacement = await self.base.lookup_replacement("train", effected_handle.base_models, {report_dealer_node})
            if replacement is None:
                self.base.logger.warn(f"no replacement for {effected_handle.base_models} found, ignore blacklists")
                replacement = await self.base.lookup_replacement("train", effected_handle.base_models, None)
            record_event(
                "replacement_lookup_end",
                recovery_id=resolved_recovery_id,
                state_type="train",
                control_id=train_control.ctrl_id,
                replacement=replacement,
            )
            self.recover_handles([(
                effected_handle,
                # avoid using dealer node which just report sucide, as the resource could be insufficient
                replacement
            )], train_control, resolved_recovery_id)
        asyncio.create_task(coro())

    # def on_node_suicide(self, dealer_node: str, node_name: str):
    #     target = self.base.train_controls.get_handle(node_name)
    #     if target is None:
    #         return
    #     self.base.logger.warn(f"{node_name} @ {dealer_node} suicide, self healing")
    #     train_control, handle = target
    #     self.recover_handles_by_train_node(train_control=train_control, effected_handle=handle, report_dealer_node=dealer_node)

    def recover_handles_by_dealer_node(self, train_control: TrainControlBase, effected_handles: List[TrainNodeHandle], dropped_node: str):
        async def coro():
            recovery_id = new_train_recovery_id()
            replacements: List[Tuple[TrainNodeHandle, Optional[List[Tuple[str, int, int]]]]] = []
            for handle in effected_handles:
                replacement = await self.base.lookup_replacement("train", handle.base_models, {dropped_node})
                if replacement is None:
                    self.base.logger.warn(f"no replacement for {handle.base_models} found, ignore blacklists")
                    replacement = await self.base.lookup_replacement("train", handle.base_models, None)
                replacements.append((handle, replacement))

            self.recover_handles(replacements, train_control, recovery_id)
        asyncio.create_task(coro())

    # def on_link_change(self, node: str, drop: bool, dsts: Set[str]):
    #     if not drop:
    #         return
    #     targets = self.base.train_controls.get_handle_by_dealer(node)
    #     if targets is None:
    #         return
    #     # self.base.logger.error("LINK CHANGED! HEALING")
    #     for train_control, handles in targets:
    #         # train_control.breaker_enable()
    #         self.base.logger.warn(f"{node} dropped, {train_control.ctrl_id} effected, self healing")
    #         self.recover_handles_by_dealer_node(train_control=train_control, effected_handles=handles, dropped_node=node)
