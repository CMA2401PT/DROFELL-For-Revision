import os
import json
from typing import List, Optional, Tuple
from uuid import uuid4
from .control_plane import AgentControlPlane
from operators.core.sampler.common import SamplingParams
from operators.core.tokenizer.common import SequenceCore
from ..dealer_node.infer_handle import InferNodeHandle
from ..compute_node.infer_meta import InferTask, UserModelDataProvider
from ..infer.infer_ctrl import InferControl, InferControlBase
from ..dealer_node.protocols import API_InferTask, GroupedInferTask, MultiPathInferTask
from node.recovery_metrics import (
    mark_pending,
    new_recovery_id,
    record_first_output,
    record_intermediate_arrival,
)
from utils.runtime_metrics import record_event
from utils.size_metrics import retained_bytes
from .. import flags

import extend_asyncio as asyncio


class AgentInferController:
    def __init__(
        self,
        base_node: AgentControlPlane,
    ) -> None:
        self.base = base_node
        drop_step = os.getenv("RECOVER_MULTI_ACTIVE_WINNER_DROP_STEP")
        self._active_winner_drop_step = (
            int(drop_step) if drop_step is not None else None
        )
        self._active_winner_drop_sent: set[tuple[str, int]] = set()
        self._active_winner_drop_relay = os.getenv(
            "RECOVER_MULTI_ACTIVE_WINNER_DROP_RELAY", ""
        )
        # self.infer_controls = InferControlNode(self.base)
        # self.remove_infer_control = self.infer_controls.__delitem__
        # if API_NodeSuicide not in self.base.apis:
        #     self.base.apis[API_NodeSuicide] = self.on_node_suicide
        # else:
        #     orig_suicide = self.base.apis[API_NodeSuicide]

        #     def on_suicide(node: str, infer_node_name: str):
        #         orig_suicide(node, infer_node_name)
        #         self.on_node_suicide(node, infer_node_name)
        #     self.base.apis[API_NodeSuicide] = on_suicide
        # self.base.apis[API_InferTask] = self.on_infer_task
        # self.base.apis[API_GroupedInferTask] = self.on_grouped_infer_task

        # self.path_search_redundancy = 1
        # self.base.link_change_cbs.append(self.on_link_change)
        # self._extra_info: Optional[ExtraInfo] = None

    def create_infer_control(
        self,
        seq_id: str, base_models: List[str],
        seq_core: SequenceCore, sampling_params: SamplingParams,
        user_model_data: Optional[Tuple[str, UserModelDataProvider]],
        keypoint_providers: List[Tuple[str, int, int]],
        retrieve: bool,

    ):
        ctrl = InferControl(
            ctrl_host_name=self.base.node_name,
            ctrl_id=seq_id,
            base_models=base_models,
            send=self.base.infer_send,
            send_infer_task=self.base.infer_batch_send,
            seq_core=seq_core,
            sampling_params=sampling_params,
            user_model_data=user_model_data,
            is_root=True
        ).init_target_nodes(
            providers=keypoint_providers,
            retrieve=retrieve
        )
        self.base.infer_controls[seq_id] = ctrl
        return ctrl

    # async def recover_handles_by_infer_node(self, report_dealer_node: str, infer_node: str):
    #     target = self.infer_controls.get_and_remove_handle(infer_node)
    #     if target is None:
    #         return
    #     infer_control, handle = target
    #     base_models = list(set(m for (m, u) in handle.models))
    #     # avoid using dealer node which just report sucide, as the resource could be insufficient
    #     _providers = await self.req_provider(base_models, {report_dealer_node})
    #     if _providers is None:
    #         self.base.logger.info(f"replacing: {handle} fail, cannot find any possible path")
    #         assert False, f"replacing: {handle} fail, cannot find any possible path"
    #     await self.recover_handles({k: ps for k, ps in zip(base_models, _providers)}, [handle], infer_control)

    # async def recover_handles_by_dealer_node(self, infer_control: InferControlBase, dropped_handles: List[InferNodeHandle], dropped_node: str):
    #     all_base_models: List[str] = list(set(m for handle in dropped_handles for (m, u) in handle.models))
    #     _providers = await self.req_provider(all_base_models, {dropped_node})
    #     if _providers is None:
    #         self.base.logger.info(f"replacing: {dropped_handles} fail, cannot find any possible path")
    #         assert False, f"replacing: {dropped_handles} fail, cannot find any possible path"
    #     await self.recover_handles({k: ps for k, ps in zip(all_base_models, _providers)}, dropped_handles, infer_control)

    def recover_handles(
        self,
        # providers: Dict[str, List[str]],
        handles: List[Tuple[
            InferNodeHandle,
            Optional[List[Tuple[str, int, int]]]
        ]],
        infer_control: InferControlBase,
        recovery_id: str,
    ):
        if not self.base.infer_controls.is_registered(infer_control):
            record_event(
                "recovery_cancelled_control_retired",
                recovery_id=recovery_id,
                control_id=infer_control.ctrl_id,
                recovery_scope="agent",
            )
            return
        # this replace breaker, as replacement is finished without async
        # thus node data will not change in replacement
        recover_tasks: List[Tuple[str, InferTask]] = []
        confirmed_output_token_ids = (
            infer_control.seq_core.output_token_ids
            if isinstance(infer_control, InferControl)
            and isinstance(infer_control.seq_core, SequenceCore)
            else None
        )
        record_event(
            "state_rebuild_start",
            recovery_id=recovery_id,
            control_id=infer_control.ctrl_id,
            handle_count=len(handles),
        )
        for handle, new_edges in handles:
            if new_edges is not None:
                # base_models = [m for (m, u) in handle.models]
                # providers_list: List[List[str]] = [providers[m] for (m, u) in handle.models]
                # # new_path = paths_to_edge(gen_lowest_energy_paths(providers_list)[0])
                # self.base.logger.info(f"replacing: {handle}")
                # inspected = await self.inspect_fastest_path(base_models, providers_list)
                # if inspected is None:
                #     self.base.logger.warn(f"cannot find replacement of {handle} for infer ctrl: {infer_control.ctrl_id}, reduce reduncy")
                #     continue
                # new_edges = list(paths_to_edge(inspected.best))
                new_handles: List[InferNodeHandle] = []
                for (dealer_node, start, count) in new_edges:

                    new_models = handle.models[start:start+count]
                    new_instace_data = handle.instance_data.get_part(new_models)
                    record_event(
                        "recovery_shadow_snapshot",
                        recovery_id=recovery_id,
                        control_id=infer_control.ctrl_id,
                        failed_compute_node=handle.infer_node_name,
                        failed_dealer=handle.dealer_node_name,
                        replacement_dealer=dealer_node,
                        models=new_models,
                        **new_instace_data.recovery_snapshot(),
                    )
                    new_init_cfg, recover_extra = new_instace_data.gen_init_cfg(
                        confirmed_output_token_ids=confirmed_output_token_ids,
                    )
                    if flags.RECOVERY_REPLAY == "chunk":
                        recover_extra = (
                            new_instace_data.coalesce_recovery_tasks(
                                recover_extra,
                                confirmed_output_token_ids=(
                                    confirmed_output_token_ids),
                            )
                        )

                    new_handle = infer_control.init_infer_node(
                        dealer=dealer_node,
                        retrieve=handle.retrieve,
                        cfg=new_init_cfg,
                        recovery_id=recovery_id,
                        recovery_task_count=len(recover_extra),
                    )
                    # print(f"new handle {new_handle.infer_node_name}")
                    new_handle.instance_data = new_instace_data
                    new_handles.append(new_handle)
                    self.base.infer_controls.reg_new_handle(new_handle, infer_control)
                    for task in recover_extra:
                        recover_tasks.append((dealer_node, task))
                        # new_handle.instance_data.update_by_task(task)
                new_handles_info = "\n  ".join(str(h) for h in new_handles)
                self.base.logger.info(f"find replacement of {handle} for infer ctrl: {infer_control.ctrl_id}:\n  "+new_handles_info)
            else:
                self.base.logger.info(f"cannot find replacement of {handle} for infer ctrl: {infer_control.ctrl_id}, reduce reduncy  ")
            # do remove
            self.base.infer_controls.remove_handle(handle.infer_node_name)
            handle.drop(False)

        assert infer_control.is_contigious(), f"cannot heal infer path, current infer path is: \n{infer_control}"
        record_event(
            "state_rebuild_end",
            recovery_id=recovery_id,
            control_id=infer_control.ctrl_id,
            replay_task_count=len(recover_tasks),
            replay_input_count=sum(
                task.recovery_decode_count for _, task in recover_tasks),
        )
        record_event(
            "replacement_graph_installed",
            recovery_id=recovery_id,
            control_id=infer_control.ctrl_id,
        )
        # Retain the old phase-boundary event for analyzers that still consume
        # it.  It is not the Promise handoff commit: recovery packets have not
        # been handed to the local send path yet.
        record_event(
            "graph_update_end",
            recovery_id=recovery_id,
            control_id=infer_control.ctrl_id,
        )
        if recover_tasks:
            record_event(
                "replay_start",
                recovery_id=recovery_id,
                control_id=infer_control.ctrl_id,
                replay_task_count=len(recover_tasks),
                replay_input_count=sum(
                    task.recovery_decode_count for _, task in recover_tasks),
            )
        mark_pending(infer_control.ctrl_id, recovery_id)
        recovery_task_count = len(recover_tasks)
        for recovery_task_index, (dealer_node, task) in enumerate(recover_tasks):
            packet_id = uuid4().hex
            recovery_role = (
                "frontier_reinject"
                if recovery_task_index == recovery_task_count - 1
                else "replay"
            )
            wrapped_task = MultiPathInferTask(
                task=task,
                resource_loc=infer_control.res_location,
                trace=[],
                trace_invalid=True,
                recovery_id=recovery_id,
                recovery_packet_id=packet_id,
                recovery_task_index=recovery_task_index,
                recovery_task_count=recovery_task_count,
                recovery_role=recovery_role,
            )
            record_event(
                "recovery_packet_send",
                recovery_id=recovery_id,
                control_id=infer_control.ctrl_id,
                dealer_node=dealer_node,
                packet_id=packet_id,
                task_index=recovery_task_index,
                task_count=recovery_task_count,
                role=recovery_role,
                step=task.step,
                is_prefill=task.is_prefill,
                recovery_decode_count=task.recovery_decode_count,
                payload_retained_bytes=retained_bytes(wrapped_task),
            )
            print(f"send recover delta: {wrapped_task} -> {dealer_node}")
            self.base.send((dealer_node, API_InferTask, wrapped_task))
            record_event(
                "recovery_packet_send_return",
                recovery_id=recovery_id,
                control_id=infer_control.ctrl_id,
                dealer_node=dealer_node,
                packet_id=packet_id,
                task_index=recovery_task_index,
                task_count=recovery_task_count,
                role=recovery_role,
                step=task.step,
                is_prefill=task.is_prefill,
                acceptance_semantics="send_returned_without_exception",
            )
        if recover_tasks:
            record_event(
                "replay_send_end",
                recovery_id=recovery_id,
                control_id=infer_control.ctrl_id,
                replay_task_count=len(recover_tasks),
                replay_input_count=sum(
                    task.recovery_decode_count for _, task in recover_tasks),
            )
        record_event(
            "promise_handoff_committed",
            recovery_id=recovery_id,
            control_id=infer_control.ctrl_id,
            recovery_task_count=recovery_task_count,
            commit_semantics=(
                "replacement_graph_installed_and_all_required_"
                "send_calls_returned_without_exception"
            ),
        )
        record_event(
            "recovery_handler_return",
            recovery_id=recovery_id,
            control_id=infer_control.ctrl_id,
            recovery_task_count=recovery_task_count,
        )

    def on_grouped_infer_task(self, src: str, grouped_task: GroupedInferTask):
        for t in grouped_task.tasks:
            self.on_infer_task(src, t)

    def on_infer_task(self, node: str, infer_task: MultiPathInferTask):
        infer_task.record_step("root")
        infer_control = self.base.infer_controls.get(infer_task.task.seq_id)
        accepted_output = bool(
            infer_control is not None
            and infer_task.task.navi.finished
            and not infer_task.task.is_recovery_replay
            and infer_task.unique_id in infer_control.result_waitor
        )
        if infer_task.task.navi.finished and not infer_task.task.is_recovery_replay:
            record_event(
                "infer_output_arrival",
                control_id=infer_task.task.seq_id,
                step=infer_task.task.step,
                source_node=node,
                trace=list(infer_task.trace),
                trace_invalid=infer_task.trace_invalid,
                accepted=accepted_output,
                recovery_id=infer_task.recovery_id,
                recovery_role=infer_task.recovery_role,
                output_token_id=(
                    infer_task.task.output[0]
                    if infer_task.task.output is not None
                    else None
                ),
            )
        # Experiment-only, opt-in fault hook.  Sending the command before the
        # accepted output resolves the request future guarantees that the
        # selected physical winner is torn down before the next decode is
        # dispatched.  Normal executions never set the environment variable.
        active_drop_key = (
            infer_task.task.seq_id,
            infer_task.task.step + 1,
        )
        if (
            accepted_output
            and self._active_winner_drop_step == infer_task.task.step + 1
            and active_drop_key not in self._active_winner_drop_sent
        ):
            self._active_winner_drop_sent.add(active_drop_key)
            target = self._active_winner_drop_relay or node
            payload = {
                "dealer_node": node,
                "sequence_id": infer_task.task.seq_id,
                "previous_winner_step": infer_task.task.step,
                "failure_step": infer_task.task.step + 1,
            }
            record_event(
                "active_winner_failure_command_send",
                target_node=target,
                **payload,
            )
            self.base.send((
                target,
                "recover_multi.drop_active_winner",
                payload,
            ))
        # A split path may route an unfinished intermediate activation back
        # through the root control node before it reaches the next provider.
        # That packet is not an output and must not complete recovery timing.
        if not infer_task.task.is_recovery_replay:
            if infer_task.task.navi.finished:
                record_first_output(
                    infer_task.task.seq_id,
                    infer_task.task.step,
                    infer_task.trace_invalid,
                    packet_recovery_id=infer_task.recovery_id,
                    recovery_packet_id=infer_task.recovery_packet_id,
                    recovery_task_index=infer_task.recovery_task_index,
                    recovery_role=infer_task.recovery_role,
                )
            else:
                record_intermediate_arrival(
                    infer_task.task.seq_id,
                    infer_task.task.step,
                    infer_task.task.navi.current,
                    infer_task.trace_invalid,
                )
        self.base.infer_controls.on_infer_task(node, infer_task)

    @staticmethod
    def _replacement_blacklist(
        infer_control: InferControlBase,
        effected_handle: InferNodeHandle,
        failed_dealer: str,
    ) -> set[str]:
        """Keep a recovered redundant path physically distinct.

        A replacement lookup that blacklists only the failed Dealer may select
        a Dealer already used by a surviving redundant handle.  The resulting
        graph is contiguous, but it silently collapses two physical paths onto
        one failure domain.  Exclude surviving Dealers that overlap the failed
        handle's model range so recovery restores, rather than merely counts,
        path redundancy.
        """

        affected_models = set(effected_handle.base_models)
        blacklisted = {failed_dealer}
        for _, handle in infer_control.all_handles:
            if handle is effected_handle:
                continue
            if affected_models.intersection(handle.base_models):
                blacklisted.add(handle.dealer_node_name)
        replacement_map = json.loads(os.getenv(
            "RECOVER_MULTI_REPLACEMENT_MAP", "{}"
        ))
        preferred = replacement_map.get(failed_dealer)
        if preferred is not None:
            blacklisted.update(
                dealer for dealer in {
                    *replacement_map.keys(), *replacement_map.values()
                }
                if dealer != preferred
            )
        return blacklisted

    def recover_handles_by_infer_node(
        self,
        infer_control: InferControlBase,
        effected_handle: InferNodeHandle,
        report_dealer_node: str,
        recovery_id: Optional[str] = None,
    ):
        async def coro():
            active_recovery_id = recovery_id or new_recovery_id()
            # avoid using dealer node which just report sucide, as the resource could be insufficient
            # lookup_uuid = uuid.uuid4().hex
            record_event(
                "replacement_lookup_start",
                recovery_id=active_recovery_id,
                control_id=infer_control.ctrl_id,
                failed_dealer=report_dealer_node,
                models=effected_handle.base_models,
            )
            await effected_handle.recover_lock.acquire()
            # print(f"!!! recover {lookup_uuid} start")
            replacement_blacklist = self._replacement_blacklist(
                infer_control,
                effected_handle,
                report_dealer_node,
            )
            replacement = await self.base.lookup_replacement(
                "infer",
                effected_handle.base_models,
                replacement_blacklist,
            )
            if replacement is None:
                self.base.logger.warn(f"no replacement for {effected_handle.base_models} found, ignore blacklists")
                replacement = await self.base.lookup_replacement("infer", effected_handle.base_models, None)
            # print(f"!!! recover {lookup_uuid} found {replacement} for {effected_handle.base_models}")
            effected_handle.recover_lock.release()
            record_event(
                "replacement_lookup_end",
                recovery_id=active_recovery_id,
                control_id=infer_control.ctrl_id,
                replacement=replacement,
            )
            self.recover_handles(
                [(effected_handle, replacement)],
                infer_control,
                active_recovery_id,
            )
        asyncio.create_task(coro())

    # def on_node_suicide(self, dealer_node: str, node_name: str):
    #     target = self.base.infer_controls.get_handle(node_name)
    #     if target is None:
    #         return
    #     self.base.logger.warn(f"{node_name} @ {dealer_node} suicide, self healing")
    #     infer_control, handle = target
    #     self.recover_handles_by_infer_node(infer_control=infer_control, effected_handle=handle, report_dealer_node=dealer_node)

    def recover_handles_by_dealer_node(self, infer_control: InferControlBase, effected_handles: List[InferNodeHandle], dropped_node: str):
        async def coro():
            recovery_id = new_recovery_id()
            # lookup_uuid = uuid.uuid4().hex
            replacements: List[Tuple[InferNodeHandle, Optional[List[Tuple[str, int, int]]]]] = []
            # print(f"recover {lookup_uuid} start")
            record_event(
                "replacement_lookup_start",
                recovery_id=recovery_id,
                control_id=infer_control.ctrl_id,
                failed_dealer=dropped_node,
                handle_count=len(effected_handles),
            )
            for handle in effected_handles:
                await handle.recover_lock.acquire()
                replacement_blacklist = self._replacement_blacklist(
                    infer_control,
                    handle,
                    dropped_node,
                )
                replacement = await self.base.lookup_replacement(
                    "infer",
                    handle.base_models,
                    replacement_blacklist,
                )
                if replacement is None:
                    self.base.logger.warn(f"no replacement for {handle.base_models} found, ignore blacklists")
                    replacement = await self.base.lookup_replacement("infer", handle.base_models, None)
                # print(f"reocver {lookup_uuid} found {replacement} for {handle.base_models}")
                handle.recover_lock.release()
                replacements.append((handle, replacement))
            # print(f"recover {lookup_uuid} end")
            record_event(
                "replacement_lookup_end",
                recovery_id=recovery_id,
                control_id=infer_control.ctrl_id,
                replacement_count=sum(1 for _, replacement in replacements if replacement is not None),
            )
            self.recover_handles(replacements, infer_control, recovery_id)
        asyncio.create_task(coro())

    # def on_link_change(self, node: str, drop: bool, dsts: Set[str]):
    #     if not drop:
    #         return
    #     targets = self.base.infer_controls.get_handle_by_dealer(node)
    #     if targets is None:
    #         return
    #     for infer_control, handles in targets:
    #         self.base.logger.warn(f"{node} dropped, {infer_control.ctrl_id} effected, self healing")
    #         self.recover_handles_by_dealer_node(infer_control=infer_control, effected_handles=handles, dropped_node=node)
