import atexit
import collections
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

from dataclasses import dataclass, field

from .backbone_base import BackboneBaseNode

from ..compute_node.infer_meta import InferTask, PutBlocks

from ..dealer_node.infer_handle import InferNodeHandle
from ..dealer_node.protocols import API_InferTask, API_InferPutBlocks, API_NodeSuicide, CreateInferNode, GroupedInferTask, InferPutBlocks
from dealers.common.base_types import ModelDescribe
from ..path_utils import InspectedPath, gen_lowest_energy_paths, paths_to_keypoints
from ..infer.infer_ctrl import InferControlBase, InferControlNode
from ..dealer_node.protocols import MultiPathInferTask
import extend_asyncio as asyncio
from .. import flags
from node.recovery_metrics import new_recovery_id
from utils.runtime_metrics import record_event
from utils.size_metrics import retained_bytes


@dataclass
class VirtualNodeExtraInfo:
    handle_host_node: str
    seq_id: str
    virtual_node_name_to_handler: str
    virtual_node_name_to_dealer: str
    forwarded_blocks: Set[Tuple[str, int]] = field(default_factory=set)


class BackboneInferNode:
    def __init__(
        self,
        base_node: BackboneBaseNode,
    ) -> None:
        self.base = base_node
        self.infer_controls = InferControlNode(self.base.base)
        # self.base.apis[API_Announce] = self.on_announce
        # self.base.apis[API_CreateInferNode] = self.on_create_infer_node
        # self.base.apis[API_InferTask] = self.on_infer_task
        # self.base.apis[API_RemoveNode] = self.on_remove_node
        # self.base.apis[API_NodeSuicide] = self.on_node_suicide
        # self.base.apis[API_InferPutBlocks] = self.on_putblocks
        # self.base.apis[API_GroupedInferTask] = self.on_grouped_infer_task
        # self.base.link_change_cbs.append(self.on_link_change)

        self.ctrl_name_fmt = "agent-infer-{}#"
        _res = re.compile(r'(agent-infer-\d+#)')

        def _resolve_ctrl_name(node_name: str) -> str:
            m = _res.match(node_name)
            assert m
            return m.group(1)
        self.ctrl_name_resolve = _resolve_ctrl_name

        # {infer_node_name(virtual):(InferControl,extra_info)}
        self.name_to_virtual_nodes: Dict[str, Tuple[InferControlBase, VirtualNodeExtraInfo]] = {}
        self.name_to_virtual_node_cache: Dict[str, Tuple[List[str], Dict[int, List[str]]]] = {}
        # reverse lookup  {ctrl_id(infer_node_name_in_dealer): infer_node_name(virtual)}
        self.ctrl_name_to_virtual_node_name: Dict[str, str] = {}
        self.name_to_virtual_node_init_waitor: Dict[str, asyncio.Future[None]] = {}
        self.cancelled_virtual_node_inits: Set[str] = set()
        self.handle_node_to_virtual_nodes: Dict[str, Set[str]] = collections.defaultdict(set)
        self.user_model_to_virtual_nodes: Dict[str, List[str]] = {}
        # self.seq_id_to_handle: Dict[str, Dict[str, InferControlBase]] = {}

        atexit.register(self._debug_check_resource_release)

    async def wait_virtual_node_init(self, node: str):
        if node not in self.name_to_virtual_node_init_waitor:
            return
        fut = self.name_to_virtual_node_init_waitor[node]
        await fut

    async def inspect_fastest_path(
        self,
        base_models: List[str],
        providers: Optional[List[List[str]]] = None,
    ):
        await flags.InspectPostpone("backbone.infer")
        if providers is None:
            providers = self.base.get_providers(base_models)
        if providers is None:
            return None
        paths = gen_lowest_energy_paths(providers)
        if paths is None:
            return None
        keypoints = paths_to_keypoints(paths, no_overlap=True)
        base_model = ModelDescribe.from_str(base_models[0]).base_model
        hidden_size = (await self.base.get_extra_info()).models[base_model].hidden_size
        trace = await self.infer_controls.inspect_fastest_path(
            base_models=base_models,
            keypoint_providers=keypoints,
            hidden_state_size=hidden_size,
        )
        return InspectedPath(providers, trace)

    def on_create_infer_node(self, src: str, cfg: CreateInferNode):
        virtual_infer_node_name = cfg.name
        assert virtual_infer_node_name not in self.name_to_virtual_node_init_waitor
        assert virtual_infer_node_name not in self.name_to_virtual_nodes
        fut: asyncio.Future[None] = asyncio.Future()
        self.name_to_virtual_node_init_waitor[virtual_infer_node_name] = fut
        self.handle_node_to_virtual_nodes.setdefault(src, set()).add(virtual_infer_node_name)
        for dst in cfg.config.user_models:
            assert not dst in self.user_model_to_virtual_nodes.get(dst, [])
            self.user_model_to_virtual_nodes.setdefault(dst, []).append(virtual_infer_node_name)

        async def coro():
            config = cfg.config
            seq_id = cfg.config.seq_id

            base_models: List[str] = config.base_models
            user_model_prefix = ModelDescribe.from_str(config.models[0][1]).base_model
            inspected = await self.base.inspect_fastest_n_path(
                inspect_algo=self.inspect_fastest_path,
                base_models=base_models,
                allow_reuse=True,
                max_n=self.base.path_select_redundancy+1,
            )
            assert inspected is not None
            if virtual_infer_node_name in self.cancelled_virtual_node_inits:
                self.cancelled_virtual_node_inits.remove(
                    virtual_infer_node_name
                )
                return
            provider_keypoints = inspected.keypoints
            ctrl_id = self.ctrl_name_fmt.format(self.base.next_id())

            ctrl = InferControlBase(
                ctrl_host_name=self.base.node_name,
                ctrl_id=ctrl_id,
                base_models=base_models,
                send=self.infer_controls.send,
                send_infer_task=self.infer_controls.infer_batch_sender.send,
                user_model_prefix=user_model_prefix,
            )
            for dealer, start, models_count in provider_keypoints:
                part_config = config.get_part(config.models[start:start+models_count])
                ctrl.init_infer_node(dealer, cfg.block_retrieve, part_config)
            self.base.logger.info(f"inspected path for: {ctrl_id}\n{ctrl}")
            extra_info = VirtualNodeExtraInfo(
                handle_host_node=src,
                seq_id=seq_id,
                virtual_node_name_to_dealer=ctrl_id,
                virtual_node_name_to_handler=virtual_infer_node_name
            )
            self.name_to_virtual_nodes[virtual_infer_node_name] = (ctrl, extra_info)
            self.ctrl_name_to_virtual_node_name[ctrl_id] = virtual_infer_node_name
            self.infer_controls[virtual_infer_node_name] = ctrl
            # if ctrl.has_tail:
            #     self.seq_id_to_handle.setdefault(seq_id, {})[virtual_infer_node_name] = ctrl
            fut.set_result(None)
            del self.name_to_virtual_node_init_waitor[virtual_infer_node_name]
            self.base.logger.info(f"agent virtual infer node: {virtual_infer_node_name} -> {ctrl_id}")
        asyncio.create_task(coro())

    def on_grouped_infer_task(self, src: str, grouped_task: GroupedInferTask):
        for t in grouped_task.tasks:
            self.on_infer_task(src, t)

    def on_infer_task(self, src: str, task: MultiPathInferTask):
        # self.base.logger.info(f"backbone node {self.base.node_name} recv: {task.task} from {src}")
        task.record_step("backbone")
        current_model = task.task.navi.current_user_model(True)
        virtual_node_names = self.user_model_to_virtual_nodes.get(current_model, [])
        if len(virtual_node_names) == 0:
            # not a virtual node
            self.infer_controls.on_infer_task(src, task)
            return

        async def coro(virtual_node_name: str, cpy_task: MultiPathInferTask):
            await self.wait_virtual_node_init(virtual_node_name)
            virtual_node, _ = self.name_to_virtual_nodes.get(virtual_node_name, (None, None))
            if virtual_node is None:
                return
            assert isinstance(virtual_node, InferControlBase), f"{type(virtual_node)}"
            if not cpy_task.trace_override:
                cpy_task.trace_override = [self.base.node_name]*len(virtual_node.base_models)
            self.name_to_virtual_node_cache[virtual_node_name] = (cpy_task.trace, cpy_task.resource_loc)
            first_user_model_in_virtual_node = virtual_node.user_models[0]
            offset = task.task.navi.index(first_user_model_in_virtual_node)
            virtual_node.update_loc(infer_task=cpy_task, offset=offset, with_last=virtual_node.has_tail)
            virtual_node.on_infer_task(task=cpy_task)
        if len(virtual_node_names) == 1:
            for virtual_node_name in virtual_node_names:
                asyncio.create_task(coro(virtual_node_name, task))
        else:
            for virtual_node_name in virtual_node_names:
                asyncio.create_task(coro(virtual_node_name, task.shallow_copy()))
        return

    def recover_handles(
        self,
        # providers: Dict[str, List[str]],
        handles: List[Tuple[
            InferNodeHandle,
            Optional[List[Tuple[str, int, int]]]
        ]],
        infer_control: InferControlBase,
        recovery_id: str,
    ) -> bool:
        if not self.infer_controls.is_registered(infer_control):
            record_event(
                "recovery_cancelled_control_retired",
                recovery_id=recovery_id,
                control_id=infer_control.ctrl_id,
                handler_node=self.base.node_name,
                recovery_scope="backbone",
            )
            return False
        recover_tasks: List[Tuple[str, InferTask]] = []
        record_event(
            "state_rebuild_start",
            recovery_id=recovery_id,
            control_id=infer_control.ctrl_id,
            handler_node=self.base.node_name,
            recovery_scope="backbone",
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
                    confirmed_output_token_ids = (
                        new_instace_data.confirmed_output_token_ids()
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
                    new_handle.instance_data = new_instace_data
                    new_handles.append(new_handle)
                    self.infer_controls.reg_new_handle(new_handle, infer_control)
                    for task in recover_extra:
                        recover_tasks.append((dealer_node, task))
                        # new_handle.instance_data.update_by_task(task)
                new_handles_info = "\n  ".join(str(h) for h in new_handles)
                self.base.logger.info(f"find replacement of {handle} for infer ctrl: {infer_control.ctrl_id}:\n  "+new_handles_info)
            else:
                self.base.logger.info(f"cannot find replacement of {handle} for infer ctrl: {infer_control.ctrl_id}, reduce reduncy  ")
            # do remove
            self.infer_controls.remove_handle(handle.infer_node_name)
            handle.drop(False)
        if not infer_control.is_contigious():
            return False
        record_event(
            "state_rebuild_end",
            recovery_id=recovery_id,
            control_id=infer_control.ctrl_id,
            handler_node=self.base.node_name,
            recovery_scope="backbone",
            replay_task_count=len(recover_tasks),
            replay_input_count=sum(
                task.recovery_decode_count for _, task in recover_tasks
            ),
        )
        record_event(
            "replacement_graph_installed",
            recovery_id=recovery_id,
            control_id=infer_control.ctrl_id,
            handler_node=self.base.node_name,
            recovery_scope="backbone",
        )
        record_event(
            "graph_update_end",
            recovery_id=recovery_id,
            control_id=infer_control.ctrl_id,
            handler_node=self.base.node_name,
            recovery_scope="backbone",
        )
        first_user_model_in_virtual_node = infer_control.user_models[0]
        # repalce res location by latest cache
        virtual_node_name = self.ctrl_name_to_virtual_node_name[infer_control.ctrl_id]
        trace, parent_loc = self.name_to_virtual_node_cache[virtual_node_name]
        recovery_task_count = len(recover_tasks)
        if recover_tasks:
            record_event(
                "replay_start",
                recovery_id=recovery_id,
                control_id=infer_control.ctrl_id,
                handler_node=self.base.node_name,
                recovery_scope="backbone",
                replay_task_count=recovery_task_count,
                replay_input_count=sum(
                    task.recovery_decode_count for _, task in recover_tasks
                ),
            )
        for recovery_task_index, (dealer_node, task) in enumerate(recover_tasks):
            packet_id = uuid4().hex
            recovery_role = (
                "frontier_reinject"
                if recovery_task_index == recovery_task_count - 1
                else "replay"
            )
            wrapped_task = MultiPathInferTask(
                task=task,
                resource_loc=parent_loc,
                trace=trace,
                trace_invalid=True,
                recovery_id=recovery_id,
                recovery_packet_id=packet_id,
                recovery_task_index=recovery_task_index,
                recovery_task_count=recovery_task_count,
                recovery_role=recovery_role,
            )
            offset = task.navi.index(first_user_model_in_virtual_node)
            infer_control.update_loc(infer_task=wrapped_task, offset=offset, with_last=infer_control.has_tail)
            record_event(
                "recovery_packet_send",
                recovery_id=recovery_id,
                control_id=infer_control.ctrl_id,
                handler_node=self.base.node_name,
                recovery_scope="backbone",
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
                handler_node=self.base.node_name,
                recovery_scope="backbone",
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
                handler_node=self.base.node_name,
                recovery_scope="backbone",
                replay_task_count=recovery_task_count,
                replay_input_count=sum(
                    task.recovery_decode_count for _, task in recover_tasks
                ),
            )
        record_event(
            "promise_handoff_committed",
            recovery_id=recovery_id,
            control_id=infer_control.ctrl_id,
            handler_node=self.base.node_name,
            recovery_scope="backbone",
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
            handler_node=self.base.node_name,
            recovery_scope="backbone",
            recovery_task_count=recovery_task_count,
        )
        return True

    def has_node(self, name: str):
        if name in self.name_to_virtual_node_init_waitor:
            return True
        return name in self.name_to_virtual_nodes

    def remove_virtual_node(self, name: str):
        self.base.logger.info(f"remove infer node: {name}")
        if name in self.name_to_virtual_node_init_waitor:
            self.name_to_virtual_node_init_waitor[name].set_result(None)
            del self.name_to_virtual_node_init_waitor[name]
            if name not in self.name_to_virtual_nodes:
                self.cancelled_virtual_node_inits.add(name)
                for handle_host, nodes in list(
                    self.handle_node_to_virtual_nodes.items()
                ):
                    nodes.discard(name)
                    if not nodes:
                        del self.handle_node_to_virtual_nodes[handle_host]
                for model, nodes in list(
                    self.user_model_to_virtual_nodes.items()
                ):
                    filtered = [node for node in nodes if node != name]
                    if filtered:
                        self.user_model_to_virtual_nodes[model] = filtered
                    else:
                        del self.user_model_to_virtual_nodes[model]
                return
        if name not in self.name_to_virtual_nodes:
            return
        node, extra = self.name_to_virtual_nodes[name]
        user_models = node.user_models.copy()
        # if extra.seq_id in self.seq_id_to_handle:
        #     if name in self.seq_id_to_handle[extra.seq_id]:
        #         del self.seq_id_to_handle[extra.seq_id][name]
        #         if not self.seq_id_to_handle[extra.seq_id]:
        #             del self.seq_id_to_handle[extra.seq_id]
        del self.ctrl_name_to_virtual_node_name[extra.virtual_node_name_to_dealer]
        del self.name_to_virtual_nodes[name]
        del self.name_to_virtual_node_cache[name]
        del self.infer_controls[name]
        self.handle_node_to_virtual_nodes[extra.handle_host_node].remove(name)
        if not self.handle_node_to_virtual_nodes[extra.handle_host_node]:
            del self.handle_node_to_virtual_nodes[extra.handle_host_node]
        for model in user_models:
            self.user_model_to_virtual_nodes[model] = [n for n in self.user_model_to_virtual_nodes[model] if n != name]
            if not self.user_model_to_virtual_nodes[model]:
                del self.user_model_to_virtual_nodes[model]

    # def on_remove_node(self, src: str, data: RemoveNode):
    #     # called by upper control node
    #     name = data.name
    #     self.remove_virtual_node(name)

    def suicide_node(self, node_name: str):
        self.base.logger.error(f"node suicide: {node_name}")
        # call upper control node
        # TODO finish slow snail suicide logic
        _, extra = self.name_to_virtual_nodes[node_name]
        self.remove_virtual_node(node_name)
        self.base.send((extra.handle_host_node, API_NodeSuicide, node_name))

    @staticmethod
    def _replacement_blacklist(
        infer_control: InferControlBase,
        effected_handle: InferNodeHandle,
        failed_dealer: str,
    ) -> Set[str]:
        """Keep a recovered Backbone path on a distinct failure domain."""

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
            record_event(
                "replacement_lookup_start",
                recovery_id=active_recovery_id,
                control_id=infer_control.ctrl_id,
                handler_node=self.base.node_name,
                recovery_scope="backbone",
                failed_dealer=report_dealer_node,
                models=effected_handle.base_models,
            )
            replacement_blacklist = self._replacement_blacklist(
                infer_control,
                effected_handle,
                report_dealer_node,
            )
            replacement = await self.base.lookup_replacement(
                effected_handle.base_models,
                self.inspect_fastest_path,
                replacement_blacklist,
            )
            if replacement is None:
                self.base.logger.warn(
                    f"no distinct replacement for {effected_handle.base_models} "
                    "found, ignore blacklists"
                )
                replacement = await self.base.lookup_replacement(
                    effected_handle.base_models,
                    self.inspect_fastest_path,
                    None,
                )
            record_event(
                "replacement_lookup_end",
                recovery_id=active_recovery_id,
                control_id=infer_control.ctrl_id,
                handler_node=self.base.node_name,
                recovery_scope="backbone",
                replacement=replacement,
            )
            ok = self.recover_handles(
                [(effected_handle, replacement)],
                infer_control,
                active_recovery_id,
            )
            if ok:
                return
            self.base.logger.warn(f"healing fail, self suicide")
            self.suicide_node(self.ctrl_name_to_virtual_node_name[infer_control.ctrl_id])
        asyncio.create_task(coro())

    # def on_node_suicide(self, dealer_node: str, node_name: str) -> None:
    #     target = self.infer_controls.get_handle(node_name)
    #     if target is None:
    #         return
    #     self.base.logger.warn(f"{node_name} @ {dealer_node} suicide, self healing")
    #     infer_control, handle = target
    #     self.recover_handles_by_infer_node(infer_control=infer_control, effected_handle=handle, report_dealer_node=dealer_node)

    def on_control_node_drop(self, node: str):
        nodes = self.handle_node_to_virtual_nodes.get(node, None)
        if not nodes:
            return
        for n in nodes:
            self.remove_virtual_node(n)
        del self.handle_node_to_virtual_nodes[node]

    async def recover_handles_by_dealer_node(self, infer_control: InferControlBase, effected_handles: List[InferNodeHandle], dropped_node: str):
        async def coro():
            recovery_id = new_recovery_id()
            replacements: List[Tuple[InferNodeHandle, Optional[List[Tuple[str, int, int]]]]] = []
            record_event(
                "replacement_lookup_start",
                recovery_id=recovery_id,
                control_id=infer_control.ctrl_id,
                handler_node=self.base.node_name,
                recovery_scope="backbone",
                failed_dealer=dropped_node,
                handle_count=len(effected_handles),
            )
            for handle in effected_handles:
                relpacement = await self.base.lookup_replacement(handle.base_models, self.inspect_fastest_path, {dropped_node})
                replacements.append((handle, relpacement))

            record_event(
                "replacement_lookup_end",
                recovery_id=recovery_id,
                control_id=infer_control.ctrl_id,
                handler_node=self.base.node_name,
                recovery_scope="backbone",
                replacement_count=sum(
                    1 for _, replacement in replacements
                    if replacement is not None
                ),
            )
            ok = self.recover_handles(replacements, infer_control, recovery_id)
            if ok:
                return
            self.base.logger.warn(f"healing fail, self suicide")
            self.suicide_node(self.ctrl_name_to_virtual_node_name[infer_control.ctrl_id])
        asyncio.create_task(coro())

    def on_dealer_node_drop(self, node: str) -> None:
        targets = self.infer_controls.get_handle_by_dealer(node)
        if targets is None:
            return
        for infer_control, handles in targets:
            self.base.logger.warn(f"{node} dropped, {infer_control.ctrl_id} effected, self healing")
            asyncio.create_task(self.recover_handles_by_dealer_node(infer_control=infer_control, effected_handles=handles, dropped_node=node))

    # def on_link_change(self, node: str, drop: bool, dsts: Set[str]):
    #     if not drop:
    #         return
    #     self.on_control_node_drop(node)
    #     self.on_dealer_node_drop(node)

    def on_put_blocks(self, src: str, put_blocks: InferPutBlocks):
        ctrl_name = self.ctrl_name_resolve(put_blocks.infer_node_name)
        virtual_node_name = self.ctrl_name_to_virtual_node_name.get(ctrl_name)
        if virtual_node_name is None:
            return

        self.infer_controls[virtual_node_name].on_putblocks(put_blocks)
        _, extra = self.name_to_virtual_nodes[virtual_node_name]

        retrieved_blocks = extra.forwarded_blocks
        target_blocks: List[int] = []
        for b in put_blocks.blocks.target_blocks:
            key = (put_blocks.blocks.user_model, b)
            if not key in retrieved_blocks:
                retrieved_blocks.add(key)
                target_blocks.append(b)
        self.base.send((extra.handle_host_node, API_InferPutBlocks, InferPutBlocks(
            infer_node_name=virtual_node_name,
            blocks=PutBlocks(
                user_model=put_blocks.blocks.user_model,
                target_blocks=target_blocks,
                layers_blocks=put_blocks.blocks.get_blocks(target_blocks)
            ),
        )), False)

    def _debug_check_resource_release(self):
        assert not self.ctrl_name_to_virtual_node_name, self.ctrl_name_to_virtual_node_name
        assert not self.name_to_virtual_node_init_waitor
        assert not self.cancelled_virtual_node_inits
        assert not self.name_to_virtual_node_cache
        assert not self.handle_node_to_virtual_nodes
        assert not self.user_model_to_virtual_nodes
        # assert not self.seq_id_to_handle
        print("backbone all released")
