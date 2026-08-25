import atexit
import collections
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

from dataclasses import dataclass, field

from .backbone_base import BackboneBaseNode

from ..dealer_node.train_handle import TrainNodeHandle

from ..compute_node.train_meta import ModelCKPT, TrainTask

from ..train.train_ctrl import TrainControlNode
from ..dealer_node.protocols import API_TrainTask, API_NodeSuicide, API_TrainModelCKPT, CreateTrainNode, TrainModelCKPT
from dealers.common.base_types import ModelDescribe
from ..path_utils import InspectedPath, fmt_inspect_path, gen_lowest_energy_paths, paths_to_keypoints
from ..train.train_ctrl import TrainControlBase, TrainControlNode
from ..dealer_node.protocols import MultiPathTrainTask
import extend_asyncio as asyncio
from .. import flags
from ..train_recovery_metrics import new_train_recovery_id
from utils.runtime_metrics import record_event


@dataclass
class VirtualNodeExtraInfo:
    handle_host_node: str
    virtual_node_name_to_handler: str
    virtual_node_name_to_dealer: str
    forwarded_ckpts: Dict[str, int] = field(default_factory=dict)


class BackboneTrainNode:
    def __init__(
        self,
        base_node: BackboneBaseNode,
    ) -> None:
        self.base = base_node
        self.train_controls = TrainControlNode(self.base.base)
        # self.base.apis[API_Announce] = self.on_announce
        # self.base.apis[API_CreateTrainNode] = self.on_create_train_node
        # self.base.apis[API_TrainTask] = self.on_train_task
        # self.base.apis[API_RemoveNode] = self.on_remove_node
        # self.base.apis[API_NodeSuicide] = self.on_node_suicide
        # self.base.apis[API_TrainModelCKPT] = self.on_model_ckpt
        # self.base.link_change_cbs.append(self.on_link_change)

        self.ctrl_name_fmt = "agent-train-{}#"
        _res = re.compile(r'(agent-train-\d+#)')

        def _resolve_ctrl_name(node_name: str) -> str:
            m = _res.match(node_name)
            assert m
            return m.group(1)
        self.ctrl_name_resolve = _resolve_ctrl_name

        # {train_node_name(virtual):(TrainControl,extra_info)}
        self.name_to_virtual_nodes: Dict[str, Tuple[TrainControlBase, VirtualNodeExtraInfo]] = {}
        self.name_to_virtual_node_cache: Dict[str, Tuple[List[str], Dict[int, List[str]]]] = {}
        # reverse lookup  {ctrl_id(train_node_name_in_dealer): train_node_name(virtual)}
        self.ctrl_name_to_virtual_node_name: Dict[str, str] = {}
        self.name_to_virtual_node_init_waitor: Dict[str, asyncio.Future[None]] = {}
        self.handle_node_to_virtual_nodes: Dict[str, Set[str]] = collections.defaultdict(set)
        self.user_model_to_virtual_nodes: Dict[str, List[str]] = {}
        # self.seq_id_to_handle: Dict[str, Dict[str, TrainControlBase]] = {}

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
        await flags.InspectPostpone("backbone.train")
        if providers is None:
            providers = self.base.get_providers(base_models)
        if providers is None:
            return None
        self.base.logger.info(f"inspecting path: \n  {fmt_inspect_path(base_models,providers)}")
        paths = gen_lowest_energy_paths(providers)
        if paths is None:
            self.base.logger.error(f"inspect path fail: \n  {fmt_inspect_path(base_models,providers,None)}")
            return None
        keypoints = paths_to_keypoints(paths, no_overlap=True)
        base_model = ModelDescribe.from_str(base_models[0]).base_model
        hidden_size = (await self.base.get_extra_info()).models[base_model].hidden_size
        trace = await self.train_controls.inspect_fastest_path(
            base_models=base_models,
            keypoint_providers=keypoints,
            hidden_state_size=hidden_size,
        )
        self.base.logger.info(f"inspected path: \n  {fmt_inspect_path(base_models,providers,trace)}")
        return InspectedPath(providers, trace)

    def on_create_train_node(self, src: str, cfg: CreateTrainNode):
        virtual_train_node_name = cfg.name
        if cfg.config.arch == "it":
            assert cfg.config.tail is not None
            if cfg.config.tail.src == "patch:tail":
                self.train_controls.on_create_patch_train_node(src, cfg)
                return

        assert virtual_train_node_name not in self.name_to_virtual_node_init_waitor
        assert virtual_train_node_name not in self.name_to_virtual_nodes
        fut: asyncio.Future[None] = asyncio.Future()
        self.name_to_virtual_node_init_waitor[virtual_train_node_name] = fut
        self.handle_node_to_virtual_nodes.setdefault(src, set()).add(virtual_train_node_name)
        for dst in cfg.config.dir_user_models:
            assert not dst in self.user_model_to_virtual_nodes.get(dst, [])
            self.user_model_to_virtual_nodes.setdefault(dst, []).append(virtual_train_node_name)

        async def coro():
            config = cfg.config

            base_models: List[str] = config.base_models
            user_model_prefix = ModelDescribe.from_str(config.models[0][1]).base_model
            inspected = await self.base.inspect_fastest_n_path(
                inspect_algo=self.inspect_fastest_path,
                base_models=base_models,
                allow_reuse=True,
                max_n=self.base.path_select_redundancy+1,
            )
            assert inspected is not None
            provider_keypoints = inspected.keypoints
            ctrl_id = self.ctrl_name_fmt.format(self.base.next_id())

            ctrl = TrainControlBase(
                ctrl_host_name=self.base.node_name,
                ctrl_id=ctrl_id,
                base_models=base_models,
                send=self.train_controls.send,
                user_model_prefix=user_model_prefix,
            )
            for dealer, start, models_count in provider_keypoints:
                part_config = config.get_part(config.models[start:start+models_count])
                ctrl.init_train_node(dealer, cfg.ckpt_period, part_config)
            self.base.logger.info(f"inspected path for: {ctrl_id}\n{ctrl}")
            extra_info = VirtualNodeExtraInfo(
                handle_host_node=src,
                virtual_node_name_to_dealer=ctrl_id,
                virtual_node_name_to_handler=virtual_train_node_name
            )
            self.name_to_virtual_nodes[virtual_train_node_name] = (ctrl, extra_info)
            self.ctrl_name_to_virtual_node_name[ctrl_id] = virtual_train_node_name
            self.train_controls[virtual_train_node_name] = ctrl
            # if ctrl.has_tail:
            #     self.seq_id_to_handle.setdefault(seq_id, {})[virtual_train_node_name] = ctrl
            fut.set_result(None)
            del self.name_to_virtual_node_init_waitor[virtual_train_node_name]
            self.base.logger.info(f"agent virtual train node: {virtual_train_node_name} -> {ctrl_id}")
        asyncio.create_task(coro())

    def on_train_task(self, src: str, task: MultiPathTrainTask):
        # self.base.logger.error(f"backbone node {self.base.node_name} recv: {task.task} from {src}")
        task.record_step("backbone")
        # A completed task can either belong to a control hosted by this
        # Backbone (for example an inner measured path probe), or be the
        # redundant terminal copy of an outer Agent task.  The former must
        # resolve the Backbone's own waiter; only the latter is absorbed.
        if task.task.navi.finished:
            ctrl = self.train_controls.get(task.control_id or "")
            if ctrl is not None:
                ctrl.on_train_task(task)
            return
        current_model = task.task.navi.current_dir_user_model_with_extend()
        if current_model in self.train_controls.patched_train_tail_models:
            task = self.train_controls.patch_train_node(task)
            current_model = task.task.navi.current_dir_user_model_with_extend()

        virtual_node_names = self.user_model_to_virtual_nodes.get(current_model, [])
        if len(virtual_node_names) == 0:
            # not a virtual node
            self.train_controls.on_train_task(src, task)
            return

        async def coro(virtual_node_name: str, cpy_task: MultiPathTrainTask):
            await self.wait_virtual_node_init(virtual_node_name)
            virtual_node, extra = self.name_to_virtual_nodes.get(virtual_node_name, (None, None))
            if virtual_node is None:
                return
            assert isinstance(virtual_node, TrainControlBase), f"{type(virtual_node)}"
            if (not cpy_task.trace_override) and (not cpy_task.trace_invalid):
                override_len = 0
                fwd_len, bwd_len = virtual_node.dir_user_models.model_seq_count
                if cpy_task.task.navi.is_backward:
                    assert bwd_len is not None
                    override_len = bwd_len
                else:
                    override_len = fwd_len
                cpy_task.trace_override = [self.base.node_name]*override_len
            self.name_to_virtual_node_cache[virtual_node_name] = (cpy_task.trace, cpy_task.resource_loc)
            virtual_node.update_loc(train_task=cpy_task)
            virtual_node.on_train_task(task=cpy_task)
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
            TrainNodeHandle,
            Optional[List[Tuple[str, int, int]]]
        ]],
        train_control: TrainControlBase,
        recovery_id: str,
    ) -> bool:
        recover_tasks: List[Tuple[str, TrainTask]] = []
        record_event(
            "state_rebuild_start",
            recovery_id=recovery_id,
            state_type="train",
            control_id=train_control.ctrl_id,
            handler_node=self.base.node_name,
            recovery_scope="backbone",
            handle_count=len(handles),
        )
        recovery_parts = []
        for handle, new_edges in handles:
            per_edge = []
            if new_edges is not None:
                for dealer_node, start, count in new_edges:
                    new_models = handle.models[start:start + count]
                    new_instance_data = handle.instance_data.get_part(new_models)
                    new_init_cfg, recover_extra = new_instance_data.gen_init_cfg()
                    per_edge.append((dealer_node, new_instance_data, new_init_cfg, recover_extra))
            recovery_parts.append((handle, new_edges, per_edge))

        for handle, new_edges, per_edge in recovery_parts:
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
                    self.train_controls.reg_new_handle(new_handle, train_control)
                    for task in recover_extra:
                        recover_tasks.append((dealer_node, task))
                new_handles_info = "\n  ".join(str(h) for h in new_handles)
                self.base.logger.info(f"find replacement of {handle} for train ctrl: {train_control.ctrl_id}:\n  "+new_handles_info)
            else:
                self.base.logger.info(f"cannot find replacement of {handle} for train ctrl: {train_control.ctrl_id}, reduce reduncy  ")
            # do remove
            self.train_controls.remove_handle(handle.train_node_name)
            handle.drop(False)
        if not train_control.is_contigious():
            return False
        record_event(
            "state_rebuild_end",
            recovery_id=recovery_id,
            state_type="train",
            control_id=train_control.ctrl_id,
            handler_node=self.base.node_name,
            recovery_scope="backbone",
            replay_task_count=len(recover_tasks),
        )
        record_event(
            "replacement_graph_installed",
            recovery_id=recovery_id,
            state_type="train",
            control_id=train_control.ctrl_id,
            handler_node=self.base.node_name,
            recovery_scope="backbone",
            replay_task_count=len(recover_tasks),
        )
        # repalce res location by latest cache
        virtual_node_name = self.ctrl_name_to_virtual_node_name[train_control.ctrl_id]
        trace, parent_loc = self.name_to_virtual_node_cache[virtual_node_name]
        for task_index, (dealer_node, task) in enumerate(recover_tasks):
            packet_id = uuid4().hex
            wrapped_task = MultiPathTrainTask(
                task=task,
                resource_loc=parent_loc,
                trace=trace,
                trace_invalid=True,
                control_id=train_control.ctrl_id,
                packet_id=packet_id,
                origin="recovery_replay",
                recovery_id=recovery_id,
                recovery_task_index=task_index,
                recovery_task_count=len(recover_tasks),
                logical_handoff_id=recovery_id,
            )
            # wrapped_task.task.debug_check_cpu()
            train_control.update_loc(train_task=wrapped_task)
            print(f"send recover delta: {wrapped_task} -> {dealer_node}")
            record_event(
                "recovery_packet_send_start",
                recovery_id=recovery_id,
                state_type="train",
                control_id=train_control.ctrl_id,
                handler_node=self.base.node_name,
                recovery_scope="backbone",
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
                handler_node=self.base.node_name,
                recovery_scope="backbone",
                packet_id=packet_id,
                packet_kind="recovery_replay",
                task_index=task_index,
                task_count=len(recover_tasks),
                minibatch_id=task.minibatch_id,
                request_version=task.request_version,
                target_dealer=dealer_node,
            )
        record_event(
            "promise_handoff_committed",
            recovery_id=recovery_id,
            state_type="train",
            control_id=train_control.ctrl_id,
            handler_node=self.base.node_name,
            recovery_scope="backbone",
            required_packet_count=len(recover_tasks) + sum(
                len(per_edge) for _, _, per_edge in recovery_parts
            ),
            replay_task_count=len(recover_tasks),
        )
        train_control.latest_handoff_id = recovery_id
        record_event(
            "recovery_handler_return",
            recovery_id=recovery_id,
            state_type="train",
            control_id=train_control.ctrl_id,
            handler_node=self.base.node_name,
            recovery_scope="backbone",
        )
        return True

    def has_node(self, name: str):
        if name in self.name_to_virtual_node_init_waitor:
            return True
        return name in self.name_to_virtual_nodes

    def remove_virtual_node(self, name: str):
        self.base.logger.info(f"remove train node: {name}")
        if name in self.name_to_virtual_node_init_waitor:
            # Path probes may remove the outer virtual node while its inner
            # measured probe is still finishing.  Let creation reach its
            # normal completion point, then remove the installed node.
            waitor = self.name_to_virtual_node_init_waitor[name]

            async def remove_after_init() -> None:
                await waitor
                if name in self.name_to_virtual_nodes:
                    self.remove_virtual_node(name)

            asyncio.create_task(remove_after_init())
            return
        if name not in self.name_to_virtual_nodes:
            return
        node, extra = self.name_to_virtual_nodes[name]
        user_models = node.dir_user_models.dir_models.copy()
        # if extra.seq_id in self.seq_id_to_handle:
        #     if name in self.seq_id_to_handle[extra.seq_id]:
        #         del self.seq_id_to_handle[extra.seq_id][name]
        #         if not self.seq_id_to_handle[extra.seq_id]:
        #             del self.seq_id_to_handle[extra.seq_id]
        del self.ctrl_name_to_virtual_node_name[extra.virtual_node_name_to_dealer]
        del self.name_to_virtual_nodes[name]
        del self.name_to_virtual_node_cache[name]
        del self.train_controls[name]
        self.handle_node_to_virtual_nodes[extra.handle_host_node].remove(name)
        if not self.handle_node_to_virtual_nodes[extra.handle_host_node]:
            del self.handle_node_to_virtual_nodes[extra.handle_host_node]
        for model in user_models:
            self.user_model_to_virtual_nodes[model] = [
                n for n in self.user_model_to_virtual_nodes[model]
                if n != name
            ]
            if not self.user_model_to_virtual_nodes[model]:
                del self.user_model_to_virtual_nodes[model]

    # def on_remove_node(self, src: str, data: RemoveNode):
    #     # called by upper control node
    #     name = data.name
    #     if name in self.train_controls.patched_train_tail_nodes:
    #         self.train_controls.on_remove_patch_train_node(src, data)
    #         return
    #     self.remove_virtual_node(name)

    def suicide_node(self, node_name: str):
        self.base.logger.error(f"node suicide: {node_name}")
        # call upper control node
        # TODO finish slow snail suicide logic
        _, extra = self.name_to_virtual_nodes[node_name]
        self.remove_virtual_node(node_name)
        self.base.send((extra.handle_host_node, API_NodeSuicide, node_name))

    def recover_handles_by_train_node(
        self,
        train_control: TrainControlBase,
        effected_handle: TrainNodeHandle,
        report_dealer_node: str,
        recovery_id: Optional[str] = None,
    ):
        async def coro():
            active_recovery_id = recovery_id or new_train_recovery_id()
            # avoid using dealer node which just report sucide, as the resource could be insufficient
            record_event(
                "replacement_lookup_start",
                recovery_id=active_recovery_id,
                state_type="train",
                control_id=train_control.ctrl_id,
                handler_node=self.base.node_name,
                recovery_scope="backbone",
                failed_dealer=report_dealer_node,
                models=effected_handle.base_models,
            )
            affected_models = set(effected_handle.base_models)
            replacement_blacklist = {report_dealer_node}
            for _, handle in train_control.all_handles:
                if handle is effected_handle:
                    continue
                if affected_models.intersection(handle.base_models):
                    replacement_blacklist.add(handle.dealer_node_name)
            replacement_map = json.loads(os.getenv(
                "PEFT_MULTI_REPLACEMENT_MAP", "{}"
            ))
            preferred = replacement_map.get(report_dealer_node)
            if preferred is not None:
                replacement_blacklist.update(
                    dealer for dealer in {
                        *replacement_map.keys(), *replacement_map.values()
                    }
                    if dealer != preferred
                )
            replacement = await self.base.lookup_replacement(
                effected_handle.base_models,
                self.inspect_fastest_path,
                replacement_blacklist,
            )
            record_event(
                "replacement_lookup_end",
                recovery_id=active_recovery_id,
                state_type="train",
                control_id=train_control.ctrl_id,
                handler_node=self.base.node_name,
                recovery_scope="backbone",
                replacement=replacement,
            )
            ok = self.recover_handles(
                [(effected_handle, replacement)],
                train_control,
                active_recovery_id,
            )
            if ok:
                return
            self.base.logger.warn(f"healing fail, self suicide")
            self.suicide_node(self.ctrl_name_to_virtual_node_name[train_control.ctrl_id])
        asyncio.create_task(coro())

    # def on_node_suicide(self, dealer_node: str, node_name: str) -> None:
    #     target = self.train_controls.get_handle(node_name)
    #     if target is None:
    #         return
    #     self.base.logger.warn(f"{node_name} @ {dealer_node} suicide, self healing")
    #     train_control, handle = target
    #     self.recover_handles_by_train_node(train_control=train_control, effected_handle=handle, report_dealer_node=dealer_node)

    def on_control_node_drop(self, node: str):
        nodes = self.handle_node_to_virtual_nodes.get(node, None)
        if not nodes:
            return
        for n in nodes:
            self.remove_virtual_node(n)
        del self.handle_node_to_virtual_nodes[node]

    async def recover_handles_by_dealer_node(self, train_control: TrainControlBase, effected_handles: List[TrainNodeHandle], dropped_node: str):
        async def coro():
            recovery_id = new_train_recovery_id()
            replacements: List[Tuple[TrainNodeHandle, Optional[List[Tuple[str, int, int]]]]] = []
            for handle in effected_handles:
                relpacement = await self.base.lookup_replacement(handle.base_models, self.inspect_fastest_path, {dropped_node})
                replacements.append((handle, relpacement))

            ok = self.recover_handles(replacements, train_control, recovery_id)
            if ok:
                return
            self.base.logger.warn(f"healing fail, self suicide")
            self.suicide_node(self.ctrl_name_to_virtual_node_name[train_control.ctrl_id])
        asyncio.create_task(coro())

    def on_dealer_node_drop(self, node: str) -> None:
        targets = self.train_controls.get_handle_by_dealer(node)
        if targets is None:
            return
        for train_control, handles in targets:
            self.base.logger.warn(f"{node} dropped, {train_control.ctrl_id} effected, self healing")
            asyncio.create_task(self.recover_handles_by_dealer_node(train_control=train_control, effected_handles=handles, dropped_node=node))

    # def on_link_change(self, node: str, drop: bool, dsts: Set[str]):
    #     if not drop:
    #         return
    #     self.on_control_node_drop(node)
    #     self.on_dealer_node_drop(node)

    def on_model_ckpt(self, src: str, ckpt: TrainModelCKPT):
        ctrl_name = self.ctrl_name_resolve(ckpt.train_node_name)
        virtual_node_name = self.ctrl_name_to_virtual_node_name.get(ctrl_name)
        if virtual_node_name is None:
            return

        self.train_controls[virtual_node_name].on_ckpt(ckpt)
        _, extra = self.name_to_virtual_nodes[virtual_node_name]

        forwarded_ckpts = extra.forwarded_ckpts
        user_model = ckpt.ckpt.user_model
        version = ckpt.ckpt.version
        if version <= forwarded_ckpts.setdefault(user_model, -1):
            return
        forwarded_ckpts[user_model] = version
        self.base.send((extra.handle_host_node, API_TrainModelCKPT, TrainModelCKPT(
            train_node_name=virtual_node_name,
            ckpt=ModelCKPT(
                user_model=user_model,
                version=version,
                state=ckpt.ckpt.state
            ),
        )), False)

    def _debug_check_resource_release(self):
        assert not self.ctrl_name_to_virtual_node_name, self.ctrl_name_to_virtual_node_name
        assert not self.name_to_virtual_node_init_waitor
        assert not self.name_to_virtual_node_cache
        assert not self.handle_node_to_virtual_nodes
        assert not self.user_model_to_virtual_nodes
        # assert not self.seq_id_to_handle
        print("backbone all released")
