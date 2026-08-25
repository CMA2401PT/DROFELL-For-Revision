import atexit
import os
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import torch


from dealers.common.base_types import DeviceTensorCarrier, ModelDescribe, gen_user_model_name, gen_user_model_names
from dealers.common.brain import OptimizerConfig, TrainForwardMeta, UserModelDataInitConfig, SchedulerConfig
from .. import flags
from .dir_model_utils import DirModel
from operators.core.ptuning.common import PtuningConfig
from operators.core.lora.common import LoraConfig
from ..base import BaseNode
from ..compute_node.train_meta import Navigate, PossibleTrainInput, TrainHeadInput, TrainInit, TrainTailInput, TrainTask
from ..dealer_node.train_handle import TrainNodeHandle
from ..dealer_node.protocols import API_CreateTrainNode, API_RemoveNode, API_TrainModelCKPT, API_TrainTask, CreateTrainNode, MultiPathTrainTask, RemoveNode, TrainModelCKPT

import extend_asyncio as asyncio
from orbit.defines import Packet
from utils.runtime_metrics import record_event


class TrainControlBase:
    def __init__(
        self,
        ctrl_host_name: str,
        ctrl_id: str,
        base_models: List[str],  # no forward, backward is included
        send: Callable[[Packet, bool], None],
        user_model_prefix: Optional[str] = None,
        is_root: bool = False,
    ) -> None:
        self.node_name = ctrl_host_name
        self.ctrl_id = ctrl_id
        self.base_models = base_models
        self.send = send
        self.is_root = is_root

        if user_model_prefix is None:
            user_model_prefix = ctrl_id
        self.user_model_prefix = user_model_prefix
        user_models = [
            gen_user_model_name(base_model=model, user_model_prefix=self.user_model_prefix)
            for model in self.base_models
        ]
        self.dir_user_models = DirModel(user_models)
        self._handles: Dict[int, List[TrainNodeHandle]] = {}
        self._index_by_train_node_name: Dict[str, Tuple[Tuple[int, Optional[int]], TrainNodeHandle]] = {}
        self._index_by_dealer: Dict[str, Set[TrainNodeHandle]] = {}
        self._res_location_cache: Optional[Dict[int, List[str]]] = None
        self._res_location_dirty: bool = True
        self.result_waitor: Dict[str, asyncio.Future[MultiPathTrainTask]] = {}
        self.train_node_counter: int = 0
        # Set only after all packets required by the latest logical recovery
        # handoff have been accepted by the local send path.
        self.latest_handoff_id: Optional[str] = None
        # self.breaker_enabled: bool = False
        # self.breaker_pendings: List[MultiPathTrainTask] = []

    def is_contigious(self):
        dealers = [0]*len(self.base_models)
        for group_i, handles in self._handles.items():
            for handle in handles:
                for i in range(handle.num_steps(is_backward=False)):
                    if group_i+i >= len(dealers):
                        continue
                    dealers[group_i+i] += 1
        return all(p > 0 for p in dealers)

    def plain_dealers_info(self):
        dealers_info: List[Tuple[str, List[Tuple[str, str]]]] = [(model, []) for model in self.base_models]
        for group_i, handles in self._handles.items():
            for handle in handles:
                for i in range(handle.num_steps(is_backward=False)):
                    ptr = group_i+i
                    if ptr < len(dealers_info):
                        dealers_info[ptr][1].append((handle.dealer_node_name, handle.train_node_name))
        return dealers_info

    def __str__(self) -> str:
        dealers_info = self.plain_dealers_info()
        s = f"TrainControl: {self.ctrl_id}\n"
        for base_model, dealers in dealers_info:
            s += f"  {gen_user_model_name(base_model,self.user_model_prefix)} -> {base_model}\n    "
            if len(dealers) > 0:
                s += ", ".join(f"{train_node}@{dealer_node}" for dealer_node, train_node in dealers)
            else:
                s += "no provider"
            s += "\n"
        return s

    def is_realated_to_dealer(self, dealer_node_name: str):
        return dealer_node_name in self._index_by_dealer

    def has_infer_node(self, infer_node_name: str):
        return infer_node_name in self._index_by_train_node_name

    @property
    def all_handles(self):
        for group_i, handles in self._handles.items():
            for handle in handles:
                yield group_i, handle

    def _mark_res_location_dirty(self):
        self._res_location_dirty = True

    @property
    def res_location(self) -> Dict[int, List[str]]:
        if self._res_location_dirty or self._res_location_cache is None:
            ret = {len(self.dir_user_models): [self.node_name]}
            for group_i, handles in self._handles.items():
                locs = [self.node_name] + [h.dealer_node_name for h in handles]
                ret[group_i] = locs
            self._res_location_cache = ret
            self._res_location_dirty = False
        return self._res_location_cache

    def get_handles_by_dealer(self, dealer_node: str):
        handles = list(self._index_by_dealer.get(dealer_node, []))
        return handles

    def get_handle(self, train_node_name: str):
        found = self._index_by_train_node_name.get(train_node_name, None)
        if not found:
            return None
        return found[1]

    # def remove_handles_by_dealer(self, dealer_node: str):
    #     # only called when node state change
    #     handles = list(self._index_by_dealer.get(dealer_node, []))
    #     for h in handles:
    #         self.remove_handle(h.train_node_name)
    #         self._index_by_train_node_name.pop(h.train_node_name, None)
    #     return handles

    def remove_handle(self, train_node_name: str):
        # only called when node state change
        found = self._index_by_train_node_name.pop(train_node_name, None)
        if not found:
            return None
        group_i, handle = found
        forwad_ptr, backward_ptr = group_i
        handles = self._handles[forwad_ptr]
        handles.remove(handle)
        if backward_ptr is not None:
            handles = self._handles[backward_ptr]
            handles.remove(handle)
        self._mark_res_location_dirty()
        dealer_set = self._index_by_dealer[handle.dealer_node_name]
        dealer_set.discard(handle)
        if not dealer_set:
            del self._index_by_dealer[handle.dealer_node_name]
        return handle

    def init_train_node(
        self,
        dealer: str,
        ckpt_period: int,
        cfg: TrainInit,
        recovery_id: Optional[str] = None,
        recovery_task_count: int = 0,
    ):
        # cfg.user_models[0]
        train_node_name = f"{self.ctrl_id}{self.train_node_counter}-<{cfg.user_models[0].split(':')[-1]}>-<{cfg.user_models[-1].split(':')[-1]}>"
        self.train_node_counter += 1
        handle = TrainNodeHandle(
            send=self.send,
            dealer_node_name=dealer,
            ckpt_period=ckpt_period,
            init_config=cfg,
            train_node_name=train_node_name,
            control_id=self.ctrl_id,
            recovery_id=recovery_id,
            recovery_task_count=recovery_task_count,
        )
        forward, backward = cfg.first_user_model()
        forward_desc = ModelDescribe.from_str(forward)
        if not forward_desc.is_decoders:
            forward_ptr = self.dir_user_models.index_dir(forward)
        else:
            forward_ptr = self.dir_user_models.index_dir(forward+",forward")
        self._handles.setdefault(forward_ptr, []).append(handle)
        backward_ptr = None
        if backward is not None:
            backward_ptr = self.dir_user_models.index_dir(backward+",backward")
            self._handles.setdefault(backward_ptr, []).append(handle)
        self._index_by_train_node_name[train_node_name] = ((forward_ptr, backward_ptr), handle)
        self._index_by_dealer.setdefault(dealer, set()).add(handle)
        self._mark_res_location_dirty()
        return handle

    # def breaker_enable(self):
    #     # replaced by no async recover/healing
    #     return
    #     # when node chain, i.e. A,B are at verion 3 (version0 + delta2)
    #     # healing happened, A,B becomes A(version0 + delta,forward 2) and B (version 0)
    #     # and severial forward samples are still in B
    #     # without breaker, this could cause B becomes version 1 and ignore forwards sends from A to B
    #     # and causing A halt becasue no backward input
    #     assert not self.breaker_enabled
    #     self.breaker_enabled = True

    # def breaker_disable(self):
    #     # replaced by no async recover/healing
    #     return
    #     assert self.breaker_enabled
    #     self.breaker_enabled = False
    #     for task in self.breaker_pendings:
    #         self.on_train_task(task)
    #     self.breaker_pendings = []

    def on_train_task(self, task: MultiPathTrainTask):
        # if self.breaker_enabled:
        #     self.breaker_pendings.append(task)
        #     return
        if task.task.navi.finished:
            if task.trace_invalid:
                # if self heading, the trace is inrealiable
                task.trace = []
            fut = self.result_waitor.pop(task.unique_id, None)
            if fut is not None:
                record_event(
                    "train_minibatch_committed",
                    recovery_id=task.recovery_id,
                    logical_handoff_id=task.logical_handoff_id,
                    control_id=task.control_id,
                    packet_id=task.packet_id,
                    origin=task.origin,
                    minibatch_id=task.task.minibatch_id,
                    request_version=task.task.request_version,
                    loss=task.task.loss,
                )
                fut.set_result(task)
                # del self.result_waitor[task.unique_id]
            return
        # res loc could be change when recover/healing happens
        if self.is_root:
            for ptr, locs in self.res_location.items():
                task.resource_loc[ptr] =\
                    list(set(task.resource_loc.setdefault(ptr, [])+locs))
        self.dispatch(task)

    def on_ckpt(self, ckpt: TrainModelCKPT):
        # print(f"{self.node_name} recv CKPT train_node:{ckpt.train_node_name} model:{ckpt.ckpt.user_model} version:{ckpt.ckpt.version}")
        found = self._index_by_train_node_name.get(ckpt.train_node_name)
        if not found:
            return
        _, handle = found
        handle.model_ckpt(ckpt.ckpt)

    def dispatch(self, train_task: MultiPathTrainTask):
        dir_user_model = train_task.task.navi.target_dir_user_model(0)
        # print(f"node: {self.node_name}")
        # print(f"dispatch to user model: {dir_user_model}")
        # print(f"self.navi_list: {self.dir_user_models}")
        for handle in self._handles[self.dir_user_models.index_dir(dir_user_model)]:
            handle.dispatch(train_task.detail_trace_fork())
        return train_task.unique_id

    def update_loc(self, train_task: MultiPathTrainTask):
        # write rsource of subgraph
        for i, locs in self.res_location.items():
            if i == len(self.dir_user_models):
                # patch for output of subgraph
                offset = train_task.task.navi.dir_user_model.index(self.dir_user_models[0])
                actual_i = offset+i
            else:
                user_model = self.dir_user_models[i]
                actual_i = train_task.task.navi.dir_user_model.index(user_model)
            exist = train_task.resource_loc.get(actual_i, None)
            if exist is None:
                train_task.resource_loc[actual_i] = locs
            else:
                train_task.resource_loc[actual_i] = list(set(exist) | set(locs))

    def remove(self):
        for handle in self._index_by_train_node_name.values():
            _, h = handle
            h.drop(drop_remote=True)
        self._handles.clear()
        self._index_by_train_node_name.clear()
        self._index_by_dealer.clear()

    def __del__(self):
        self.remove()


class TrainControl(TrainControlBase):
    def __init__(
        self,
        ctrl_host_name: str,
        ctrl_id: str,
        base_models: List[str],
        send: Callable[[Packet, bool], None],
        user_model_config: Union[PtuningConfig, LoraConfig],
        optimizer_config: OptimizerConfig,
        scheduler_config: Optional[SchedulerConfig],
        user_model_data_provider: Callable[[str, str], UserModelDataInitConfig],
        user_model_prefix: Optional[str] = None,
        res_loc_expire_period: int = 10,
        is_root: bool = False,
    ) -> None:
        TrainControlBase.__init__(
            self=self,
            ctrl_host_name=ctrl_host_name,
            ctrl_id=ctrl_id,
            base_models=base_models,
            send=send,
            user_model_prefix=user_model_prefix,
            is_root=is_root
        )
        self.user_model_config = user_model_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.user_model_data_provider = user_model_data_provider

        self._cached_udpated_res: Optional[Dict[int, List[str]]] = None
        self.res_loc_expire_period = res_loc_expire_period
        self.minibatches_completes: int = 0

    def _mark_res_location_dirty(self):
        super()._mark_res_location_dirty()
        self._cached_udpated_res = None

    def init_target_nodes(
        self,
        providers: List[Tuple[str, int, int]],
        ckpt_period: int,
    ):
        for dealer, start, models_count in providers:
            base_models = self.base_models[start:start+models_count]
            name_pairs = gen_user_model_names(base_models, self.ctrl_id)

            cfg = TrainInit.gen(
                models=name_pairs,
                user_model_config=self.user_model_config,
                optimizer_config=self.optimizer_config,
                scheduler_config=self.scheduler_config,
                user_model_data_provider=self.user_model_data_provider
            )
            self.init_train_node(dealer, ckpt_period, cfg)
        return self

    def _train_step(
        self,
        req_version: int,
        full_batch_ids: List[str],
        minibatch_id: str,
        sample: PossibleTrainInput,
        detail_trace: bool = False
    ):
        if self.minibatches_completes % self.res_loc_expire_period == 0:
            self._cached_udpated_res = None
        res_loc = self._cached_udpated_res
        if res_loc is None:
            res_loc = {k: v for k, v in self.res_location.items()}
        packet_id = uuid4().hex
        origin = "path_probe" if self.ctrl_id.startswith("train-inspect-") else "normal"
        train_task = MultiPathTrainTask(
            task=TrainTask(
                request_version=req_version,
                minibatch_id=minibatch_id,
                batch_ids=full_batch_ids,
                navi=Navigate(
                    dir_user_model=self.dir_user_models.dir_models,
                    current=0,
                ),
                input=sample,
            ),
            resource_loc=res_loc,
            trace=[],
            detail_trace=[] if detail_trace else None,
            control_id=self.ctrl_id,
            packet_id=packet_id,
            origin=origin,
            logical_handoff_id=self.latest_handoff_id,
        )
        record_event(
            "train_task_created",
            control_id=self.ctrl_id,
            packet_id=packet_id,
            origin=origin,
            logical_handoff_id=self.latest_handoff_id,
            minibatch_id=minibatch_id,
            request_version=req_version,
            full_batch_size=len(full_batch_ids),
        )
        train_task.record_step("start")
        self.dispatch(train_task)

    async def train_step(
        self,
        req_version: int,
        full_batch_ids: List[str],
        minibatch_id: str,
        sample: PossibleTrainInput,
    ):
        ret = await self.train_step_with_detail(req_version=req_version, full_batch_ids=full_batch_ids, minibatch_id=minibatch_id, sample=sample, detail_trace=False)
        assert ret.task.loss is not None
        return ret.task.loss

    async def train_step_with_detail(
        self,
        req_version: int,
        full_batch_ids: List[str],
        minibatch_id: str,
        sample: PossibleTrainInput,
        detail_trace: bool = False
    ):
        self._train_step(req_version=req_version, full_batch_ids=full_batch_ids, minibatch_id=minibatch_id, sample=sample, detail_trace=detail_trace)
        fut: asyncio.Future[MultiPathTrainTask] = asyncio.Future()
        self.result_waitor[minibatch_id] = fut
        ret = await fut
        # only update if this node is root contorl node
        if ret.trace_invalid:
            self._cached_udpated_res = None
            ret.detail_trace = None
        else:
            self._cached_udpated_res = ret.resource_loc
            ret.record_step("finish")
        return ret


class TrainControlNode:
    def __init__(self, base_node: BaseNode) -> None:
        self.base = base_node
        self.send = self.base.send
        self.train_contorls: Dict[str, TrainControlBase] = {}
        self.base.apis[API_TrainTask] = self.on_train_task
        self.base.apis[API_TrainModelCKPT] = self.on_train_model_ckpt
        self.base.apis[API_CreateTrainNode] = self.on_create_patch_train_node
        self.base.apis[API_RemoveNode] = self.on_remove_patch_train_node
        self._node_name_to_ctrl: Dict[str, TrainControlBase] = {}
        self._dealer_to_ctrls: Dict[str, Set[TrainControlBase]] = {}
        self._user_model_to_ctrls: Dict[str, Set[TrainControlBase]] = {}
        self.patched_train_tail_nodes: Dict[str, str] = {}
        self.patched_train_tail_models: Dict[str, int] = {}
        atexit.register(self._debug_check_resource_release)

    def on_create_patch_train_node(self, node: str, cfg: CreateTrainNode):
        assert cfg.config.arch == "it"
        assert cfg.config.tail is not None
        patched_model = cfg.config.tail.dst
        patched_node = cfg.name
        self.patched_train_tail_nodes[patched_node] = patched_model
        self.patched_train_tail_models[patched_model] = 1 +\
            self.patched_train_tail_models.setdefault(patched_model, 0)

    def on_remove_patch_train_node(self, node: str, cfg: RemoveNode):
        patched_model = self.patched_train_tail_nodes.pop(cfg.name)
        self.patched_train_tail_models[patched_model] -= 1
        if not self.patched_train_tail_models[patched_model]:
            del self.patched_train_tail_models[patched_model]

    def patch_train_node(self, train_task: MultiPathTrainTask) -> MultiPathTrainTask:
        train_task.task.navi.next()
        inp = train_task.task.input
        assert isinstance(inp, TrainTailInput)
        train_task.task.input = inp.carrier
        train_task.trace.append("patch")
        return train_task

    def on_train_task(self, node: str, train_task: MultiPathTrainTask):
        # if not infer_task.task.seq_id in self.infer_controls:
        # this could happen at last generating step
        # where packet arrives after ctrl is removed
        #     return
        # self.infer_controls[infer_task.task.seq_id].on_infer_task(infer_task)
        model = train_task.task.navi.current_dir_user_model_with_extend()
        if model in self.patched_train_tail_models:
            train_task = self.patch_train_node(train_task)
            model = train_task.task.navi.current_dir_user_model_with_extend()

        ctrls = self._user_model_to_ctrls.get(model)
        if not ctrls:
            # self.base.logger.error(f"discard: {infer_task} because {model} not found")
            return
        if len(ctrls) == 1:
            for ctrl in ctrls:
                ctrl.on_train_task(train_task)
        else:
            for ctrl in ctrls:
                ctrl.on_train_task(train_task.shallow_copy())

    def on_train_model_ckpt(self, node: str, ckpt: TrainModelCKPT):
        ctrl = self._node_name_to_ctrl.get(ckpt.train_node_name)
        if not ctrl:
            return
        ctrl.on_ckpt(ckpt)

    def __setitem__(self, ctrl_id: str, ctrl: TrainControlBase):
        self.train_contorls[ctrl_id] = ctrl
        dir_user_models = ctrl.dir_user_models
        for model in dir_user_models:
            self._user_model_to_ctrls.setdefault(model, set()).add(ctrl)
        # if ctrl.has_tail:
        #     assert ctrl.seq_id is not None
        #     self.tail_recivers.setdefault(ctrl.seq_id, set()).add(ctrl_id)
        for _, handle in ctrl.all_handles:
            self._node_name_to_ctrl[handle.train_node_name] = ctrl
            self._dealer_to_ctrls.setdefault(handle.dealer_node_name, set()).add(ctrl)

    def __delitem__(self, ctrl_id: str):
        ctrl = self.train_contorls[ctrl_id]
        dir_user_models = ctrl.dir_user_models
        for dir_model in dir_user_models:
            self._user_model_to_ctrls[dir_model].remove(ctrl)
            if not self._user_model_to_ctrls[dir_model]:
                del self._user_model_to_ctrls[dir_model]
        # if ctrl.has_tail:
        #     assert ctrl.seq_id is not None
        #     recivers = self.tail_recivers.get(ctrl.seq_id)
        #     if recivers:
        #         recivers.discard(ctrl_id)
        #     if not recivers:
        #         self.tail_recivers.pop(ctrl.seq_id, None)
        visited: Set[str] = set()
        for _, handle in ctrl.all_handles:
            if handle.train_node_name in visited:
                continue
            visited.add(handle.train_node_name)
            del self._node_name_to_ctrl[handle.train_node_name]
            lst = self._dealer_to_ctrls[handle.dealer_node_name]
            lst.remove(ctrl)
            if not lst:
                del self._dealer_to_ctrls[handle.dealer_node_name]
        del self.train_contorls[ctrl_id]

    def __getitem__(self, ctrl_id: str):
        return self.train_contorls[ctrl_id]

    def get(self, ctrl_id: str):
        return self.train_contorls.get(ctrl_id)

    def reg_new_handle(self, handle: TrainNodeHandle, ctrl: TrainControlBase):
        self._node_name_to_ctrl[handle.train_node_name] = ctrl
        self._dealer_to_ctrls.setdefault(handle.dealer_node_name, set()).add(ctrl)

    def get_handle(self, train_node_name: str):
        ctrl = self._node_name_to_ctrl.get(train_node_name)
        if ctrl is None:
            return
        handle = ctrl.get_handle(train_node_name)
        assert handle is not None
        return ctrl, handle

    def remove_handle(self, train_node_name: str):
        ctrl = self._node_name_to_ctrl.get(train_node_name)
        if ctrl is None:
            return
        handle = ctrl.remove_handle(train_node_name)
        assert handle is not None
        self._node_name_to_ctrl.pop(train_node_name, None)
        dealer_name = handle.dealer_node_name
        lst = self._dealer_to_ctrls.get(dealer_name)
        if lst and ctrl in lst:
            lst.remove(ctrl)
            if not lst:
                del self._dealer_to_ctrls[dealer_name]
        return ctrl, handle

    def get_handle_by_dealer(self, dealer_node_name: str) -> Optional[List[Tuple[TrainControlBase, List[TrainNodeHandle]]]]:
        ctrls = self._dealer_to_ctrls.get(dealer_node_name)
        if ctrls is None or len(ctrls) == 0:
            return None
        targets: List[Tuple[TrainControlBase, List[TrainNodeHandle]]] = []
        for ctrl in list(ctrls):
            handles = ctrl.get_handles_by_dealer(dealer_node_name)
            assert len(handles) > 0
            targets.append((ctrl, handles))
        return targets

    async def inspect_fastest_path(
        self,
        base_models: List[str],
        keypoint_providers: List[Tuple[str, int, int]],
        hidden_state_size: int,
    ):
        await flags.InspectPostpone("train")
        force_probe = os.getenv(
            "DROFELL_FORCE_TRAIN_PATH_PROBE", "0"
        ).lower() in {"1", "true", "yes", "on"}
        can_use_synthetic_trace = (
            len(keypoint_providers)
            == len(set(start for _, start, _ in keypoint_providers))
        )
        if can_use_synthetic_trace and not force_probe:
            record_event(
                "train_path_probe_execution",
                mode="synthetic",
                candidate_edge_count=len(keypoint_providers),
            )
            trace = [""]*(max(s+l for _, s, l in keypoint_providers))
            for p, s, l in keypoint_providers:
                for _i in range(l):
                    trace[s+_i] = p
            return trace
        record_event(
            "train_path_probe_execution",
            mode="measured",
            candidate_edge_count=len(keypoint_providers),
            forced=force_probe,
        )
        has_tail = any(ModelDescribe.from_str(m).is_tail for m in base_models)
        has_head = ModelDescribe.from_str(base_models[0]).is_head
        has_decoders = any(ModelDescribe.from_str(m).is_decoders for m in base_models)

        inspect_id = f"train-inspect-{self.base.node_name}-{uuid4().hex[:8]}"

        tail_patched_base_models = base_models
        tail_patched_keypoint_providers = keypoint_providers
        if has_decoders and not has_tail:
            # patch tail
            tail_patched_base_models = base_models+["patch:tail"]
            tail_patched_keypoint_providers = keypoint_providers+[
                (self.base.node_name, len(base_models), 1)
            ]

        ctrl = TrainControl(
            ctrl_host_name=self.base.node_name,
            send=self.base.send,
            ctrl_id=inspect_id,
            base_models=tail_patched_base_models,
            user_model_config=PtuningConfig(),
            optimizer_config=OptimizerConfig(),
            scheduler_config=None,
            user_model_data_provider=lambda src, dst: UserModelDataInitConfig(init_method="reinit")

        ).init_target_nodes(
            providers=tail_patched_keypoint_providers,
            ckpt_period=0
        )
        self[inspect_id] = ctrl
        req_version = 0
        while True:
            minibatch_id = uuid4().hex
            full_batch_ids: List[str] = [minibatch_id]
            sample: PossibleTrainInput

            if has_head:
                sample = TrainHeadInput(
                    is_bfloat16=False,
                    input_lens=[128],
                    position_ids=[i for i in range(128)],
                    input_ids=[3 for _ in range(128)],
                    label_ids=[3 for _ in range(128)]
                )
            elif has_decoders:
                sample = TrainForwardMeta(
                    sample_batch_id=minibatch_id,
                    hidden_states=DeviceTensorCarrier(torch.empty(size=(128, hidden_state_size), device="cpu", requires_grad=False, dtype=torch.float16), no_gc=None),
                    input_lens=[128],
                    position_ids=[i for i in range(128)],
                    label_ids=[3 for _ in range(128)],
                    device_pos_idx=None
                )
            elif has_tail:
                assert has_tail
                sample = TrainTailInput(
                    carrier=DeviceTensorCarrier(torch.empty(size=(128, hidden_state_size), device="cpu", requires_grad=False, dtype=torch.float16), no_gc=None),
                    label_ids=[3 for _ in range(128)],
                )
            else:
                raise ValueError(f"cannot handle arch of {base_models}")

            task = await ctrl.train_step_with_detail(
                req_version=req_version,
                full_batch_ids=full_batch_ids,
                minibatch_id=minibatch_id,
                sample=sample
            )
            if task.trace_invalid:
                req_version += 1
                continue
            trace = task.trace
            assert len(trace) == len(ctrl.dir_user_models), f"trace: {trace}, dir user models: {ctrl.dir_user_models}"
            break

        assert len(trace) == len(ctrl.dir_user_models)
        del self[inspect_id]
        return trace[:len(base_models)]

    def _debug_check_resource_release(self):
        assert not self._node_name_to_ctrl, self._node_name_to_ctrl
        assert not self._dealer_to_ctrls
        assert not self.train_contorls
        assert not self.patched_train_tail_nodes
        assert not self.patched_train_tail_models
        print("train ctrl all released")
