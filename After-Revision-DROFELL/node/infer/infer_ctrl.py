import atexit
import os
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, cast
from uuid import uuid4

from .. import flags
import torch


from dealers.common.base_types import DeviceTensorCarrier, ModelDescribe, gen_user_model_names, gen_user_model_name
from .sender import InferBatchSender
from ..base import BaseNode
from ..compute_node.infer_meta import InferInit, InferTask, Navigate, UserModelDataProvider
from ..dealer_node.infer_handle import InferNodeHandle, SendInferTask
from ..dealer_node.protocols import API_InferTask, API_InferPutBlocks, InferPutBlocks, MultiPathInferTask
from operators.core.tokenizer.common import SequenceCore, SequenceLengthData, StopConfig
from operators.core.sampler.common import SamplingParams

import extend_asyncio as asyncio
from orbit.defines import Packet
from utils.runtime_metrics import record_event


class InferControlBase:
    def __init__(
        self,
        ctrl_host_name: str,
        ctrl_id: str,
        base_models: List[str],
        send: Callable[[Packet, bool], None],
        send_infer_task: SendInferTask,
        user_model_prefix: Optional[str] = None,
        is_root: bool = False,
    ) -> None:
        self.node_name = ctrl_host_name
        self.ctrl_id = ctrl_id
        self.unique_str = uuid4().hex.replace("-", "")
        self.base_models = base_models
        self.send = send
        self.send_infer_task = send_infer_task
        self.is_root = is_root

        if user_model_prefix is None:
            user_model_prefix = ctrl_id
        self.user_model_prefix = user_model_prefix
        self.user_models = [
            gen_user_model_name(base_model=model, user_model_prefix=self.user_model_prefix)
            for model in self.base_models
        ]
        self.has_tail = any(
            ModelDescribe.from_str(model).is_tail
            for model in self.base_models
        )
        self._handles: Dict[int, List[InferNodeHandle]] = {}
        self._index_by_infer_node_name: Dict[str, Tuple[int, InferNodeHandle]] = {}
        self._index_by_dealer: Dict[str, Set[InferNodeHandle]] = {}
        self._tail_recivers: Dict[str, InferNodeHandle] = {}
        self._res_location_cache: Optional[Dict[int, List[str]]] = None
        self._res_location_dirty: bool = True
        # self.seq_id: Optional[str] = None
        self.result_waitor: Dict[Tuple[str, int], asyncio.Future[MultiPathInferTask]] = {}
        self.infer_node_counter: int = 0

    # def _set_seq_id(self, seq_id: str):
    #     if self.seq_id is None:
    #         self.seq_id = seq_id
    #     else:
    #         assert self.seq_id == seq_id

    def is_contigious(self):
        dealers = [0]*len(self.base_models)
        for group_i, handles in self._handles.items():
            for handle in handles:
                for i in range(handle.num_models):
                    dealers[group_i+i] += 1
        return all(p > 0 for p in dealers)

    def plain_dealers_info(self):
        dealers_info: List[Tuple[str, List[Tuple[str, str]]]] = [(model, []) for model in self.base_models]
        for group_i, handles in self._handles.items():
            for handle in handles:
                for i in range(handle.num_models):
                    dealers_info[group_i+i][1].append((handle.dealer_node_name, handle.infer_node_name))
        return dealers_info

    def __str__(self) -> str:
        dealers_info = self.plain_dealers_info()
        s = f"InferControl: {self.ctrl_id}-{self.unique_str}\n"
        for base_model, dealers in dealers_info:
            s += f"  {gen_user_model_name(base_model,self.user_model_prefix)} -> {base_model}\n    "
            if len(dealers) > 0:
                s += ", ".join(f"{infer_node}@{dealer_node}" for dealer_node, infer_node in dealers)
            else:
                s += "no provider"
            s += "\n"
        return s

    def is_realated_to_dealer(self, dealer_node_name: str):
        return dealer_node_name in self._index_by_dealer

    def has_infer_node(self, infer_node_name: str):
        return infer_node_name in self._index_by_infer_node_name

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
            # patch tail output
            ret = {len(self.base_models): [self.node_name]}
            for group_i, handles in self._handles.items():
                # 保留自 00ec44a：用 dict.fromkeys 去重而非 set()，保持顺序
                # 稳定（本节点在前、handle 顺序其后）。改回 set() 会让路由列表
                # 顺序随进程 hash 抖动，冗余路径的选择就不可复现。
                # 这一条与"并集 vs 覆盖"的取舍无关：同一 dealer 承载多个 handle
                # 时本来就该只出现一次，去重不改变集合语义、不引入顺序依赖。
                locs = list(dict.fromkeys(
                    [self.node_name]
                    + [handle.dealer_node_name for handle in handles]
                ))
                ret[group_i] = locs
            self._res_location_cache = ret
            self._res_location_dirty = False
        return self._res_location_cache

    # @property
    # def dealer_edges(self) -> Set[Tuple[str, int, int]]:
    #     edges: Set[Tuple[str, int, int]] = set()
    #     for group_i, handles in self._handles.items():
    #         for h in handles:
    #             edges.add((h.dealer_node_name, group_i, h.num_models))
    #     return edges

    def get_handles_by_dealer(self, dealer_node: str):
        handles = list(self._index_by_dealer.get(dealer_node, []))
        return handles

    def get_handle(self, infer_node_name: str):
        # only called when node state change
        found = self._index_by_infer_node_name.get(infer_node_name, None)
        if not found:
            return None
        return found[1]

    def remove_handle(self, infer_node_name: str):
        # only called when node state change
        found = self._index_by_infer_node_name.pop(infer_node_name, None)
        if not found:
            return None
        group_i, handle = found
        self._tail_recivers.pop(handle.infer_node_name, None)
        handles = self._handles[group_i]
        handles.remove(handle)
        self._mark_res_location_dirty()
        dealer_set = self._index_by_dealer[handle.dealer_node_name]
        dealer_set.discard(handle)
        if not dealer_set:
            del self._index_by_dealer[handle.dealer_node_name]
        return handle

    def init_infer_node(
        self,
        dealer: str,
        retrieve: bool,
        cfg: InferInit,
        recovery_id: Optional[str] = None,
        recovery_task_count: int = 0,
    ):
        cfg.user_models[0]
        infer_node_name = f"{self.ctrl_id}-{self.unique_str}-{self.infer_node_counter}-<{cfg.user_models[0].split(':')[-1]}>-<{cfg.user_models[-1].split(':')[-1]}>"
        self.infer_node_counter += 1
        handle = InferNodeHandle(
            send=self.send,
            send_infer_task=self.send_infer_task,
            dealer_node_name=dealer,
            retrieve=retrieve,
            init_config=cfg,
            infer_node_name=infer_node_name,
            recovery_id=recovery_id,
            recovery_task_count=recovery_task_count,
        )
        ptr = self.base_models.index(cfg.base_models[0])
        self._handles.setdefault(ptr, []).append(handle)
        self._index_by_infer_node_name[infer_node_name] = (ptr, handle)
        self._index_by_dealer.setdefault(dealer, set()).add(handle)
        if handle.has_tail:
            self._tail_recivers[infer_node_name] = handle
        #     self._set_seq_id(handle.seq_id)
        self._mark_res_location_dirty()
        return handle

    def on_infer_task(self, task: MultiPathInferTask):
        if task.task.navi.finished:
            if task.task.is_recovery_replay:
                # Recovery replay only rebuilds runtime/shadow state.  Its
                # outputs were already confirmed before recovery, so it must
                # not resolve the live request future or enter tail dedup.
                for handle in self._tail_recivers.values():
                    handle.update_recovery_result(task)
                return
            if task.trace_invalid:
                # if self heading, the trace is inrealiable
                task.trace = []
            fut = self.result_waitor.pop(task.unique_id, None)
            if fut is not None:
                fut.set_result(task)
            for handle in self._tail_recivers.values():
                handle.update_result(task)
            return
        # res loc could be change when recover/healing happens
        #
        # 这里**保持并集**，不要改成覆盖。原因见
        # 一稿返修/代码理解/13 §4.6 与 exps/恢复修复取舍说明.md：
        #
        # `00ec44a` 曾把这里改成权威覆盖 `list(locs)` 来治高故障率下的恢复失败，
        # 但事后定位表明：覆盖并不是必要修复。直接故障来自替换候选未复核
        # 连通性（TOCTOU），以及跨请求 control 生命周期串包；见
        # agent/control_plane.py、backbone/backbone_base.py 与
        # exps/恢复跨请求串包与初始化取消修复说明.md。
        # 而陈旧条目本身只造成**一次无效发送** —— `infer/sender.py` 里
        # `send_cb`(=remove_barrier) 在 `base_send` 之前无条件触发，
        # 投给已死节点不会漏放屏障、不会卡死；重复到达也被接纳逻辑幂等吸收。
        # 陈旧窗口另有 `res_loc_expire_period`（默认 10 步）兜底。
        #
        # 改成覆盖会引入概念模型层面的代价：并集是顺序无关的单调累积
        # （多径下任意顺序合并结果一致，所以无需 epoch/quorum），
        # 覆盖是 last-writer-wins ⇒ 顺序敏感、隐含"根视图权威"这一新假设，
        # 与论文主张的"resource-location map 不依赖共识"冲突。
        # 因此这里回退为并集；候选复核和 control 生命周期隔离在各自层解决。
        if self.is_root:
            for ptr, locs in self.res_location.items():
                task.resource_loc[ptr] = list(dict.fromkeys(
                    task.resource_loc.setdefault(ptr, []) + list(locs)
                ))
        self.dispatch(task)

    def on_putblocks(self, put_blocks: InferPutBlocks):
        # print(f"put blocks: {put_blocks.infer_node_name} -> {put_blocks.model} -> {put_blocks.blocks.target_blocks}")
        found = self._index_by_infer_node_name.get(put_blocks.infer_node_name)
        if not found:
            return
        _, handle = found
        handle.put_blocks(put_blocks.blocks)

    def dispatch(self, infer_task: MultiPathInferTask):
        model, _, _ = infer_task.task.navi.target(0)
        # print(f"node: {self.node_name}")
        # print(f"dispatch to user model: {model}")
        # print(f"self.navi_list: {self.user_models}")
        for handle in self._handles[self.user_models.index(model)]:
            handle.dispatch(infer_task.detail_trace_fork())
        return infer_task.unique_id

    def update_loc(self, infer_task: MultiPathInferTask, offset: int, with_last: bool):
        last = len(self.base_models)
        for i, locs in self.res_location.items():
            if not with_last and last == i:
                continue
            actual_i = i+offset
            exist = infer_task.resource_loc.get(actual_i, None)
            if exist is None:
                infer_task.resource_loc[actual_i] = locs
            else:
                infer_task.resource_loc[actual_i] = list(set(exist) | set(locs))

    def remove(self):
        for handle in self._index_by_infer_node_name.values():
            _, h = handle
            h.drop(drop_remote=True)
        self._handles.clear()
        self._index_by_infer_node_name.clear()
        self._index_by_dealer.clear()
        self._tail_recivers.clear()

    def __del__(self):
        self.remove()


class InferControl(InferControlBase):
    def __init__(
        self,
        ctrl_host_name: str,
        ctrl_id: str,
        base_models: List[str],
        send: Callable[[Packet, bool], None],
        send_infer_task: SendInferTask,
        seq_core: Union[SequenceCore, SequenceLengthData],
        sampling_params: Optional[SamplingParams],
        user_model_data: Optional[Tuple[str, UserModelDataProvider]],
        user_model_prefix: Optional[str] = None,
        res_loc_expire_period: int = 10,
        is_root: bool = False,
    ) -> None:
        InferControlBase.__init__(
            self=self,
            ctrl_host_name=ctrl_host_name,
            ctrl_id=ctrl_id,
            base_models=base_models,
            send=send,
            send_infer_task=send_infer_task,
            user_model_prefix=user_model_prefix,
            is_root=is_root
        )
        self.seq_id = ctrl_id
        self.user_model_data = user_model_data
        self.seq_core = seq_core.clone()
        self._is_seq_core = isinstance(self.seq_core, SequenceCore)
        self.sampling_params = sampling_params
        self._cached_udpated_res: Optional[Dict[int, List[str]]] = None
        self.res_loc_expire_period = res_loc_expire_period

    def _mark_res_location_dirty(self):
        super()._mark_res_location_dirty()
        self._cached_udpated_res = None

    def init_target_nodes(
        self,
        providers: List[Tuple[str, int, int]],
        retrieve: bool,
    ):
        for dealer, start, models_count in providers:
            base_models = self.base_models[start:start+models_count]
            name_pairs = gen_user_model_names(base_models, self.ctrl_id)
            assert self.seq_id is not None
            cfg = InferInit.gen(
                seq_id=self.seq_id,
                models=name_pairs,
                seq_data=self.seq_core.clone(),
                sampling_params=self.sampling_params,
                user_model_name_and_slicer=self.user_model_data,
            )
            self.init_infer_node(dealer, retrieve, cfg)
        return self

    @property
    def generating_i(self):
        return self.seq_core.output_count

    def _infer_step(self, inp: Union[List[int], int, DeviceTensorCarrier], is_prefill: bool, detail_trace: bool = False):
        if self.generating_i % self.res_loc_expire_period == 0:
            self._cached_udpated_res = None
        res_loc = self._cached_udpated_res
        if res_loc is None:
            res_loc = {k: v for k, v in self.res_location.items()}
        assert self.seq_id is not None
        infer_task = MultiPathInferTask(
            task=InferTask(
                seq_id=self.seq_id,
                step=self.generating_i,
                navi=Navigate(self.user_models),
                is_prefill=is_prefill,
                input=inp,
                output=None
            ),
            resource_loc=res_loc,
            trace=[],
            detail_trace=[] if detail_trace else None
        )
        infer_task.record_step("start")
        self.dispatch(infer_task)
        return infer_task.unique_id

    def append_token(self, token: Optional[int] = None):
        if self._is_seq_core:
            assert token is not None
            cast(SequenceCore, self.seq_core).append_token_id(token)
        else:
            cast(SequenceLengthData, self.seq_core).append_token()

    async def infer_step(self, inp: Union[List[int], int, DeviceTensorCarrier], is_prefill: bool, detail_trace: bool = False):
        ret = await self.infer_step_with_detail(inp, is_prefill=is_prefill, detail_trace=detail_trace)
        assert ret.task.output is not None
        next_token, log_probs = ret.task.output
        assert isinstance(self.seq_core, SequenceCore)
        self.append_token(next_token)
        stopped = self.seq_core.stopped
        return stopped, next_token, log_probs, ret.detail_trace

    async def infer_step_with_detail(self, inp: Union[List[int], int, DeviceTensorCarrier], is_prefill: bool, detail_trace: bool = False):
        unique_id = self._infer_step(inp, is_prefill, detail_trace=detail_trace)
        fut: asyncio.Future[MultiPathInferTask] = asyncio.Future()
        self.result_waitor[unique_id] = fut
        ret = await fut
        # only update if this node is root contorl node
        if ret.trace_invalid:
            self._cached_udpated_res = None
            ret.detail_trace = None
        else:
            self._cached_udpated_res = ret.resource_loc
            ret.record_step("finish")
        return ret


class InferControlNode:
    def __init__(
        self,
        base_node: BaseNode,
        strict_sequence_routing: bool = False,
    ) -> None:
        self.base = base_node
        self.send = self.base.send
        self.infer_batch_sender = InferBatchSender(base_node.node_name, base_node.send)
        self.strict_sequence_routing = strict_sequence_routing
        self.infer_controls: Dict[str, InferControlBase] = {}
        self._retired_control_ids: Set[str] = set()
        self.base.apis[API_InferTask] = self.on_infer_task
        self.base.apis[API_InferPutBlocks] = self.on_infer_putblocks
        self._node_name_to_ctrl: Dict[str, InferControlBase] = {}
        self._dealer_to_ctrls: Dict[str, Set[InferControlBase]] = {}
        self._user_model_to_ctrls: Dict[str, Set[InferControlBase]] = {}
        # self.tail_recivers: Dict[str, Set[str]] = {}
        atexit.register(self._debug_check_resource_release)

    def on_infer_task(self, node: str, infer_task: MultiPathInferTask):
        if self.strict_sequence_routing:
            ctrl = self.infer_controls.get(infer_task.task.seq_id)
            if ctrl is None:
                record_event(
                    "stale_infer_task_dropped",
                    control_id=infer_task.task.seq_id,
                    step=infer_task.task.step,
                    recovery_id=infer_task.recovery_id,
                    recovery_packet_id=infer_task.recovery_packet_id,
                )
                return
            ctrl.on_infer_task(infer_task)
            return
        # if not infer_task.task.seq_id in self.infer_controls:
        # this could happen at last generating step
        # where packet arrives after ctrl is removed
        #     return
        # self.infer_controls[infer_task.task.seq_id].on_infer_task(infer_task)
        model = infer_task.task.navi.current_user_model(True)
        ctrls = self._user_model_to_ctrls.get(model)
        if not ctrls:
            # self.base.logger.error(f"discard: {infer_task} because {model} not found")
            return
        if len(ctrls) == 1:
            for ctrl in ctrls:
                ctrl.on_infer_task(infer_task)
        else:
            for ctrl in ctrls:
                ctrl.on_infer_task(infer_task.shallow_copy())

    def on_infer_putblocks(self, node: str, put_blocks: InferPutBlocks):
        ctrl = self._node_name_to_ctrl.get(put_blocks.infer_node_name)
        if not ctrl:
            return
        ctrl.on_putblocks(put_blocks)

    # def create(
    #     self,
    #     seq_id: str, base_models: List[str],
    #     seq_core: SequenceCore, sampling_params: SamplingParams,
    #     user_model_data: Optional[Tuple[str, UserModelDataProvider]],
    #     keypoint_providers: List[Tuple[str, int, int]],
    #     retrieve: bool,

    # ):
    #     ctrl = InferControl(
    #         ctrl_host_name=self.base.node_name,
    #         seq_id=seq_id,
    #         base_models=base_models,
    #         send=self.base.send,
    #         seq_core=seq_core,
    #         sampling_params=sampling_params,
    #         user_model_data=user_model_data
    #     ).init_target_nodes(
    #         providers=keypoint_providers,
    #         retrieve=retrieve
    #     )

    #     return ctrl

    def __setitem__(self, ctrl_id: str, ctrl: InferControlBase):
        current = self.infer_controls.get(ctrl_id)
        if current is not None and current is not ctrl:
            raise ValueError(f"inference control already exists: {ctrl_id}")
        if self.strict_sequence_routing and ctrl_id in self._retired_control_ids:
            raise ValueError(f"inference control ID cannot be reused: {ctrl_id}")
        self.infer_controls[ctrl_id] = ctrl
        user_models = ctrl.user_models
        for model in user_models:
            self._user_model_to_ctrls.setdefault(model, set()).add(ctrl)
        # if ctrl.has_tail:
        #     assert ctrl.seq_id is not None
        #     self.tail_recivers.setdefault(ctrl.seq_id, set()).add(ctrl_id)
        for _, handle in ctrl.all_handles:
            self._node_name_to_ctrl[handle.infer_node_name] = ctrl
            self._dealer_to_ctrls.setdefault(handle.dealer_node_name, set()).add(ctrl)

    def __delitem__(self, ctrl_id: str):
        ctrl = self.infer_controls[ctrl_id]
        user_models = ctrl.user_models
        for model in user_models:
            self._user_model_to_ctrls[model].remove(ctrl)
            if not self._user_model_to_ctrls[model]:
                del self._user_model_to_ctrls[model]
        # if ctrl.has_tail:
        #     assert ctrl.seq_id is not None
        #     recivers = self.tail_recivers.get(ctrl.seq_id)
        #     if recivers:
        #         recivers.discard(ctrl_id)
        #     if not recivers:
        #         self.tail_recivers.pop(ctrl.seq_id, None)
        for _, handle in ctrl.all_handles:
            del self._node_name_to_ctrl[handle.infer_node_name]
            lst = self._dealer_to_ctrls.get(handle.dealer_node_name, None)
            if lst and ctrl in lst:
                lst.remove(ctrl)
                if not lst:
                    del self._dealer_to_ctrls[handle.dealer_node_name]
        del self.infer_controls[ctrl_id]
        if self.strict_sequence_routing:
            self._retired_control_ids.add(ctrl_id)

    def __getitem__(self, ctrl_id: str):
        return self.infer_controls[ctrl_id]

    def get(self, ctrl_id: str):
        return self.infer_controls.get(ctrl_id)

    def is_registered(self, ctrl: InferControlBase) -> bool:
        return any(current is ctrl for current in self.infer_controls.values())

    def reg_new_handle(self, handle: InferNodeHandle, ctrl: InferControlBase):
        self._node_name_to_ctrl[handle.infer_node_name] = ctrl
        self._dealer_to_ctrls.setdefault(handle.dealer_node_name, set()).add(ctrl)

    def get_handle(self, infer_node_name: str):
        ctrl = self._node_name_to_ctrl.get(infer_node_name)
        if ctrl is None:
            return
        handle = ctrl.get_handle(infer_node_name)
        assert handle is not None
        return ctrl, handle

    def remove_handle(self, infer_node_name: str):
        ctrl = self._node_name_to_ctrl.get(infer_node_name)
        if ctrl is None:
            return
        handle = ctrl.remove_handle(infer_node_name)
        assert handle is not None
        self._node_name_to_ctrl.pop(infer_node_name, None)
        dealer_name = handle.dealer_node_name
        lst = self._dealer_to_ctrls.get(dealer_name)
        if lst and ctrl in lst:
            lst.remove(ctrl)
            if not lst:
                del self._dealer_to_ctrls[dealer_name]
        return ctrl, handle

    def get_handle_by_dealer(self, dealer_node_name: str):
        ctrls = self._dealer_to_ctrls.get(dealer_node_name)
        if ctrls is None or len(ctrls) == 0:
            return None
        targets: List[Tuple[InferControlBase, List[InferNodeHandle]]] = []
        for ctrl in list(ctrls):
            handles = ctrl.get_handles_by_dealer(dealer_node_name)
            assert len(handles) > 0
            targets.append((ctrl, handles))
        self._dealer_to_ctrls.pop(dealer_node_name, None)
        return targets

    async def inspect_fastest_path(
        self,
        base_models: List[str],
        keypoint_providers: List[Tuple[str, int, int]],
        hidden_state_size: int,
    ):
        await flags.InspectPostpone("infer")
        force_probe = os.getenv(
            "DROFELL_FORCE_INFER_PATH_PROBE", "0"
        ).lower() in {"1", "true", "yes", "on"}
        can_use_synthetic_trace = (
            len(keypoint_providers)
            == len(set(start for _, start, _ in keypoint_providers))
        )
        if can_use_synthetic_trace and not force_probe:
            record_event(
                "infer_path_probe_execution",
                mode="synthetic",
                candidate_edge_count=len(keypoint_providers),
            )
            trace = [""]*(max(s+l for _, s, l in keypoint_providers))
            for p, s, l in keypoint_providers:
                for _i in range(l):
                    trace[s+_i] = p
            return trace

        record_event(
            "infer_path_probe_execution",
            mode="measured",
            candidate_edge_count=len(keypoint_providers),
            forced=force_probe,
        )

        inspect_id = f"infer-inspect-{self.base.node_name}-{uuid4().hex[:8]}"
        ctrl = InferControl(
            ctrl_host_name=self.base.node_name,
            send=self.base.send,
            send_infer_task=self.infer_batch_sender.send_no_group,
            ctrl_id=inspect_id,
            base_models=base_models,
            seq_core=SequenceCore(stop_config=StopConfig(16, ignore_eos=True)),
            sampling_params=SamplingParams(
                temperature=0.0,
                logprobs=None,
            ),
            user_model_data=None,
        ).init_target_nodes(providers=keypoint_providers, retrieve=False)
        self[inspect_id] = ctrl
        start_with_head = ModelDescribe.from_str(base_models[0]).is_head
        prefill_input = \
            [3] if start_with_head else\
            DeviceTensorCarrier(torch.zeros(dtype=torch.float16, size=(1, hidden_state_size), device="cpu"), no_gc=None)
        await ctrl.infer_step_with_detail(prefill_input, True)
        ctrl.append_token(3)

        trace = None

        while True:
            generating_input = \
                3 if start_with_head else\
                DeviceTensorCarrier(torch.zeros(dtype=torch.float16, size=(hidden_state_size,), device="cpu"), no_gc=None)
            task = await ctrl.infer_step_with_detail(generating_input, False)
            ctrl.append_token(3)
            if task.trace_invalid:
                continue
            trace = task.trace
            assert len(trace) == len(base_models), f"trace: {trace}, base_models: {base_models}"
            break
        del self[inspect_id]
        assert len(trace) == len(base_models)
        return trace

    def _debug_check_resource_release(self):
        # assert not self._node_name_to_ctrl, self._node_name_to_ctrl
        # assert not self._dealer_to_ctrls
        # assert not self._user_model_to_ctrls
        # assert not self.infer_controls
        assert not self._node_name_to_ctrl, self._node_name_to_ctrl
        assert not self._dealer_to_ctrls
        assert not self._user_model_to_ctrls
        assert not self.infer_controls
        # assert not self.tail_recivers, self.tail_recivers
        print("infer ctrl all released")
