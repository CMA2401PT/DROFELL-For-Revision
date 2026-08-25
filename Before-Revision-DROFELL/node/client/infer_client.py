from typing import List, Optional, Tuple
# import uuid
from .client_base import ClientBaseNode
from operators.core.sampler.common import SamplingParams
from operators.core.tokenizer.common import SequenceCore
from ..dealer_node.infer_handle import InferNodeHandle
from ..compute_node.infer_meta import InferTask, UserModelDataProvider
from ..infer.infer_ctrl import InferControl, InferControlBase
from ..dealer_node.protocols import API_InferTask, GroupedInferTask, MultiPathInferTask

import extend_asyncio as asyncio


class ClientInferNode:
    def __init__(
        self,
        base_node: ClientBaseNode,
    ) -> None:
        self.base = base_node
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
    ):
        # this replace breaker, as replacement is finished without async
        # thus node data will not change in replacement
        recover_tasks: List[Tuple[str, InferTask]] = []
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
                    new_init_cfg, recover_extra = new_instace_data.gen_init_cfg()

                    new_handle = infer_control.init_infer_node(dealer=dealer_node, retrieve=handle.retrieve, cfg=new_init_cfg)
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
        for dealer_node, task in recover_tasks:
            wrapped_task = MultiPathInferTask(task=task, resource_loc=infer_control.res_location, trace=[], trace_invalid=True)
            print(f"send recover delta: {wrapped_task} -> {dealer_node}")
            self.base.send((dealer_node, API_InferTask, wrapped_task))

    def on_grouped_infer_task(self, src: str, grouped_task: GroupedInferTask):
        for t in grouped_task.tasks:
            self.on_infer_task(src, t)

    def on_infer_task(self, node: str, infer_task: MultiPathInferTask):
        infer_task.record_step("root")
        self.base.infer_controls.on_infer_task(node, infer_task)

    def recover_handles_by_infer_node(self, infer_control: InferControlBase, effected_handle: InferNodeHandle, report_dealer_node: str):
        async def coro():
            # avoid using dealer node which just report sucide, as the resource could be insufficient
            # lookup_uuid = uuid.uuid4().hex
            await effected_handle.recover_lock.acquire()
            # print(f"!!! recover {lookup_uuid} start")
            replacement = await self.base.lookup_replacement("infer", effected_handle.base_models, {report_dealer_node})
            if replacement is None:
                self.base.logger.warn(f"no replacement for {effected_handle.base_models} found, ignore blacklists")
                replacement = await self.base.lookup_replacement("infer", effected_handle.base_models, None)
            # print(f"!!! recover {lookup_uuid} found {replacement} for {effected_handle.base_models}")
            effected_handle.recover_lock.release()
            self.recover_handles([(effected_handle, replacement)], infer_control)
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
            # lookup_uuid = uuid.uuid4().hex
            replacements: List[Tuple[InferNodeHandle, Optional[List[Tuple[str, int, int]]]]] = []
            # print(f"recover {lookup_uuid} start")
            for handle in effected_handles:
                await handle.recover_lock.acquire()
                replacement = await self.base.lookup_replacement("infer", handle.base_models, {dropped_node})
                if replacement is None:
                    self.base.logger.warn(f"no replacement for {handle.base_models} found, ignore blacklists")
                    replacement = await self.base.lookup_replacement("infer", handle.base_models, None)
                # print(f"reocver {lookup_uuid} found {replacement} for {handle.base_models}")
                handle.recover_lock.release()
                replacements.append((handle, replacement))
            # print(f"recover {lookup_uuid} end")
            self.recover_handles(replacements, infer_control)
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
