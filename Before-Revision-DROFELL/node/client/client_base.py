from typing import List, Literal, Optional, Set, Tuple

from dealers.common.base_types import ModelDescribe
from ..infer.infer_ctrl import InferControlNode

from ..aux_node.protocols import ExtraInfo, req_extra_info, req_providers
from ..base import BaseNode
from ..path_utils import (InspectedMultiPaths, InspectedPath, fmt_inspect_path,
                          gen_lowest_energy_paths, paths_to_edge,
                          paths_to_keypoints, sort_by_tile_priority)
from ..train.train_ctrl import TrainControlNode


class ClientBaseNode:
    def __init__(
        self,
        base_node: BaseNode,
        path_search_redundancy: int = 1
    ) -> None:
        self.base = base_node
        self.node_name = self.base.node_name
        self.logger = self.base.logger
        self.send = self.base.send
        self.infer_controls = InferControlNode(self.base)
        self.remove_infer_control = self.infer_controls.__delitem__
        self.infer_send = self.infer_controls.send
        self.infer_batch_send = self.infer_controls.infer_batch_sender.send
        self.train_controls = TrainControlNode(self.base)
        self.remove_train_control = self.train_controls.__delitem__
        self.train_send = self.train_controls.send
        self.path_search_redundancy = path_search_redundancy
        self._extra_info: Optional[ExtraInfo] = None

    async def get_extra_info(self):
        if self._extra_info is None:
            self._extra_info = await req_extra_info(self.base)
        return self._extra_info

    async def req_provider(
        self,
        base_models: List[str],
        black_list: Optional[Set[str]] = None
    ):
        _providers = await req_providers(self.base, base_models)
        _providers = [(model, [
            (d, perf)
            for d, perf in ps
            if (black_list is None or d not in black_list)
        ]) for model, ps in _providers]
        providers = [[d for (d, perf) in dealers] for (model, dealers) in _providers]

        selected_providers: List[List[str]] = []
        for ps in providers:
            group = sort_by_tile_priority(ps)
            connected_group: List[str] = []
            selected_providers.append(connected_group)
            for node in group:
                if await self.base.try_connect(node):
                    connected_group.append(node)
                if len(connected_group) == self.path_search_redundancy+1:
                    break
            if not connected_group:
                return None
        return selected_providers

    async def inspect_fastest_path(
        self,
        type: Literal["train", "infer"],
        base_models: List[str],
        providers: Optional[List[List[str]]] = None,
    ):
        if providers is None:
            providers = await self.req_provider(base_models)
        self.base.logger.info(f"inspecting path: \n  {fmt_inspect_path(base_models,providers)}")
        if providers is None:
            return None
        paths = gen_lowest_energy_paths(providers)
        if paths is None:
            self.base.logger.error(f"inspect path fail: \n  {fmt_inspect_path(base_models,providers,None)}")
            return None
        keypoints = paths_to_keypoints(paths, no_overlap=True)
        base_model = ModelDescribe.from_str(base_models[0]).base_model
        hidden_size = (await self.get_extra_info()).models[base_model].hidden_size
        impl = self.infer_controls.inspect_fastest_path if type == "infer" else self.train_controls.inspect_fastest_path
        trace = await impl(
            base_models=base_models,
            keypoint_providers=keypoints,
            hidden_state_size=hidden_size,
        )
        self.base.logger.info(f"inspected path: \n  {fmt_inspect_path(base_models,providers,trace)}")
        return InspectedPath(providers, trace)

    async def inspect_fastest_n_path(
        self,
        type: Literal["train", "infer"],
        base_models: List[str],
        allow_reuse: bool,
        max_n: int = 1,
        providers: Optional[List[List[str]]] = None,
    ):
        fastest_paths: List[List[str]] = []
        while len(fastest_paths) < max_n:
            inspected = await \
                self.inspect_fastest_path(type, base_models, providers)
            if inspected is None:
                break
            fastest_paths.append(inspected.best)
            providers = inspected.reduce_providers(allow_reuse)
            if providers is None:
                break
        if len(fastest_paths) == 0:
            return None
        return InspectedMultiPaths(fastest_paths, paths_to_keypoints(fastest_paths, no_overlap=True))

    async def lookup_replacement(self, type: Literal["train", "infer"], base_models: List[str], black_lists: Optional[Set[str]]) -> Optional[List[Tuple[str, int, int]]]:
        providers = await self.req_provider(base_models, black_lists)
        if providers is None:
            return None
        inspected = await self.inspect_fastest_path(type, base_models, providers)
        if inspected is None:
            return None
        return list(paths_to_edge(inspected.best))
