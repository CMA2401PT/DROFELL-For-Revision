import json
import os
from typing import Any, Callable, Coroutine, List, Optional, Set


from ..aux_node.protocols import GLOBAL_AUX_NODE, API_Announce, API_RegBackboneNode, Announce, ExtraInfo, req_extra_info
from ..aux_node.aux_node import AuxNode
from ..base import BaseNode
from ..path_utils import InspectedMultiPaths, InspectedPath, paths_to_edge, paths_to_keypoints, sort_by_tile_priority
from ..base import BaseNode
from utils.runtime_metrics import record_event

InspectAlgorithm = Callable[[List[str], List[List[str]]], Coroutine[Any, Any, Optional[InspectedPath]]]


class BackboneBaseNode:
    def __init__(
        self,
        base_node: BaseNode,
        max_search_redundancy: int = 0,
        path_gen_redundancy: int = 0,
    ) -> None:
        self.base = base_node
        self.send = self.base.send
        self.logger = self.base.logger
        self.node_name = self.base.node_name

        self.base.tags.add("backbone")
        self.aux = AuxNode(self.base, ExtraInfo())
        self.announce = Announce(virtual=True)
        self.path_select_redundancy = path_gen_redundancy
        self.path_search_redundancy = max_search_redundancy
        self._extra_info: Optional[ExtraInfo] = None

        self._backbone_counter = 0
        self.base.send(("aux", API_RegBackboneNode, self.base.node_name))

    async def get_extra_info(self):
        if self._extra_info is None:
            self._extra_info = await req_extra_info(self.base)
        return self._extra_info

    def next_id(self):
        _id = self._backbone_counter
        self._backbone_counter += 1
        return _id

    def on_announce(self, node: str, announce: Announce):
        self.aux.announces[node] = announce
        self.aux.update_tps(node, announce, drop=False)
        self.announce.models_loaded = set(self.aux.model_providers.keys())
        self.base.send((GLOBAL_AUX_NODE, API_Announce, self.announce))

    def get_provider(self, base_model: str, black_list: Optional[Set[str]] = None):
        _providers =\
            self.aux.model_providers.get(base_model, {})
        providers = [
            d for d in _providers.keys()
            if (black_list is None or d not in black_list)
        ]
        providers = sort_by_tile_priority(providers)
        providers = providers[:min(len(providers), self.path_search_redundancy+1)]
        if len(providers) == 0:
            return None
        return providers

    def get_providers(self, base_models: List[str], black_list: Optional[Set[str]] = None):
        model_providers: List[List[str]] = []
        for m in base_models:
            ps = self.get_provider(m, black_list)
            if ps is None:
                return None
            model_providers.append(ps)
        return model_providers

    async def inspect_fastest_n_path(
        self,
        inspect_algo: InspectAlgorithm,
        base_models: List[str],
        allow_reuse: bool,
        max_n: int = 1,
    ):
        providers = self.get_providers(base_models)
        if providers is None:
            return None
        initial_dealers = set(json.loads(os.getenv(
            "RECOVER_MULTI_INITIAL_BACKBONE_DEALERS", "[]"
        )))
        if initial_dealers:
            providers = [
                [dealer for dealer in group if dealer in initial_dealers]
                for group in providers
            ]
            if any(not group for group in providers):
                return None
        self.base.logger.info(f"inspecting path for {base_models} from: \n  "+"\n  ".join(f"{m} {ps}" for m, ps in zip(base_models, providers)))
        fastest_paths: List[List[str]] = []
        while len(fastest_paths) < max_n:
            inspected = await inspect_algo(base_models, providers)
            if inspected is None:
                break
            fastest_paths.append(inspected.best)
            providers = inspected.reduce_providers(allow_reuse)
            if providers is None:
                break
        if len(fastest_paths) == 0:
            return None
        return InspectedMultiPaths(fastest_paths, paths_to_keypoints(fastest_paths, no_overlap=True))

    async def lookup_replacement(self, base_models: List[str], inspect_aglo: InspectAlgorithm, black_lists: Optional[Set[str]]):
        # FIX 00ec44a (2026-08-17): 与 agent/control_plane.py::lookup_replacement
        # 对称的 TOCTOU 修复 —— Backbone 侧同样要在使用候选前复核连通性。
        # 两处必须保持一致：只有 Agent 侧修、Backbone 侧不修的话，
        # 故障被 Backbone 吸收的那条路径仍会把任务交给已失效的 Dealer。
        # 差异仅在探测手段：这里用 `await self.base.try_connect(dealer)`
        # （Backbone 需要主动建连），Agent 侧用 `self.base.connected(dealer)`。
        excluded = set() if black_lists is None else set(black_lists)
        while True:
            providers = self.get_providers(base_models, excluded)
            if providers is None:
                return None
            inspected = await inspect_aglo(base_models, providers)
            if inspected is None:
                return None
            disconnected = {
                dealer
                for dealer in inspected.best
                if not await self.base.try_connect(dealer)
            }
            if not disconnected:
                return list(paths_to_edge(inspected.best))
            record_event(
                "replacement_candidate_invalidated",
                handler_node=self.node_name,
                recovery_scope="backbone",
                dealers=sorted(disconnected),
                models=base_models,
            )
            excluded.update(disconnected)
