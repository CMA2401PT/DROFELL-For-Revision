from typing import Any, Callable, Coroutine, List, Optional, Set


from ..aux_node.protocols import GLOBAL_AUX_NODE, API_Announce, API_RegBackboneNode, Announce, ExtraInfo, req_extra_info
from ..aux_node.aux_node import AuxNode
from ..base import BaseNode
from ..path_utils import InspectedMultiPaths, InspectedPath, paths_to_edge, paths_to_keypoints, sort_by_tile_priority
from ..base import BaseNode

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
        providers = self.get_providers(base_models, black_lists)
        if providers is None:
            return None
        inspected = await inspect_aglo(base_models, providers)
        if inspected is None:
            return None
        return list(paths_to_edge(inspected.best))
