import collections
from typing import Dict, Set, Tuple

from dealers.common.base_types import ModelDescribe
from dealers.common.brain import Profile
from ..base import BaseNode

from .protocols import API_Announce, API_RegBackboneNode, API_ReqBackboneNodes, API_ReqExtralInfo, API_ReqProvider, API_ReqTPS, Announce, ExtraInfo, ReqProvider, ReqProviderResp, ReqTPS, ReqTPSResp


class AuxNode:
    def __init__(self, base_node: BaseNode, extra_info: ExtraInfo) -> None:
        self.base = base_node
        self.extra_info = extra_info
        self.base.logger.info("node inited")
        self.announces: Dict[str, Announce] = {}
        self.model_total_tps: Dict[Tuple[str, str], int] = collections.defaultdict(lambda: 0)
        self.model_providers: Dict[str, Dict[str, Dict[str, int]]] = collections.defaultdict(dict)

        self.backbones: Set[str] = set()

        self.base.apis[API_Announce] = self.on_announce
        self.base.apis[API_ReqTPS] = self.on_req_tps
        self.base.apis[API_ReqProvider] = self.on_req_providers
        self.base.apis[API_ReqExtralInfo] = self.on_req_extra_info
        self.base.apis[API_RegBackboneNode] = self.on_regbackbone
        self.base.apis[API_ReqBackboneNodes] = self.on_reqbackbone_nodes
        self.base.link_change_cbs.append(self.on_link_change)

    def on_link_change(self, node: str, drop: bool, dsts: Set[str]):
        if drop:
            if node in self.announces:
                self.update_tps(node, self.announces[node], drop=True)
                del self.announces[node]
            self.backbones.discard(node)

    def on_regbackbone(self, node: str, data: None):
        self.backbones.add(node)
        for provider in self.announces.keys():
            self.base.send((provider, API_RegBackboneNode, node))

    def on_reqbackbone_nodes(self, node: str, ticket: int):
        self.base.send_resp(node, ticket, self.backbones)

    def on_announce(self, node: str, announce: Announce):
        self.announces[node] = announce
        self.update_tps(node, announce, drop=False)
        if announce.virtual:
            self.base.logger.info("update virtual announce")
        else:
            self.base.logger.info(f"TPS Updated:\n"+self._total_tps_str().replace("\n", "\n  "))
        self.base.logger.info(f"providers: \n  {self._all_providers_str()}")

    def on_req_tps(self, node: str, req: ReqTPS):
        models = req.models
        profiles: Dict[Tuple[str, str], int] = {}
        for m in models:
            for e in ("fwd", "fwd+bwd", "generating"):
                profiles[(m, e)] = self.model_total_tps[(m, e)]
        self.base.send_resp(node, req.ticket_id, ReqTPSResp(profiles))

    def on_req_providers(self, node: str, req: ReqProvider):
        providers = [
            (k, list(self.model_providers[k].items()))
            for k in req.models
        ]
        if req.full:
            providers = [
                (k, list(v.items())) for k, v in self.model_providers.items()
            ]
        self.base.send_resp(node, req.ticket_id, ReqProviderResp(providers))

    def on_req_extra_info(self, node: str, ticket: int):
        self.base.send_resp(node, ticket, self.extra_info)

    def _total_tps_str(self):
        grouped: Dict[str, Dict[str, int]] = {}
        for k, v in self.model_total_tps.items():
            grouped.setdefault(k[0], {})[k[1]] = v
        s = ""
        for model, perf in grouped.items():
            s += f"model:{model} "+",\t".join(f"{k}:{v}" for k, v in perf.items())+" TPS\n"
        if s == "":
            s = "no model announced"
        return s

    def _all_providers_str(self):
        s = "\n".join(
            f"model:{model}\n\t"+"\n\t".join(p for p, _ in ps.items())
            for model, ps in self.model_providers.items()
        )
        if s == "":
            s = "no model providers"
        return s

    def update_tps(self, node: str, announce: Announce, drop: bool):
        for model in sorted(list(announce.models_loaded)):
            desc = ModelDescribe.from_str(model)
            tps = announce.profile.get(desc.base_model, Profile())
            decoders = desc.num_decoders
            sign = -1 if drop else 1
            group_tps = {
                "fwd": int(tps.perdecoder_forward_tps/decoders)*sign,
                "fwd+bwd": int(tps.perdecoder_frowarad_backward_tps/decoders)*sign,
                "generating": int(tps.perdecoder_generating_tps/decoders)*sign,
            }
            for k, _tps in group_tps.items():
                self.model_total_tps[(model, k)] += _tps
            if not drop:
                self.model_providers.setdefault(model, {})[node] = group_tps
            else:
                for k in list(self.model_providers.keys()):
                    self.model_providers[k] = {
                        n: perf
                        for n, perf in self.model_providers[k].items()
                        if n != node
                    }
                    if not self.model_providers[k]:
                        del self.model_providers[k]
                if self.model_total_tps[(model, "fwd")] == 0:
                    del self.model_total_tps[(model, "fwd")]
                    del self.model_total_tps[(model, "fwd+bwd")]
                    del self.model_total_tps[(model, "generating")]
