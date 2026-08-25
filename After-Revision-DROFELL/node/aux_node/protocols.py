from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import extend_asyncio as asyncio
from dealers.common.brain import Profile
from ..base import BaseNode

GLOBAL_AUX_NODE = "aux"


@dataclass
class Announce:
    total_memory: int = 0
    base_model_memory: int = 0
    profile: Dict[str, Profile] = field(default_factory=dict)
    models_loaded: Set[str] = field(default_factory=set)
    virtual: bool = False


API_Announce = "announce"

API_RegBackboneNode = "reg_backbone_node"
API_ReqBackboneNodes = "req_backbone_nodes"


async def req_backbone_nodes(base: BaseNode):
    fut: asyncio.Future[Set[str]] = asyncio.Future()
    ticket = base.set_cb(GLOBAL_AUX_NODE, fut.set_result)
    base.send((GLOBAL_AUX_NODE, API_ReqBackboneNodes, ticket))
    backbone_nodes = await fut
    return backbone_nodes


@dataclass
class ReqTPS:
    ticket_id: int
    models: List[str]


@dataclass
class ReqTPSResp:
    profile: Dict[Tuple[str, str], int]


API_ReqTPS = "req_tps"


async def req_tps(base: BaseNode, models: List[str], target_node: str = GLOBAL_AUX_NODE):
    fut: asyncio.Future[ReqTPSResp] = asyncio.Future()
    ticket = base.set_cb(target_node, fut.set_result)
    base.send((target_node, API_ReqTPS, ReqTPS(ticket_id=ticket, models=models)))
    profiles = await fut
    return profiles.profile


@dataclass
class ReqProvider:
    ticket_id: int
    models: List[str]
    full: bool = False


@dataclass
class ReqProviderResp:
    providers: List[Tuple[str, List[Tuple[str, Dict[str, int]]]]]


API_ReqProvider = "req_provider"


async def req_providers(base: BaseNode, models: Optional[List[str]], target_node: str = GLOBAL_AUX_NODE) \
        -> List[Tuple[str, List[Tuple[str, Dict[str, int]]]]]:
    fut: asyncio.Future[ReqProviderResp] = asyncio.Future()
    ticket = base.set_cb(target_node=target_node, cb=fut.set_result)
    if models is None:
        base.send((target_node, API_ReqProvider, ReqProvider(ticket_id=ticket, models=[], full=True)))
    else:
        base.send((target_node, API_ReqProvider, ReqProvider(ticket_id=ticket, models=models)))
    providers: ReqProviderResp = await fut
    return providers.providers


@dataclass
class ExtraModelInfo:
    hidden_size: int
    num_decoders: int
    decoders_groups: List[str]
    head: str
    tail: str

    @property
    def models(self):
        return [self.head, *self.decoders_groups, self.tail]


API_ReqExtralInfo = "req_extra_info"


@dataclass
class ExtraInfo:
    models: Dict[str, ExtraModelInfo] = field(default_factory=dict)
    network_name: str = ""
    network_spec_hash: str = ""


async def req_extra_info(base: BaseNode) -> ExtraInfo:
    fut: asyncio.Future[ExtraInfo] = asyncio.Future()
    ticket = base.set_cb(GLOBAL_AUX_NODE, fut.set_result)
    base.send((GLOBAL_AUX_NODE, API_ReqExtralInfo, ticket))
    return await fut
