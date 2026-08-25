from abc import ABC
import extend_asyncio as asyncio

from .commander import OrbitNetCommander
from ..core import Orbit, Packet
from .define import LinkDefine, LinkPerfDescribe


class NetModel(ABC):
    def can_initiate_connect(self, src: str, dst: str) -> bool:
        return True

    def connect_perf(self, src: str, dst: str) -> LinkDefine:
        return ("tcp", LinkPerfDescribe())


class OrbitCPNCommander(OrbitNetCommander):
    def __init__(self, orbit: Orbit, net_model: NetModel) -> None:
        super().__init__(orbit)
        self.net_model = net_model
        self.orbit.append_handler("NetBuilder.connect", self._create_connect)

    def _create_connect(self, packet: Packet):
        src, _, (ident, dst) = packet
        can_init = self.net_model.can_initiate_connect(src, dst)
        if not can_init:
            self.orbit.send(src, "NetBuilder.ret", (ident, False))
            return

        async def coro():
            await self.create_link(src, dst, self.net_model.connect_perf(src, dst))
            await self.create_link(dst, src, self.net_model.connect_perf(dst, src))
            self.orbit.send(src, "NetBuilder.ret", (ident, True))
            self.notify_node_join(dst, src)
        asyncio.create_task(coro())
