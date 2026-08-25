from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Type, TypeVar

import extend_asyncio as asyncio
from orbit.defines import Context, OrbitName, Packet
from orbit.link.define import (PacketTypeNewNode, PacketTypeNodeDrop, RecvPref,
                               SendPerf)

_Ret = TypeVar('_Ret')


class BaseNode:
    def __init__(
        self,
        ctx: Context,
        tags: Optional[Set[str]] = None
    ) -> None:
        self.node_name = ctx.name
        self.connections: Set[str] = set()
        self.send = ctx.send
        self._ctx = ctx
        self.perf_net: Callable[[str], Coroutine[Any, Any, Tuple[SendPerf, RecvPref]]]
        self.logger = ctx.logger
        if tags is None:
            tags = set()
        self.tags = tags
        self._ticket_couner = 0
        self._cbs: Dict[int, Tuple[str, Callable[[Any], None]]] = {}
        self.apis: Dict[str, Callable[[str, Any], None]] = {}
        self.link_change_cbs: List[Callable[[str, bool, Set[str]], None]] = []
        self.apis["req_tags"] = self._on_req_tags

    async def try_connect(self, target: str):
        ok = await self._ctx.try_connect(target)
        if ok:
            self.connections.add(target)
        return ok

    def connected(self, target: str):
        return target in self.connections

    def set_cb(self, target_node: str, cb: Callable[[Any], None]) -> int:
        self._ticket_couner += 1
        ticket = self._ticket_couner
        self._cbs[ticket] = (target_node, cb)
        return ticket

    def send_resp(self, target_node: str, ticket: int, data: Any):
        self.send((target_node, "resp", (ticket, data)))

    def _on_link_change(self, node: str, drop: bool, dsts: Set[str]):
        for cb in self.link_change_cbs:
            cb(node, drop, dsts)
        if not drop:
            self.logger.info(f"node connected: {node}")
        else:
            for k in list(self._cbs.keys()):
                tgt, cb = self._cbs[k]
                if tgt == node:
                    del self._cbs[k]
                    cb(None)
            self.logger.info(f"node disconnected: {node}")

    def _on_req_tags(self, node: str, ticket: int):
        self.send_resp(node, ticket, self.tags)

    async def req_tags(self, node: str) -> Set[str]:
        fut: asyncio.Future[Set[str]] = asyncio.Future()
        ticket = self.set_cb(target_node=node, cb=fut.set_result)
        self.send((node, "req_tags", ticket))
        tags = await fut
        return tags

    async def rpc_no_args(self, node: str, api: str, T: Type[_Ret]) -> _Ret:
        fut: asyncio.Future[_Ret] = asyncio.Future()
        ticket = self.set_cb(target_node=node, cb=fut.set_result)
        self.send((node, api, ticket))
        tags = await fut
        return tags

    async def rpc_with_args(self, node: str, api: str, args: Any, T: Type[_Ret]) -> _Ret:
        fut: asyncio.Future[_Ret] = asyncio.Future()
        ticket = self.set_cb(target_node=node, cb=fut.set_result)
        self.send((node, api, (ticket, args)))
        tags = await fut
        return tags

    def on_packet(self, packet: Packet):
        if packet[0] == OrbitName:
            change_node = packet[2]
            if packet[1] == PacketTypeNewNode:
                self.connections.add(change_node)
                self._on_link_change(packet[2], drop=False, dsts=self.connections)
            elif packet[1] == PacketTypeNodeDrop:
                if change_node in self.connections:
                    self.connections.remove(change_node)
                self._on_link_change(packet[2], drop=True, dsts=self.connections)
            return
        if packet[1] == "resp":
            ticket, data = packet[2]
            _, cb = self._cbs[ticket]
            del self._cbs[ticket]
            cb(data)
            return

        src, api, data = packet
        self.apis[api](src, data)
