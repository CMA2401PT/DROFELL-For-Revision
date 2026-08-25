import extend_asyncio as asyncio
from typing import Any, Awaitable, Dict, Tuple
import uuid
from ..core import LocationByName, Orbit, Packet
from .define import Builder, LinkDefine, PacketTypeNewNode


class OrbitNetCommander(Builder):
    def __init__(self, orbit: Orbit) -> None:
        self.orbit = orbit
        self.orbit.append_handler("NetBuilder.resp", self._on_callback)
        self.resps: Dict[str, asyncio.Future[Any]] = {}

    def _await_result(self) -> Tuple[str, Awaitable[Any]]:
        resp_uuid = uuid.uuid4().hex
        fut: asyncio.Future[Any] = asyncio.Future()
        self.resps[resp_uuid] = fut
        return resp_uuid, fut

    def _on_callback(self, packet: Packet):
        key, value = packet[2]
        fut: asyncio.Future[None] = self.resps.pop(key)
        fut.set_result(value)

    async def create_link(self, src: LocationByName, dst: LocationByName, link: LinkDefine) -> None:
        link_type, link_args = link
        src_handle, dst_handle = self.orbit[src], self.orbit[dst]
        src_loc, dst_loc = src_handle.location, dst_handle.location
        colocated = (
            src_loc.host == dst_loc.host
            and src_loc.process == dst_loc.process
            and src_loc.thread == dst_loc.thread
        )
        if colocated:
            link_type = "virtual"
        if link_type == "virtual":
            assert colocated
            resp_uuid, ret = self._await_result()
            dst_handle.write_cmd("NetBuilder.build", (resp_uuid, src, "reciver", link_type, link_args))
            await ret

            resp_uuid, ret = self._await_result()
            src_handle.write_cmd("NetBuilder.build", (resp_uuid, dst, "sender", link_type, None))
            await ret
        elif link_type == "tcp":
            resp_uuid, ret = self._await_result()
            dst_handle.write_cmd("NetBuilder.build", (resp_uuid, src, "reciver", link_type, link_args))
            host, port = await ret
            resp_uuid, ret = self._await_result()
            src_handle.write_cmd("NetBuilder.build", (resp_uuid, dst, "sender", link_type, (host, port)))
            await ret

    def notify_node_join(self, recv_node: LocationByName, new_node: LocationByName) -> None:
        self.orbit[recv_node].write_cmd(PacketTypeNewNode, new_node)
