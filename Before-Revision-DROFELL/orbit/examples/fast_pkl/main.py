import asyncio
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from attr import dataclass

from orbit import Context, OrbitName, Location, LaunchPad, SimpleLogger, Orbit
from orbit.link import OrbitCPNCommander, FasterSerializeObject, LinkDefine, LinkPerfDescribe, PacketTypeNodeDrop, ContextNetExtender
import extend_asyncio as asyncio
from orbit.link.cpn import NetModel


class FasterSerializeFoo(FasterSerializeObject):
    def __init__(self, data: str):
        self.data = data
        self.extra = f"extra-{data}"
        self.no_gc = None

    def faster_reduce(self) -> Tuple[Any, List[Union[bytes, memoryview]], Optional[Any]]:
        return self.extra, [self.data.encode()], None

    @staticmethod
    def faster_recover(extras: Any, large_data: List[Union[bytes, memoryview]], no_gc: Optional[Any]):
        data = "fast-"+bytes(large_data[0]).decode()
        ret = FasterSerializeFoo(data)
        ret.extra = extras
        ret.no_gc = no_gc
        return ret

    def __str__(self):
        return f"<Faster Serialize: {self.data}, Extra={self.extra}>"

    def __repr__(self):
        return self.__str__()


@dataclass
class SimpleContainer:
    container: List[Any]


def aux_node(ctx: Context):
    async def coro(ctx: Context):
        net_extend = ContextNetExtender(ctx)
        ctx.report_ready()
        target_nodes: Set[str] = set()
        while True:
            packet = await ctx.recv()
            if packet[0] == OrbitName:
                if packet[1] == PacketTypeNodeDrop:
                    dropped_node = packet[2]
                    ctx.logger.warn(f"node drop: {dropped_node}: {net_extend.dsts}{net_extend.srcs}")
                    target_nodes.discard(dropped_node)
                continue
            src, api, data = packet
            if api == "listen":
                target_nodes.add(src)
            elif api == "broadcast":
                for node in target_nodes:
                    if node == src:
                        continue
                    ctx.send((node, "broadcast", src))
    asyncio.get_event_loop().create_task(coro(ctx))


def example_node(ctx: Context):
    async def coro(ctx: Context):
        net_extend = ContextNetExtender(ctx)
        ctx.report_ready()
        assert await net_extend.try_connect("aux")
        ctx.send(("aux", "listen", None))
        ctx.send(("aux", "broadcast", None))

        while True:
            packet = await ctx.recv()
            if packet[0] == OrbitName:
                if packet[1] == PacketTypeNodeDrop:
                    ctx.logger.warn(
                        f"node drop: {packet[2]}: {net_extend.dsts}{net_extend.srcs}")
                continue
            src, api, new_node = packet
            if api == "broadcast":
                ctx.logger.info(f"notified new node: {new_node}, connecting...")
                connected = await net_extend.try_connect(new_node)
                if not connected:
                    continue
                ctx.logger.info(f"notified new node: {new_node}, can init connect={connected}, dsts={net_extend.dsts}srcs={net_extend.srcs}")
                send_pref, recv_pref = await net_extend.pref_net(new_node)
                ctx.logger.info(
                    f"{ctx.location}<->{new_node} Send:{send_pref} Recv:{recv_pref}")
                ctx.send((new_node, "normal bytes", b"hello"))
                ctx.send((new_node, "normal obj", {"hello": "world"}))
                ctx.send((new_node, "list/tuple objs", ["hello", b"hello"]))
                ctx.send((new_node, "single fast", FasterSerializeFoo("hello")))
                ctx.send((new_node, "single fast container", SimpleContainer([FasterSerializeFoo("S-A"), FasterSerializeFoo("S-B")])))
                ctx.send((new_node, "multiple fast", [FasterSerializeFoo(
                    "hello"), FasterSerializeFoo("world")]))
                ctx.send((new_node, "multiple fast", [
                    SimpleContainer([FasterSerializeFoo("S-A"), FasterSerializeFoo("S-B")]),
                    SimpleContainer([FasterSerializeFoo("S-C"), SimpleContainer([FasterSerializeFoo("S-D-1"), FasterSerializeFoo("S-D-2")])])
                ]))
                ctx.send((new_node, "normal obj", "done"))
                continue
            ctx.logger.info(f"recv: {packet[2]}")
    asyncio.get_event_loop().create_task(coro(ctx))


functions: Dict[str, Callable[..., None]] = {
    "node": example_node,
    "aux": aux_node
}


class NetModelLimited(NetModel):
    def connect_perf(self, src: str, dst: str) -> LinkDefine:
        return ("tcp", LinkPerfDescribe(10, 2**10))


def init_fn():
    def on_except(_: asyncio.Loop, e: Exception):
        raise e
        os._exit(-1)
    asyncio.config_impl("golang", True)
    loop = asyncio.get_event_loop()
    loop.on_except(on_except)
    return loop


def run_orbit():
    launch_pad = LaunchPad(init_fn)
    logger = SimpleLogger()

    async def coro():
        orbit = Orbit(launch_pad, functions, logger=logger,  log_level="info")
        OrbitCPNCommander(orbit, NetModelLimited())
        await orbit.init(port=7241, satellites=None)

        await orbit.launch(loc=Location("", 0, 0,  "aux"), fn_name="aux")
        for i in range(2, 4):
            new_node = f"O-{i}-0-2"
            await orbit.launch(loc=Location("", 0, 0,  new_node), fn_name="node")
        await asyncio.Future()
    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


if __name__ == "__main__":
    run_orbit()
