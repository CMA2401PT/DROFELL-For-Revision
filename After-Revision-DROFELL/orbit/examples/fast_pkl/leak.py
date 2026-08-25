import asyncio
import gc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from cuda_extends.hack import tensor_from_memory_view, tensor_to_memory_view
from orbit import Context, OrbitName, Location, LaunchPad, SimpleLogger, Orbit
from orbit.link import OrbitCPNCommander, LinkDefine, LinkPerfDescribe, PacketTypeNewNode, PacketTypeNodeDrop, ContextNetExtender
import extend_asyncio as asyncio
from orbit.link import FasterSerializeObject


class DeviceTensorCarrier(FasterSerializeObject):
    def __init__(self, t: torch.Tensor, no_gc: Optional[Any]) -> None:
        self.t = t
        self.no_gc = no_gc

    def __str__(self):
        if self.t is None:
            return "<Tensor Carrier: Empty>"
        else:
            return f"<Tensor Carrier: {self.t}>"

    def free(self):
        self.t = None  # type:ignore
        self.no_gc = None

    @property
    def tensor(self) -> torch.Tensor:
        assert self.t is not None
        return self.t  # type:ignore

    def faster_reduce(self) -> Tuple[Any, List[Union[bytes, memoryview]], Optional[Any]]:
        assert self.t is not None
        assert self.t.dtype == torch.float16 or self.t.dtype == torch.bfloat16
        return (self.t.size(), self.t.dtype), [tensor_to_memory_view(self.t)], self.no_gc

    @staticmethod
    def faster_recover(extras: Any, large_data: List[Union[bytes, memoryview]], no_gc: Optional[Any]):
        mv = large_data[0]
        if not isinstance(mv, memoryview):
            mv = memoryview(mv)
        sizes, dt = extras
        t = tensor_from_memory_view(mv, sizes, dt)
        s = DeviceTensorCarrier(t, no_gc)
        return s


class FasterSerializeFoo:
    def __init__(self, data: DeviceTensorCarrier):
        self.data = data

    def __str__(self):
        return f"<Faster Serialize: {self.data}>"

    def __repr__(self):
        return self.__str__()


def sender(ctx: Context):
    async def coro(ctx: Context):
        net_extend = ContextNetExtender(ctx)
        ctx.report_ready()
        await net_extend.wait_node_net_ready()
        ctx.logger.info(f"net ready, connected nodes: {net_extend.dsts}")

        a = DeviceTensorCarrier(torch.ones((1, 3), dtype=torch.float16), None)
        b = DeviceTensorCarrier(torch.ones((1, 3), dtype=torch.bfloat16)*2, None)
        while True:
            packet = await ctx.recv()
            if packet[0] == OrbitName:
                if packet[1] == PacketTypeNewNode:
                    new_node = packet[2]
                    for i in range(3):
                        ctx.send((new_node, "multiple fast", [FasterSerializeFoo(a), FasterSerializeFoo(b)]))
                    ctx.send((new_node, "done", None))
                elif packet[1] == PacketTypeNodeDrop:
                    pass
    asyncio.get_event_loop().create_task(coro(ctx))


def reciver(ctx: Context):
    async def coro(ctx: Context):
        net_extend = ContextNetExtender(ctx)
        ctx.report_ready()
        await net_extend.wait_node_net_ready()
        ctx.logger.info(f"net ready, connected nodes: {net_extend.dsts}")

        while True:
            packet = await ctx.recv()
            if packet[0] == OrbitName:
                continue
            ctx.logger.warn(f"recv {packet}")
            if packet[1] == "done":
                gc.collect()
                objs = [o for o in gc.get_objects() if isinstance(o, FasterSerializeFoo)]
                print(len(objs), "instances alive")
    asyncio.get_event_loop().create_task(coro(ctx))


functions: Dict[str, Callable[..., None]] = {
    "sender": sender,
    "reciver": reciver,
}


def net_model(src: Location, dst: Location) -> LinkDefine:
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
        net_builder = OrbitCPNCommander(orbit, net_model)
        await orbit.init(port=7241, satellites=None)

        await net_builder.launch_and_join(loc=Location("", 0, 0,  "sender"), fn_name="sender")
        await net_builder.launch_and_join(loc=Location("", 0, 0,  "reciver"), fn_name="reciver")
        await asyncio.Future()
    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


if __name__ == "__main__":
    run_orbit()
