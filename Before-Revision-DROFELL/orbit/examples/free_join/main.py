import asyncio
import argparse
from typing import Callable, Dict, Set
from orbit import *
import extend_asyncio as asyncio
from orbit.link import OrbitCPNCommander, PacketTypeNodeDrop, ContextNetExtender
from orbit.link.cpn import NetModel


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
                    ctx.logger.warn(f"node drop: {packet[2]}: {net_extend.dsts}{net_extend.srcs}")
                    continue
            src, api, new_node = packet
            if api == "broadcast":
                ctx.logger.info(f"notified new node: {new_node}, connecting...")
                connected = await net_extend.try_connect(new_node)
                ctx.logger.info(f"notified new node: {new_node}, can init connect={connected}, dsts={net_extend.dsts}srcs={net_extend.srcs}")
                if not connected:
                    continue
                send_pref, recv_pref = await net_extend.pref_net(new_node)
                ctx.logger.info(f"{ctx.location}<->{new_node} Send:{send_pref} Recv:{recv_pref}")
    asyncio.get_event_loop().create_task(coro(ctx))


functions: Dict[str, Callable[..., None]] = {
    "aux": aux_node,
    "example": example_node
}


def init_fn():
    def on_except(_: asyncio.Loop, e: Exception):
        raise e
        # print(e)
        # import os
        # os._exit(-1)
    asyncio.config_impl("golang", True)
    loop = asyncio.get_event_loop()
    loop.on_except(on_except)
    return loop


def run_orbit():
    launch_pad = LaunchPad(init_fn)
    logger = SimpleLogger()

    async def coro():
        orbit = Orbit(launch_pad, functions, logger=logger,  log_level="info")
        OrbitCPNCommander(orbit, NetModel())
        await orbit.init(port=7241, satellites=tuple())
        await orbit.launch(Location(name="aux"), "aux")
        await orbit.launch(Location(name="node0"), "example")
        await asyncio.Future()
    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


def run_satellite(role: str):
    launch_pad = LaunchPad(init_fn)

    async def coro():
        satellite = Satellite(launch_pad=launch_pad, functions_dict=functions, host_id=role, network_hostname="localhost")
        satellite.start("localhost:7241")
        await satellite.ready()
        satellite.launch(example_node)
        await asyncio.Future()

    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", "-r", type=str, default=OrbitName)
    args, _ = parser.parse_known_args()
    role = args.role
    if role == OrbitName:
        run_orbit()
    else:
        run_satellite(role)
