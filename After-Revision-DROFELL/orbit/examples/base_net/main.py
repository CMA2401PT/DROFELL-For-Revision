import asyncio
import argparse
from typing import Callable, Dict, Set
from orbit import *
from orbit.link import OrbitNetCommander, ContextNetExtender, PacketTypeNewNode, LinkPerfDescribe
import extend_asyncio as asyncio


def example_node(ctx: Context):
    async def coro(ctx: Context):
        net_extend = ContextNetExtender(ctx)
        ctx.report_ready()
        while True:
            packet = await ctx.recv()
            if packet[0] == OrbitName:
                if packet[1] == PacketTypeNewNode:
                    new_node = packet[2]
                    ctx.logger.info(f"new node join: {new_node}: {net_extend.dsts}{net_extend.srcs}")
                    send_pref, recv_pref = await net_extend.pref_net(new_node)
                    ctx.logger.info(f"{ctx.location}<->{new_node} Send:{send_pref} Recv:{recv_pref}")
                    continue
            ctx.logger.warn(f"recv {packet}")
    asyncio.get_event_loop().create_task(coro(ctx))


functions: Dict[str, Callable[..., None]] = {
    "node": example_node,
}


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
        net_builder = OrbitNetCommander(orbit)
        await orbit.init(port=7241, satellites=None)
        current_nodes: Set[str] = set()
        for i in range(2, 5):
            new_node = f"O-{i}-0-0"
            await orbit.launch(loc=Location("", i, 0, new_node), fn_name="node")
            for exist_node in current_nodes:
                await net_builder.create_link(new_node, exist_node, ("tcp", LinkPerfDescribe(0, 2**10)))
                await net_builder.create_link(exist_node, new_node, ("tcp", LinkPerfDescribe(0, 2**10)))
                net_builder.notify_node_join(exist_node, new_node)
                net_builder.notify_node_join(new_node, exist_node)
            current_nodes.add(new_node)
        await asyncio.sleep(3)
    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


def run_satellite(role: str):
    launch_pad = LaunchPad(init_fn)

    async def coro():
        satellite = Satellite(launch_pad=launch_pad, functions_dict=functions, host_id=role, network_hostname="localhost")
        await satellite.run("localhost:7241")

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
