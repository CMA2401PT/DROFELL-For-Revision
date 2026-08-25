import asyncio
import argparse
import os
from typing import Set
from uuid import uuid4
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


def init_fn():
    def on_except(_: asyncio.Loop, e: Exception):
        raise e
        os._exit(-1)
    asyncio.config_impl("golang", True)
    loop = asyncio.get_event_loop()
    loop.on_except(on_except)
    return loop


def run_orbit(host_name: str, port: int):
    launch_pad = LaunchPad(init_fn)
    logger = SimpleLogger()

    async def coro():
        orbit = Orbit(
            network_hostname=host_name,
            launch_pad=launch_pad,
            functions_dict={},
            logger=logger,
            log_level="info"
        )
        net_builder = OrbitCPNCommander(orbit, NetModel())
        await orbit.init(port=port, satellites=tuple())
        await orbit.launch_local("aux", aux_node)
        await asyncio.Future()
    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


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
                ctx.logger.info("perf net")
                send_pref, recv_pref = await net_extend.pref_net(new_node)
                ctx.logger.info(f"{ctx.location}<->{new_node} Send:{send_pref} Recv:{recv_pref}")
    asyncio.get_event_loop().create_task(coro(ctx))


def run_satellite(orbit_addr: str, host_name: str, host_id: str = "", node_name: str = ""):
    if host_id == "":
        host_id = uuid4().hex
    if node_name == "":
        node_name = uuid4().hex
    launch_pad = LaunchPad(init_fn)

    async def coro():
        satellite = Satellite(launch_pad=launch_pad, functions_dict={}, host_id=host_id, network_hostname=host_name)
        satellite.start(orbit_addr=orbit_addr)
        await satellite.ready()
        satellite.launch(example_node, node_name=node_name)
        await asyncio.Future()
    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="", help="host name (e.g. ip), blank to auto deduce", required=False)
    parser.add_argument("--host-id", type=str, default=OrbitName, help="satellite host id", required=False)
    parser.add_argument("--port", type=int, default=7241, help="port of orbit", required=False)
    parser.add_argument("--orbit", type=str, default="localhost:7241", help="orbit addr", required=False)
    parser.add_argument("--name", type=str, default="", help="name of node", required=False)
    args, _ = parser.parse_known_args()
    if args.host_id == OrbitName:
        run_orbit(host_name=args.host, port=args.port)
    else:
        run_satellite(orbit_addr=args.orbit, host_name=args.host, host_id=args.host_id, node_name=args.name)
