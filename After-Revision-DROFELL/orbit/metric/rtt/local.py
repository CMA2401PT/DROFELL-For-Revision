import asyncio
import os
import time
from typing import Callable, Dict, List
from orbit import *
import extend_asyncio as asyncio
import matplotlib.pyplot as plt
from orbit.link import OrbitCPNCommander, LinkDefine, LinkPerfDescribe, PacketTypeNewNode, PacketTypeNodeDrop, ContextNetExtender


def echo_node(ctx: Context):
    async def coro(ctx: Context):
        net_extend = ContextNetExtender(ctx)
        ctx.report_ready()
        await net_extend.wait_node_net_ready()

        while True:
            src_name, api_name, data = await ctx.recv()
            if src_name == OrbitName:
                if api_name == PacketTypeNewNode:
                    new_node = data
                    ctx.logger.info(f"new node join: {new_node}: {net_extend.dsts}{net_extend.srcs}")
                    continue
                elif api_name == PacketTypeNodeDrop:
                    ctx.logger.warn(f"node drop: {data}: {net_extend.dsts}{net_extend.srcs}")
            else:
                ctx.send((src_name, "pong", None))
    asyncio.get_event_loop().create_task(coro(ctx))


N = 1000  # ping-pong 次数


def plot_results(all_rtts: Dict[str, List[float]]):
    plt.figure(figsize=(10, 5), dpi=330)
    for mode, rtts in all_rtts.items():
        plt.plot(rtts, label=f'{mode} (avg={sum(rtts)/len(rtts):.3f} ms)')
    plt.xlabel("Round")
    plt.ylabel("RTT (ms)")
    plt.title("RTT Comparison between Parallel Methods")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "orbit.png"))
    print("Saved orbit.png")


def main(ctx: Context):
    async def coro(ctx: Context):
        target_name = ctx.args.target_name
        net_extend = ContextNetExtender(ctx)
        ctx.report_ready()
        await net_extend.wait_node_net_ready()
        rtts = []
        for _ in range(N):
            start = time.perf_counter()
            ctx.send(("echo", "ping", None))
            while True:
                src_name, api_name, data = await ctx.recv()
                if api_name == "pong":
                    break
            end = time.perf_counter()
            rtts.append((end - start) * 1000)  # ms
        print(f"RTTs: min={min(rtts):.3f} ms, max={max(rtts):.3f} ms, avg={sum(rtts)/len(rtts):.3f} ms")
        ctx.send((OrbitName, "done", rtts))
    asyncio.get_event_loop().create_task(coro(ctx))


functions: Dict[str, Callable[..., None]] = {
    "echo": echo_node,
    "main": main,
}


def net_model(src: Location, dst: Location) -> LinkDefine:
    return ("tcp", LinkPerfDescribe(0, 2**10))


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
        await orbit.init(port=7241, satellites=tuple())
        await net_builder.launch_and_join(loc=Location("", 1, 0, 2, "echo"), fn_name="echo")
        await net_builder.launch_and_join(loc=Location("", 0, 0, 2, "main_proc"), fn_name="main", target_name="proc")
        _, _, proc_rtts = await orbit.recv()
        await asyncio.sleep(0.5)
        await net_builder.launch_and_join(loc=Location("", 1, 1, 2, "main_thread"), fn_name="main", target_name="thread")
        _, _, thread_rtts = await orbit.recv()
        await asyncio.sleep(0.5)
        await net_builder.launch_and_join(loc=Location("", 1, 0, 3, "main_coro"), fn_name="main", target_name="coro")
        _, _, coro_rtts = await orbit.recv()
        plot_results({"proc": proc_rtts, "thread": thread_rtts, "coro": coro_rtts})
    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


if __name__ == "__main__":
    run_orbit()
