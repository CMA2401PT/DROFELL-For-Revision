import argparse
import os
from typing import Any, List

import torch

import extend_asyncio as asyncio
from node.dealer_node.dealer_node import ModelTargets
from orbit import *
from orbit.defines import Location
from orbit.link import OrbitCPNCommander
from orbit.link.cpn import NetModel, OrbitCPNCommander
from scripts.online import aux_node, dealer_node

print("setup: 7B basic")

model_name = "teared-Llama-2-7B-GPTQ"

PORT = 7241


def PROC_INIT():
    def on_except(_: asyncio.Loop, e: Exception):
        raise e
        os._exit(-1)
    asyncio.config_impl("golang", True)
    loop = asyncio.get_event_loop()
    loop.on_except(on_except)
    return loop


_no_gc: List[Any] = []


async def SETUP(launch_pad: LaunchPad, *args: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-time", action="store_true", default=False, help="enabled time trace")
    parser.add_argument("--sync-cuda", action="store_true", default=False, help="sync cuda when doing time trace")
    setupargs, unknown = parser.parse_known_args(args)

    orbit = Orbit(
        launch_pad=launch_pad,
        functions_dict={
            "aux": aux_node,
            "dealer": dealer_node,
        },
        logger=SimpleLogger(),
        log_level="info"
    )
    net_builder = OrbitCPNCommander(orbit, NetModel())
    _no_gc.append(net_builder)
    await orbit.init(port=PORT, satellites=tuple())

    await orbit.launch(Location(name="aux"), fn_name="aux")
    _, total = torch.cuda.memory.mem_get_info()
    await orbit.launch(
        loc=Location(name="dealer-0", process=0), fn_name="dealer",
        models=ModelTargets.from_auto(
            model_names=[model_name],
            memory_limit=int(total*0.6),
            decoders_limit=0
        ),
        device_str="cuda:0",
        time_trace=setupargs.trace_time,
        time_trace_with_sync_cuda=setupargs.sync_cuda
    )

    return orbit
