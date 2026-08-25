import argparse
from asyncio import Task
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

import extend_asyncio as asyncio
from dealers.memory_schedule.restriction import MemoryConfig
from node.dealer_node.dealer_node import (BackboneSelectConfig, DealerNode,
                                          ModelTargets)
from node.dealer_node.protocols import MultiPathInferTask
from orbit import *
from orbit.defines import Location
from orbit.link import OrbitCPNCommander
from orbit.link.cpn import NetModel, OrbitCPNCommander
from scripts.online import aux_node, dealer_node, extra_info
from scripts.online.common_nodes import backbone_node, dealer_coro

DEFAULT_DROP_PROB: float = 0
WITH_BACKBONE = False


def embed_drop_to_dealer(dealer: DealerNode):
    print(f"dealer {dealer.base.node_name} will drop node with possibility of: {DEFAULT_DROP_PROB}")
    progress: Dict[Tuple[str, int], int] = {}
    drop_prob: float = DEFAULT_DROP_PROB
    dealer.base.tags.add("infer_drop_reset")

    def _infer_drop_reset(node: str, ticket_and_prob: Tuple[int, float]):
        nonlocal drop_prob, progress
        ticket, drop_prob = ticket_and_prob
        progress = {}
        dealer.base.logger.warn(f"drop possibility reset to: {drop_prob}")
        dealer.base.send_resp(node, ticket, drop_prob)

    dealer.base.apis["infer_drop_reset"] = _infer_drop_reset

    def on_infer_task(src: str, task: MultiPathInferTask, from_group: bool = False):
        # print(f"dealer node {dealer.base.node_name} recv: {task.task} from {src}")
        model, _, _ = task.task.navi.target(0)
        node_names = dealer.user_model_to_nodes.get(model, [])
        if len(node_names) == 0:
            dealer.base.logger.error(f"cannot find {model} for task: {task.task.unique_id}, discard")
            return
        metrics = None
        if task.detail_trace is not None:
            metrics = [("dealer-recv", time.time_ns())]

        node_id = (task.task.seq_id, task.task.navi.current)
        step = task.task.step
        if step == 0 and (not task.trace_invalid):
            # reset, it is another test
            # progress = {}
            pass
        duplicate = step <= progress.get(node_id, -1)
        if not duplicate:
            progress[node_id] = task.task.step

        for node_name in node_names:
            # print(task.trace_invalid)
            if (not duplicate) and (not task.trace_invalid) and random.random() < drop_prob:
                async def drop_coro(node_name: str):
                    await dealer.wait_node_init(node_name)
                    if node_name not in dealer.name_to_nodes:
                        return
                    dealer.base.logger.warn(f"trigger: {node_name} node drop")
                    dealer.suicide_node(node_name)
                asyncio.create_task(drop_coro(node_name))
            else:
                asyncio.create_task(dealer.handle_infer_task(node_name, task.shallow_copy(), None if metrics is None else metrics.copy()))

    dealer.on_infer_task = on_infer_task


def dealer_node(ctx: Context):
    async def coro(
        ctx: Context,
        device_str: str,
        models: ModelTargets,
        memory_config: Optional[MemoryConfig] = None,
        backbone: Optional[BackboneSelectConfig] = None,
        time_trace: bool = False,
        time_trace_with_sync_cuda: bool = False,
    ):
        base_node, dealer, init_coro = dealer_coro(ctx, device_str, models, memory_config, backbone, time_trace, time_trace_with_sync_cuda)
        embed_drop_to_dealer(dealer)

        asyncio.create_task(init_coro())

        while True:
            packet = await ctx.recv()
            base_node.on_packet(packet)

    asyncio.get_event_loop().create_task(coro(ctx, *ctx.args.args, **ctx.args.kwargs))


print("setup: 7B drop")
model_name = "teared-Llama-2-7B-GPTQ"
PORT = 7241


def PROC_INIT(busy: bool = True):
    def on_except(_: asyncio.Loop, e: Exception):
        raise e
        os._exit(-1)
    asyncio.config_impl("golang", busy_wait=busy)
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
            "backbone": backbone_node
        },
        logger=SimpleLogger(),
        log_level="info"
    )
    net_builder = OrbitCPNCommander(orbit, NetModel())
    _no_gc.append(net_builder)
    await orbit.init(port=PORT, satellites=tuple())

    await orbit.launch(Location(name="aux"), fn_name="aux")
    _, total = torch.cuda.memory.mem_get_info()

    if WITH_BACKBONE:
        await orbit.launch(
            loc=Location(name="T0-backbone-1", process=0), fn_name="backbone",
        )

    await asyncio.sleep(1)

    DEVIDES = int(os.getenv("DEVIDES", 4))
    DEVICES = int(os.getenv("DEVICES", 2))
    DUPLICATES = int(os.getenv("DUPLICATES", 1))

    model_extra_info = extra_info.models[model_name]
    head, model_groups, tail = model_extra_info.head, model_extra_info.decoders_groups, model_extra_info.tail
    assert len(model_groups) % DEVIDES == 0
    devide_size = len(model_groups)//DEVIDES

    targets = [ModelTargets.from_specific(
        decoders=model_groups[devide_size*i:devide_size*(i+1)],
        bundle={
            model_groups[0]: [head],
            model_groups[-1]: [tail]
        }
    ) for i in range(DEVIDES)]

    tasks: List[Task[None]] = []
    for target_i, target in enumerate(targets):
        if DUPLICATES > 1:
            assert DUPLICATES <= DEVICES
            for d_i in range(DUPLICATES):
                name = f"T1-dealer-{target_i}-{d_i}"
                task = asyncio.create_task(
                    coro=orbit.launch(
                        loc=Location(name=name, process=d_i), fn_name="dealer",
                        models=target,
                        device_str=f"cuda:{d_i}",
                        time_trace=setupargs.trace_time,
                        time_trace_with_sync_cuda=setupargs.sync_cuda,
                    )
                )
                tasks.append(task)
        else:
            per_device_nodes = (DEVIDES+DEVICES-1)//DEVICES
            d_i = target_i//per_device_nodes
            name = f"T1-dealer-{target_i}-{d_i}"
            task = asyncio.create_task(
                coro=orbit.launch(
                    loc=Location(name=name, process=d_i), fn_name="dealer",
                    models=target,
                    device_str=f"cuda:{d_i}",
                    time_trace=setupargs.trace_time,
                    time_trace_with_sync_cuda=setupargs.sync_cuda,
                )
            )
            tasks.append(task)
    asyncio.gather(*tasks)
    return orbit


def run_orbit(host_name: str, port: int):
    launch_pad = LaunchPad(lambda: PROC_INIT(False))
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


def run_satellite(
    orbit_addr: str,
    host_name: str,
    host_id: str,
    node_name: str,
    num_decoders: int,
    device_str: str,
    trace_time: bool = False,
    sync_cuda: bool = False,
):
    assert host_id != "", "host id should be sth like satellite-1"
    assert node_name != "", "node name should be sth like T1-dealer-1"
    launch_pad = LaunchPad(PROC_INIT)

    async def coro():
        satellite = Satellite(launch_pad=launch_pad, functions_dict={}, host_id=host_id, network_hostname=host_name)
        satellite.start(orbit_addr=orbit_addr)
        await satellite.ready()
        target = ModelTargets.from_auto(
            model_names=[model_name],
            memory_limit=0,
            decoders_limit=num_decoders,
        )
        satellite.launch(
            target_fn=dealer_node,
            node_name=node_name,
            models=target,
            device_str=device_str,
            time_trace=trace_time,
            time_trace_with_sync_cuda=sync_cuda,
        )
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
    parser.add_argument("--num-decoders", type=int, default=32, help="number of decoders to load in this node", required=False)
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use", required=False)

    args, _ = parser.parse_known_args()
    if args.host_id == OrbitName:
        run_orbit(host_name=args.host, port=args.port)
    else:
        run_satellite(
            orbit_addr=args.orbit, host_name=args.host, host_id=args.host_id,
            node_name=args.name, num_decoders=args.num_decoders, device_str=args.device
        )
