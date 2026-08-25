import argparse
import os
from typing import Any, List, Literal

from dealers.memory_schedule.restriction import MemoryConfigPercentage
import extend_asyncio as asyncio
from node.dealer_node.dealer_node import ModelTargets
from orbit import *
from orbit.defines import Location
from orbit.link import OrbitCPNCommander
from orbit.link.cpn import NetModel, OrbitCPNCommander
from scripts.online import aux_node, dealer_node, models, extra_info

print("setup: 70B")

model_name = "teared-Llama-2-70B-GPTQ"
model_info = models.models_storage[model_name]

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

infer_memory_config = MemoryConfigPercentage(
    operate_reserved=100 * (2 ** 20),
    unmanaged_reserved=10*(2**20),  # 10MiB
    major_percentage=0.5,
    minor_maximum=256*(2**30)
)

train_memory_config = MemoryConfigPercentage(
    operate_reserved=3 * (2 ** 30),
    major_percentage=0.5,
    unmanaged_reserved=500*(2**20),  # 500MiB
    minor_maximum=1024*(2**20),
    minor_minimum=512*(2**20),
)


async def SETUP(launch_pad: LaunchPad, memory_config_type: Literal["infer", "train"], *args: str):
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

    model_extra_info = extra_info.models[model_name]
    head, model_groups, tail = model_extra_info.head, model_extra_info.decoders_groups, model_extra_info.tail
    models = ModelTargets.from_specific(
        decoders=model_groups, bundle={}
    )
    models.add_bundle(model_groups[0], head)
    models.add_bundle(model_groups[-1], tail)
    memory_config = {
        "infer": infer_memory_config,
        "train": train_memory_config
    }[memory_config_type]
    t0 = asyncio.create_task(orbit.launch(
        loc=Location(name="T0-dealer-1", process=0), fn_name="dealer",
        models=models.shallow_copy(
            decoders=models.decoders[:len(models.decoders)//2]
        ),
        memory_config=memory_config,
        device_str="cuda:0",
        time_trace=setupargs.trace_time,
        time_trace_with_sync_cuda=setupargs.sync_cuda
    ))

    await asyncio.sleep(1)

    t1 = asyncio.create_task(orbit.launch(
        loc=Location(name="T0-dealer-2", process=1), fn_name="dealer",
        models=models.shallow_copy(
            decoders=models.decoders[len(models.decoders)//2:]
        ),
        memory_config=memory_config,
        device_str="cuda:1",
        time_trace=setupargs.trace_time,
        time_trace_with_sync_cuda=setupargs.sync_cuda
    ))

    asyncio.gather(t0, t1)
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
    task_type: Literal["infer", "train"],
):
    assert host_id != "", "host id should be sth like satellite-1"
    assert node_name != "", "node name should be sth like T1-dealer-1"
    assert task_type in ["infer", "train"]
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
        memory_config = {
            "infer": infer_memory_config,
            "train": train_memory_config
        }[task_type]
        satellite.launch(
            target_fn=dealer_node,
            node_name=node_name,
            models=target,
            device_str=device_str,
            memory_config=memory_config,
            time_trace=task_type == "train",
            time_trace_with_sync_cuda=False,
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
    parser.add_argument("--num-decoders", type=int, default=40, help="number of decoders to load in this node", required=False)
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use", required=False)
    parser.add_argument("--task-type", type=str, default="", help="task type: infer/train", choices=["infer", "train"], required=False)

    args, _ = parser.parse_known_args()
    if args.host_id == OrbitName:
        run_orbit(host_name=args.host, port=args.port)
    else:
        run_satellite(
            orbit_addr=args.orbit, host_name=args.host, host_id=args.host_id,
            node_name=args.name, num_decoders=args.num_decoders, device_str=args.device,
            task_type=args.task_type
        )
