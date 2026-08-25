import os
from asyncio import Task
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import extend_asyncio as asyncio
from dataset.common import ProcessedRecord
from dealers.memory_schedule.restriction import MemoryConfig
from node.aux_node.protocols import req_extra_info
from node.base import BaseNode
from node.client.client import ClientNode
from node.compute_node.train_meta import TrainHeadInput
from node.dealer_node.dealer_node import BackboneSelectConfig, ModelTargets
from node.path_utils import paths_to_keypoints
from orbit import *
from orbit.core import SimpleOrbit
from orbit.defines import Context, Location
from orbit.link import OrbitCPNCommander
from orbit.link.cpn import NetModel, OrbitCPNCommander
from orbit.link.extender import ContextNetExtender
from scripts.online import aux_node, extra_info, models
from scripts.online.base.gen_train_aux_data import gen_train_aux
from scripts.online.common_nodes import dealer_coro

model_name = "teared-Llama-2-7B-GPTQ"
model_info = models.models_storage[model_name]


def dealer_node(ctx: Context):
    async def coro(
        ctx: Context,
        device_str: str,
        models: ModelTargets,
        memory_config: Optional[MemoryConfig] = None,
        backbone: Optional[BackboneSelectConfig] = None,
        drop: int = -1,
        drop_by_suicide: bool = True,
        time_trace: bool = False,
        time_trace_with_sync_cuda: bool = False,
    ):
        base_node, dealer, init_coro = dealer_coro(ctx, device_str, models, memory_config, backbone, time_trace, time_trace_with_sync_cuda)
        asyncio.create_task(init_coro())

        packet_count = 0
        while True:
            packet_count += 1
            if drop > 0 and packet_count < drop:
                print(f"will drop @ {packet_count}=={drop}")
            if packet_count == drop:
                if drop_by_suicide:
                    for node_to_sucide in list(dealer.name_to_nodes.keys()):
                        ctx.logger.warn(f"node to sucide: {node_to_sucide}")
                        dealer.suicide_node(node_to_sucide)
                else:
                    os._exit(-1)
            packet = await ctx.recv()
            base_node.on_packet(packet)

    asyncio.get_event_loop().create_task(coro(ctx, *ctx.args.args, **ctx.args.kwargs))


async def amain(ctrl: SimpleOrbit, net_builder: OrbitCPNCommander):
    await ctrl.launch(Location(name="aux"), fn_name="aux")

    model_extra_info = extra_info.models[model_name]
    head, model_groups, tail = model_extra_info.head, model_extra_info.decoders_groups, model_extra_info.tail
    models = ModelTargets.from_specific(
        decoders=model_groups, bundle={}
    )
    models.add_bundle(model_groups[0], head)
    models.add_bundle(model_groups[-1], tail)

    await ctrl.launch(
        loc=Location(name="T0-dealer-1", process=1), fn_name="dealer",
        models=ModelTargets.from_auto([model_name], (4*(2**30))),
        device_str="cuda:0",
    )
    await asyncio.sleep(0.1)
    await ctrl.launch(
        loc=Location(name="T0-dealer-2", process=2), fn_name="dealer",
        models=ModelTargets.from_auto([model_name], (4*(2**30))),
        device_str="cuda:0",
        drop=40,
        drop_by_suicide=False,
    )

    for i, decoder in enumerate(models.decoders):
        backup_dealer_models = models.shallow_copy()
        backup_dealer_models.decoders = [decoder]
        await ctrl.launch(
            loc=Location(name=f"T1-dealer-backup-{i}", process=0), fn_name="dealer",
            models=backup_dealer_models,
            device_str="cuda:1",
        )

    await asyncio.sleep(0.1)
    await ctrl.launch(Location(name="client"), fn_name="client")
    await asyncio.Future()


def client_node(ctx: Context):
    async def coro(ctx: Context):
        ctx.report_ready()
        net_extend = ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        ctrl = ClientNode(base_node)

        async def test(ctx: Context):
            # print(inspected)
            extra_info = await req_extra_info(base_node)
            base_models = extra_info.models[model_name].models
            cfg_aux, feeder, maintainer = gen_train_aux(
                tokenizer_path=model_info.tokenizer_path,
                eos_token_id=model_info.config.tokenizer_special_tokens.eos_token_id,
                minibatch_size=2,
                batch_size=2,
            )

            inspected = await ctrl.inspect_fastest_path("train", base_models, providers=None)
            assert inspected is not None
            keypoints = paths_to_keypoints(paths=[inspected.best])
            print(keypoints)

            async def instance():
                user_model_name = "user-0#"
                train_ctrl = ctrl.create_train_control(
                    user_model_name=user_model_name,
                    base_models=base_models,
                    user_model_config=cfg_aux.user_model_config,
                    optimizer_config=cfg_aux.optimizer_config,
                    scheduler_config=cfg_aux.scheduler_config,
                    user_model_init_data_provider=cfg_aux.user_model_data_provider,
                    keypoint_providers=keypoints,
                    ckpt_period=2
                )

                async def do_one_batch(sample: Tuple[List[str], int, str, List[ProcessedRecord]]):
                    batch_ids, req_version, minibatch_id, records = sample
                    _input = TrainHeadInput.from_records(records, is_bfloat16=False)
                    loss = await train_ctrl.train_step(req_version=req_version, full_batch_ids=batch_ids, minibatch_id=minibatch_id, sample=_input)
                    print(maintainer.update_loss(loss))

                running_tasks: Tuple[Task[None], ...]
                _running_tasks: List[Task[None]] = []
                for _ in range(2):
                    sample = await feeder.get_next(wait=False)
                    assert sample is not None
                    t = asyncio.create_task(do_one_batch(sample))
                    _running_tasks.append(t)

                running_tasks = tuple(_running_tasks)

                for i in range(30):
                    sample = await feeder.get_next(wait=False)
                    if sample is None:
                        break
                    task: Task[None] = asyncio.create_task(do_one_batch(sample))
                    running_tasks = (*running_tasks, task)
                    done, pending = await asyncio.wait_first(running_tasks)
                    running_tasks = tuple(pending)

                while running_tasks:
                    done, pending = await asyncio.wait_first(running_tasks)
                    running_tasks = tuple(pending)

                ctrl.remove_train_control(user_model_name)
            await instance()
            await asyncio.sleep(1)
            exit(0)
        asyncio.create_task(test(ctx))

        while True:
            packet = await ctx.recv()
            base_node.on_packet(packet)
    asyncio.get_event_loop().create_task(coro(ctx))


def start(
    functions: Dict[str, Callable[..., None]],
    ctrl_fn: Callable[[Orbit, OrbitCPNCommander], Coroutine[Any, Any, None]],
):
    def process_init_fn():
        def on_except(_: asyncio.Loop, e: Exception):
            raise e
            os._exit(-1)
        asyncio.config_impl("golang", False)
        loop = asyncio.get_event_loop()
        loop.on_except(on_except)
        return loop

    launch_pad = LaunchPad(process_init_fn)

    async def coro():
        orbit = Orbit(
            launch_pad=launch_pad,
            functions_dict=functions,
            logger=SimpleLogger(),
            log_level="info"
        )
        net_builder = OrbitCPNCommander(orbit, NetModel())
        await orbit.init(port=7241, satellites=tuple())
        await ctrl_fn(orbit, net_builder)
    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


def process_init_fn():
    def on_except(_: asyncio.Loop, e: Exception):
        raise e
        os._exit(-1)
    asyncio.config_impl("golang", False)
    loop = asyncio.get_event_loop()
    loop.on_except(on_except)
    return loop


if __name__ == "__main__":
    launch_pad = LaunchPad(process_init_fn)

    async def coro():
        orbit = Orbit(
            launch_pad=launch_pad,
            functions_dict={
                "aux": aux_node,
                "client": client_node,
                "dealer": dealer_node,
            },
            logger=SimpleLogger(),
            log_level="info"
        )
        net_builder = OrbitCPNCommander(orbit, NetModel())
        await orbit.init(port=7241, satellites=tuple())
        await amain(orbit, net_builder)
    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()
