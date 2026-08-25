from asyncio import Task
from typing import Any, Callable, Coroutine, Dict, List, Tuple

from dataset.common import ProcessedRecord
from dealers.memory_schedule.restriction import MemoryConfigPercentage
import extend_asyncio as asyncio
from node.aux_node.protocols import req_extra_info
from node.base import BaseNode
from node.agent import AgentNode
from node.compute_node.train_meta import TrainHeadInput
from node.dealer_node.dealer_node import ModelTargets
from node.path_utils import paths_to_keypoints
from scripts.online.base.gen_train_aux_data import gen_train_aux
from orbit import *
from orbit.core import SimpleOrbit
from orbit.defines import Context, Location
from orbit.link import OrbitCPNCommander
from orbit.link.cpn import NetModel, OrbitCPNCommander
from orbit.link.extender import ContextNetExtender
from scripts.online import aux_node, dealer_node, models, extra_info

model_name = "teared-Llama-2-70B-GPTQ"
model_info = models.models_storage[model_name]


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
        models=models.shallow_copy(
            decoders=models.decoders[:len(models.decoders)//2]
        ),
        memory_config=MemoryConfigPercentage(),
        device_str="cuda:0",
    )
    await ctrl.launch(
        loc=Location(name="T0-dealer-2", process=2), fn_name="dealer",
        models=models.shallow_copy(
            decoders=models.decoders[len(models.decoders)//2:]
        ),
        memory_config=MemoryConfigPercentage(),
        device_str="cuda:1",
    )

    await asyncio.sleep(1)
    await ctrl.launch(Location(name="agent"), fn_name="agent")
    await asyncio.Future()


def agent_node(ctx: Context):
    async def coro(ctx: Context):
        ctx.report_ready()
        net_extend = ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        ctrl = AgentNode(base_node)

        async def test(ctx: Context):
            # print(inspected)
            extra_info = await req_extra_info(base_node)
            base_models = extra_info.models[model_name].models
            cfg_aux, feeder, maintainer = gen_train_aux(
                tokenizer_path=model_info.tokenizer_path,
                eos_token_id=model_info.config.tokenizer_special_tokens.eos_token_id,
                minibatch_size=1,
                batch_size=4,
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
                    user_model_init_config=cfg_aux.user_model_init_config,
                    keypoint_providers=keypoints,
                    ckpt_period=2
                )

                async def do_one_batch(sample: Tuple[List[str], int, str, List[ProcessedRecord]]):
                    batch_ids, req_version, minibatch_id, records = sample
                    _input = TrainHeadInput.from_records(records, is_bfloat16=False)
                    loss = await train_ctrl.train_step(req_version=req_version, full_batch_ids=batch_ids, minibatch_id=minibatch_id, sample=_input)
                    print(maintainer.update_loss(loss))

                running_tasks: Tuple[Task[None], ...] = tuple([
                    asyncio.create_task(do_one_batch(await feeder.get_next(wait=False)))
                    for _ in range(2)
                ])
                for i in range(30):
                    sample = await feeder.get_next(wait=False)
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
                "agent": agent_node,
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
