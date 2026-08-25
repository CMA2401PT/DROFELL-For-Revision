import gc
import time
from typing import Any, Callable, Coroutine, Dict, List, Tuple

import data
from dealers.memory_schedule.restriction import MemoryConfigPercentage
import extend_asyncio as asyncio
from node.aux_node.protocols import req_extra_info
from node.base import BaseNode
from node.dealer_node.dealer_node import ModelTargets
from node.dealer_node.protocols import MultiPathInferTask
from node.agent import AgentNode
from node.path_utils import paths_to_keypoints
from operators.core.sampler.common import SamplingParams
from operators.core.tokenizer.common import Sequence, SequenceCore, StopConfig
from operators.core.tokenizer.real import ExtendTokenizer
from orbit import *
from orbit.core import SimpleOrbit
from orbit.defines import Context, Location
from orbit.link import OrbitCPNCommander
from orbit.link.cpn import NetModel, OrbitCPNCommander
from orbit.link.extender import ContextNetExtender
from scripts.online import aux_node, dealer_node, models, extra_info
from scripts.online.base.get_user_model import gen_user_model

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
        user_model, build_prompt = gen_user_model(
            user_model_name="advgen-ptuning",
            base_model_config=model_info.config,
            model_path=data.trained_model_path_of("ptuning_70B_80Layers.pt")
        )

        async def test_seq(ctx: Context, seq_id: str, base_models: List[str], keypoints: List[Tuple[str, int, int]]):
            tokenizer = ExtendTokenizer.load_tokenier(
                path=model_info.tokenizer_path,
                cfg=model_info.config.tokenizer_special_tokens,
            )
            prompt = "类型#裤*风格#英伦*风格#简约"
            prompt_ids = tokenizer.encode(build_prompt(prompt))

            seq = Sequence(core=SequenceCore(
                stop_config=StopConfig(max_tokens=256, ignore_eos=False),
                prompt_token_ids=prompt_ids.copy(),
            ), prompt=prompt)
            infer_ctrl = ctrl.create_infer_control(
                seq_id=seq_id,
                base_models=base_models,
                seq_core=seq.core.clone(),
                sampling_params=SamplingParams(temperature=0.0, logprobs=1),
                user_model_data=user_model,
                keypoint_providers=keypoints, retrieve=False
            )

            stopped, next_token, logprobs, detail_traces = \
                await infer_ctrl.infer_step(
                    inp=prompt_ids.copy(),
                    is_prefill=True,
                    detail_trace=False
                )
            if detail_traces is not None:
                print(f"detail traces: {detail_traces}")
            seq.core.append_token_id(next_token)
            tokenizer.incrementally_decode_seq(seq)
            counter = 0
            st = time.time()
            while not stopped:
                stopped, next_token, logprobs, detail_traces =\
                    await infer_ctrl.infer_step(
                        inp=next_token,
                        is_prefill=False,
                        detail_trace=False
                    )
                if detail_traces is not None:
                    print(f"detail traces: {detail_traces}")
                seq.core.append_token_id(next_token)
                tokenizer.incrementally_decode_seq(seq)
                counter += 1
            print(seq.output_text)
            print(counter/(time.time()-st))
            ctrl.remove_infer_control(seq_id)

        async def test(ctx: Context):
            # print(inspected)
            extra_info = await req_extra_info(base_node)
            base_models = extra_info.models[model_name].models
            inspected = await ctrl.inspect_fastest_path("infer", base_models, None)
            assert inspected
            keypoints = paths_to_keypoints(paths=[inspected.best])
            # keypoints = [("T0-dealer-1", 0, 3), ("T0-dealer-2", 3, 3)]
            for i in range(1):
                await test_seq(ctx, f"seq-{i}", base_models, keypoints)
            objs = [o for o in gc.get_objects() if isinstance(o, MultiPathInferTask)]
            print(len(objs), "instances alive")
            await asyncio.sleep(0.1)
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
