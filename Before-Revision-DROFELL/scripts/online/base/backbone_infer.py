import gc
import os
import re
import time
from typing import List, Optional, Tuple

import extend_asyncio as asyncio
from dealers.common.base_types import group_model_with_split
from dealers.memory_schedule.restriction import MemoryConfig
from node.aux_node.protocols import req_extra_info
from node.base import BaseNode
from node.client.client import ClientNode
from node.dealer_node.dealer_node import BackboneSelectConfig, ModelTargets
from node.dealer_node.protocols import MultiPathInferTask
from node.path_utils import paths_to_keypoints
from operators.core.sampler.common import SamplingParams
from operators.core.tokenizer.common import Sequence, SequenceCore, StopConfig
from operators.core.tokenizer.real import ExtendTokenizer
from orbit import *
from orbit.core import SimpleOrbit
from orbit.defines import Context, Location
from orbit.link import OrbitCPNCommander
from orbit.link.cpn import NetModel, OrbitCPNCommander
from orbit.link.define import LinkDefine, LinkPerfDescribe
from orbit.link.extender import ContextNetExtender
from scripts.online import aux_node, backbone_node, models
from scripts.online.base.get_user_model import gen_user_model
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

    head, model_groups, tail = group_model_with_split(
        base_model_name=model_name,
        decoders=model_info.config.num_decoders,
        group_fn=lambda layers: [8 for _ in range(layers//8)]
    )

    models = ModelTargets.from_specific(
        decoders=model_groups, bundle={}
    )
    models.add_bundle(model_groups[0], head)
    models.add_bundle(model_groups[-1], tail)

    await ctrl.launch(
        loc=Location(name="T0-backbone-1", process=0), fn_name="backbone",
    )

    await ctrl.launch(
        loc=Location(name="T0-backbone-2", process=0), fn_name="backbone",
    )
    await asyncio.sleep(0.3)

    await ctrl.launch(
        loc=Location(name="T1-dealer-1-1", process=2), fn_name="dealer",
        models=ModelTargets.from_specific([
            "teared-Llama-2-7B-GPTQ:decoders-000-008"
        ], {
            "teared-Llama-2-7B-GPTQ:decoders-000-008": ["teared-Llama-2-7B-GPTQ:head"]
        }),
        backbone=BackboneSelectConfig.from_specific(specific=set({"T0-backbone-1"})),
        device_str="cuda:0",
    )
    await ctrl.launch(
        loc=Location(name="T1-dealer-2-1", process=2), fn_name="dealer",
        models=ModelTargets.from_specific([
            "teared-Llama-2-7B-GPTQ:decoders-008-016",
            "teared-Llama-2-7B-GPTQ:decoders-016-024"
        ], {}),
        backbone=BackboneSelectConfig.from_specific(specific=set({"T0-backbone-2"})),
        device_str="cuda:0",
        drop=80,
        drop_by_suicide=True,
    )

    await ctrl.launch(
        loc=Location(name="T1-dealer-2-2", process=1), fn_name="dealer",
        models=ModelTargets.from_specific([
            "teared-Llama-2-7B-GPTQ:decoders-024-032"
        ], {
            "teared-Llama-2-7B-GPTQ:decoders-024-032": ["teared-Llama-2-7B-GPTQ:tail"]
        }),
        backbone=BackboneSelectConfig.from_specific(specific=set({"T0-backbone-2"})),
        device_str="cuda:0",
        # drop=80,
        # drop_by_suicide=True,
    )

    for i, decoder in enumerate(models.decoders[:]):
        backup_dealer_models = models.shallow_copy()
        backup_dealer_models.decoders = [decoder]
        await ctrl.launch(
            loc=Location(name=f"T3-backup-{i}", process=0), fn_name="dealer",
            models=backup_dealer_models,
            backbone=BackboneSelectConfig.from_specific(specific=set({})),
            device_str="cuda:1",
        )
        await asyncio.sleep(0.3)

    await asyncio.sleep(2)
    await ctrl.launch(Location(name="client"), fn_name="client")
    await asyncio.Future()


def client_node(ctx: Context):
    async def coro(ctx: Context):
        ctx.report_ready()
        net_extend = ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        ctrl = ClientNode(base_node)
        user_model, build_prompt = gen_user_model("advgen-ptuning", model_info.config)

        async def test_seq(
            ctx: Context,
            seq_id: str,
            base_models: List[str],
            keypoints: List[Tuple[str, int, int]],
        ):
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
                keypoint_providers=keypoints, retrieve=True
            )
            print(infer_ctrl)

            stopped, next_token, logprobs, detail_traces = \
                await infer_ctrl.infer_step(
                    inp=prompt_ids.copy(),
                    is_prefill=True,
                    detail_trace=False
                )
            if detail_traces is not None:
                print(f"detail trace: {detail_traces}")
            seq.core.append_token_id(next_token)
            tokenizer.incrementally_decode_seq(seq)
            counter = 0
            st = time.time()
            while not stopped:
                stopped, next_token, logprobs, detail_traces = \
                    await infer_ctrl.infer_step(
                        inp=next_token,
                        is_prefill=False,
                        detail_trace=False
                    )

                seq.core.append_token_id(next_token)
                tokenizer.incrementally_decode_seq(seq)
                counter += 1
            print(seq.output_text)
            print(counter/(time.time()-st))
            if detail_traces is not None:
                print(f"detail trace: {detail_traces}")
            ctrl.remove_infer_control(seq_id)

        async def test(ctx: Context):
            # print(inspected)
            extra_info = await req_extra_info(base_node)
            base_models = extra_info.models[model_name].models
            inspected = await ctrl.inspect_fastest_path("infer", base_models, None)
            assert inspected
            keypoints = paths_to_keypoints(paths=[inspected.best])
            # keypoints = [("T0-backbone-1", 0, 2), ("T0-backbone-2", 2, 4)]
            for i in range(1):
                await test_seq(ctx, f"seq-{i}#", base_models, keypoints)
            objs = [o for o in gc.get_objects() if isinstance(o, MultiPathInferTask)]
            print(len(objs), "instances alive")
            await asyncio.sleep(0.1)
            exit(0)

        asyncio.create_task(test(ctx))

        while True:
            packet = await ctx.recv()
            base_node.on_packet(packet)
    asyncio.get_event_loop().create_task(coro(ctx))


class LimitedNetModel(NetModel):
    def can_initiate_connect(self, src: str, dst: str) -> bool:
        if "dealer" in dst:
            if "backbone" in src:
                m = re.search(r'backbone-(\d+)', src)
                assert m is not None
                pool = int(m.group(1))
                if f"dealer-{pool}" in dst:
                    return True
            return False
        return True

    def connect_perf(self, src: str, dst: str) -> LinkDefine:
        if "dealer" in dst:
            if "backbone" in src:
                m = re.search(r'backbone-(\d+)', src)
                assert m is not None
                pool = int(m.group(1))
                if f"dealer-{pool}" in dst:
                    return ("tcp", LinkPerfDescribe())
                else:
                    return ("tcp", LinkPerfDescribe(10, 128))
        return ("tcp", LinkPerfDescribe())


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
                "backbone": backbone_node
            },
            logger=SimpleLogger(),
            log_level="info"
        )
        net_builder = OrbitCPNCommander(orbit, LimitedNetModel())
        await orbit.init(port=7241, satellites=tuple())
        await amain(orbit, net_builder)
    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()
