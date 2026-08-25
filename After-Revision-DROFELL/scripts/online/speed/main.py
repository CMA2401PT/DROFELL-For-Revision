import json
import os
import time
from typing import List, Tuple

import extend_asyncio as asyncio
from node.aux_node.protocols import req_extra_info
from node.base import BaseNode
from node.dealer_node.dealer_node import ModelTargets
from node.path_utils import paths_to_keypoints
from node.agent import AgentNode
from operators.core.sampler.common import SamplingParams
from operators.core.tokenizer.common import Sequence, SequenceCore, StopConfig
from operators.core.tokenizer.real import ExtendTokenizer
from orbit import *
from orbit.core import SimpleOrbit
from orbit.defines import Context, Location
from orbit.link import OrbitCPNCommander
from orbit.link.cpn import NetModel, OrbitCPNCommander
from orbit.link.extender import ContextNetExtender
from scripts.online import aux_node, dealer_node, models

model_name = "teared-Llama-2-7B-GPTQ"
model_info = models.models_storage[model_name]


async def amain(ctrl: SimpleOrbit, net_builder: OrbitCPNCommander):
    await ctrl.launch(Location(name="aux"), fn_name="aux")

    await ctrl.launch(
        loc=Location(name="dealer-1", process=0), fn_name="dealer",
        models=ModelTargets.from_auto(model_names=[model_name], memory_limit=(4*(2**30)), decoders_limit=0),
        device_str="cuda:0",
    )
    await asyncio.sleep(0.1)
    await ctrl.launch(
        loc=Location(name="dealer-2", process=0), fn_name="dealer",
        models=ModelTargets.from_auto(model_names=[model_name], memory_limit=(4*(2**30)), decoders_limit=0),
        device_str="cuda:0",
    )
    await asyncio.sleep(0.1)
    await ctrl.launch(Location(name="agent"), fn_name="agent")
    await asyncio.Future()


def agent_node(ctx: Context):
    async def coro(ctx: Context):
        ctx.report_ready()
        net_extend = ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        ctrl = AgentNode(base_node)

        tokenizer = ExtendTokenizer.load_tokenier(
            path=model_info.tokenizer_path,
            cfg=model_info.config.tokenizer_special_tokens,
        )
        prompt = "once upon a day"
        prompt_ids = tokenizer.encode(prompt)

        seq = Sequence(core=SequenceCore(
            stop_config=StopConfig(max_tokens=128, ignore_eos=False),
            prompt_token_ids=prompt_ids.copy(),
        ), prompt=prompt)

        async def run_seq(
            seq_id: str,
            base_models: List[str],
            keypoints: List[Tuple[str, int, int]],
            seq: Sequence,
            with_detail_traces: bool = False
        ):

            infer_ctrl = ctrl.create_infer_control(
                seq_id=seq_id,
                base_models=base_models,
                seq_core=seq.core.clone(),
                sampling_params=SamplingParams(temperature=0.0, logprobs=1),
                user_model_data=None,
                keypoint_providers=keypoints,
                retrieve=False
            )

            detail_traces_record: List[List[Tuple[str, int]]] = []
            output_time_record: List[int] = []

            stopped, next_token, logprobs, detail_traces = \
                await infer_ctrl.infer_step(
                    inp=prompt_ids.copy(),
                    is_prefill=True,
                    detail_trace=with_detail_traces
                )
            output_time_record.append(time.time_ns())
            if with_detail_traces:
                assert detail_traces is not None
                detail_traces_record.append(detail_traces)
            seq.core.append_token_id(next_token)
            while not stopped:
                stopped, next_token, logprobs, detail_traces =\
                    await infer_ctrl.infer_step(
                        inp=next_token,
                        is_prefill=False,
                        detail_trace=with_detail_traces
                    )
                output_time_record.append(time.time_ns())
                if with_detail_traces:
                    assert detail_traces is not None
                    detail_traces_record.append(detail_traces)
                seq.core.append_token_id(next_token)
            tokenizer.decode_seq(seq)
            ctrl.remove_infer_control(seq_id)
            return seq, output_time_record, detail_traces_record

        async def test():
            # print(inspected)
            extra_info = await req_extra_info(base_node)
            base_models = extra_info.models[model_name].models
            inspected = await ctrl.inspect_fastest_path("infer", base_models, None)
            assert inspected
            keypoints = paths_to_keypoints(paths=[inspected.best])
            print("keypoints: ", keypoints)
            output, time_record, _ = \
                await run_seq(f"sample", base_models, keypoints, seq=seq.clone())
            print(f"sample output: {output.output_text}, rough speed: {(len(time_record)*1e9)/(time_record[-1]-time_record[0]):.2f}TPS",)
            batch_size = 8

            batch_tasks = [
                run_seq(
                    seq_id=f"seq-{i}#",
                    base_models=base_models, keypoints=keypoints,
                    seq=seq.clone(),
                    with_detail_traces=True,
                )
                for i in range(batch_size)
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            time_traces: List[List[int]] = []
            detail_time_traces: List[List[List[Tuple[str, int]]]] = []
            for i, result in enumerate(batch_results):
                seq_out = result[0]
                print(f"seq-{i} > "+seq_out.output_text.replace('\n', '\\n'))
                time_traces.append(result[1])
                detail_time_traces.append(result[2])
            total_tokens = sum(len(t) for t in time_traces)
            total_time = (max(max(t) for t in time_traces)-min(min(t) for t in time_traces))/1e9
            print(f"{total_tokens/total_time:.2f}TPS")
            with open(os.path.join(os.path.dirname(__file__), "out.json"), "w")as f:
                json.dump({
                    "trace": time_traces,
                    "detail": detail_time_traces
                }, f)
            # wait resource release
            await asyncio.sleep(1)
            exit(0)

        asyncio.create_task(test())

        while True:
            packet = await ctx.recv()
            base_node.on_packet(packet)
    asyncio.get_event_loop().create_task(coro(ctx))


if __name__ == "__main__":
    def process_init_fn():
        def on_except(_: asyncio.Loop, e: Exception):
            raise e
            os._exit(-1)
        asyncio.config_impl("golang", True)
        loop = asyncio.get_event_loop()
        loop.on_except(on_except)
        return loop

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
