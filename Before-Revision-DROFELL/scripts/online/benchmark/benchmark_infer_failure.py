import asyncio
from dataclasses import dataclass
import json
import os
from typing import List, Literal, Optional, Union
from node import flags
from operators.core.lora.common import LoraConfig
import extend_asyncio as asyncio
from node.aux_node.protocols import req_extra_info
from node.base import BaseNode
from node.client.client import ClientNode
from operators.core.ptuning.common import PtuningConfig
from operators.core.sampler.common import SamplingParams
from operators.core.tokenizer.common import SequenceCore, StopConfig
from operators.core.tokenizer.real import ExtendTokenizer
from orbit import *
from orbit.defines import Context
from orbit.link import (ContextNetExtender)
from orbit.link.extender import ContextNetExtender
from scripts.online import models
from node.aux_node.protocols import req_providers
from benchmark_infer import run_benchmark_multi, gen_user_model


@dataclass
class Profile:
    model_name: str = "teared-Llama-2-7B-GPTQ"
    multi_path: int = 1
    drop_prob: float = 0
    drop_postpone: float = 0.3
    user_model_data_name: str = ""
    user_model_data_config: Optional[Union[PtuningConfig, LoraConfig]] = None
    user_model_data_file: str = ""
    should_retrieve: bool = False
    sampler_str: str = "once upon a time"
    seq_len: int = 128
    parallel_seqs: int = 1
    record_detail_trace: bool = False
    output_dir: str = ""

    async def gen_init_path(self, base_models: List[str], client_node: ClientNode):
        inspected = await client_node.inspect_fastest_n_path(
            type="infer",
            base_models=base_models,
            allow_reuse=False,
            max_n=self.multi_path,
            providers=None
        )
        assert inspected
        keypoints = inspected.keypoints
        return keypoints


async def reset_drop_prob(client_node: ClientNode, drop_prob: float):
    dealer_nodes = set(p for _, ps in await req_providers(client_node.base_node, None) for p, _ in ps)
    for dealer_node in dealer_nodes:
        tags = await client_node.base_node.req_tags(dealer_node)
        print(f"{dealer_node} {tags}")
        if 'infer_drop_reset' in tags:
            await client_node.base_node.rpc_with_args(node=dealer_node, api="infer_drop_reset", args=drop_prob, T=float)


async def run_benchmark_with_profile(client_node: ClientNode, profile: Profile):
    await reset_drop_prob(client_node=client_node, drop_prob=profile.drop_prob)

    async def InspectPostpone(kind: Literal[
            "infer",
            "train",
            "backbone.infer",
            "backbone.train"
    ]):
        await asyncio.sleep(profile.drop_postpone)
    flags.InspectPostpone = InspectPostpone
    model_info = models.models_storage[profile.model_name]
    tokenizer = ExtendTokenizer.load_tokenier(
        path=model_info.tokenizer_path,
        cfg=model_info.config.tokenizer_special_tokens,
    )
    extra_info = await req_extra_info(client_node.base_node)
    base_models = extra_info.models[profile.model_name].models
    keypoints = await profile.gen_init_path(base_models, client_node)
    print("keypoints: ", keypoints)

    user_model, build_prompt = gen_user_model(
        user_model_name=profile.user_model_data_name,
        base_model_config=model_info.config,
        user_model_config=profile.user_model_data_config,
        user_model_data_file=profile.user_model_data_file
    )

    def create_instance(name: str, prompt_ids: List[int], max_len: int):
        infer_ctrl = client_node.create_infer_control(
            seq_id=name,
            base_models=base_models,
            seq_core=SequenceCore(
                stop_config=StopConfig(max_tokens=max_len, ignore_eos=True),
                prompt_token_ids=prompt_ids.copy(),
            ),
            sampling_params=SamplingParams(temperature=0.0, logprobs=1),
            user_model_data=user_model,
            keypoint_providers=keypoints,
            retrieve=profile.should_retrieve
        )
        return infer_ctrl

    time_record_multi, detail_time_traces = await run_benchmark_multi(
        tokenizer=tokenizer,
        create_instance=create_instance,
        remove_instance=lambda instance: client_node.remove_infer_control(instance.ctrl_id),
        sample_string=build_prompt(profile.sampler_str),
        max_len=profile.seq_len,
        multi_task_count=profile.parallel_seqs,
        with_detail_trace=profile.record_detail_trace
    )
    os.makedirs(profile.output_dir, exist_ok=True)
    with open(os.path.join(profile.output_dir, "result.json"), "w") as f:
        json.dump(time_record_multi, f)
    if len(detail_time_traces[0]) > 0:
        with open(os.path.join(profile.output_dir, "traces.json"), "w") as f:
            json.dump(detail_time_traces, f)
    print("to quick review result, use the following cmd:")
    print(f"python analysis_infer.py {profile.output_dir}")

    await asyncio.sleep(0.3)


def benchmark_node(ctx: Context):
    async def benchmark_tasks(client_node: ClientNode):
        output_dir = os.environ.get("OUTPUT_DIR", "output")
        multi_path = int(os.environ.get("MULTI_PATH", "1"))
        for repeat in range(10):
            for drop_prob_name, drop_prob_p in [("drop_p0", 0), ("drop_p1e-4", 1e-4), ("drop_p1e-3", 1e-3), ("drop_p1e-2", 1e-2), ("drop_p5e-2", 5e-2)]:
                profile = Profile(output_dir=f"{output_dir}/{drop_prob_name}/repeat_{repeat}", drop_prob=drop_prob_p, seq_len=1028, multi_path=multi_path)
                await run_benchmark_with_profile(client_node=client_node, profile=profile)

    async def coro(ctx: Context):
        ctx.report_ready()
        net_extend = ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        ctrl = ClientNode(base_node)
        asyncio.create_task(benchmark_tasks(client_node=ctrl))

        while True:
            packet = await ctx.recv()
            base_node.on_packet(packet)
    asyncio.get_event_loop().create_task(coro(ctx))


def run_satellite(role: str, orbit_addr: str):
    def init_fn():
        def on_except(_: asyncio.Loop, e: Exception):
            raise e
        asyncio.config_impl("golang", True)
        loop = asyncio.get_event_loop()
        loop.on_except(on_except)
        return loop
    launch_pad = LaunchPad(init_fn)

    async def coro():
        satellite = Satellite(launch_pad=launch_pad, functions_dict={}, host_id=role, network_hostname="", stub_remote_log=True)
        satellite.start(orbit_addr)
        await satellite.ready()
        satellite.launch(benchmark_node)
        await asyncio.Future()

    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


if __name__ == "__main__":
    run_satellite(role="benchmark", orbit_addr="localhost:7241")
