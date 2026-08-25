import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch

from operators.core.ptuning.real import Ptuning

from operators.core.lora.real import Lora

from operators.core.lora.common import LoraConfig
from operators.core.lora.render import LoraRenderConfig

from dealers.common.base_types import DeviceTensorCarrier
import extend_asyncio as asyncio
from node.aux_node.protocols import req_extra_info
from node.base import BaseNode
from node.client.client import ClientNode
from node.infer.infer_ctrl import InferControl
from operators.core.ptuning.common import PtuningConfig
from operators.core.sampler.common import SamplingParams
from operators.core.tokenizer.common import Sequence, SequenceCore, StopConfig
from operators.core.tokenizer.real import ExtendTokenizer
from operators.helpers.configs.quant_llama import LlamaConfig
from orbit import *
from orbit.defines import Context
from orbit.link import (ContextNetExtender)
from orbit.link.extender import ContextNetExtender
from scripts.online import models
from scripts.online.benchmark.analysis_infer import Analyzer
import importlib


def gen_user_model(
        user_model_name: str,
        base_model_config: LlamaConfig,
        user_model_config: Union[LoraConfig, PtuningConfig, None],
        user_model_data_file: str
    ) -> Tuple[
    Optional[Tuple[
        str, Callable[[Tuple[int, int]], Tuple[
            List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]],
            Optional[LoraRenderConfig]
        ]]
    ]],
    Callable[[str], str]
]:
    if user_model_config is None:
        assert user_model_name == ""
        return None, lambda prompt=str: prompt
    assert user_model_name != ""
    state_file = user_model_data_file
    lora_cfg = None
    cfg = user_model_config
    is_lora = isinstance(cfg, LoraConfig)
    if is_lora:
        lora_cfg = cfg.render_config
        cfg.check_or_set_decoders_config(
            num_layers=base_model_config.num_decoders,
            head_dim=base_model_config.head_dim,
            num_key_value_heads=base_model_config.num_key_value_heads,
            num_query_heads=base_model_config.num_query_heads
        )
    else:
        cfg.check_or_set_decoders_config(
            num_layers=base_model_config.num_decoders,
            num_key_value_heads=base_model_config.num_key_value_heads,
            head_dim=base_model_config.head_dim
        )
    user_model = Lora(cfg) if is_lora else Ptuning(cfg)
    state_dict = torch.load(state_file, map_location="cpu")
    user_model.load_state_dict(state_dict)
    user_data =\
        cast(Lora, user_model).get_inference_lora_blocks(torch.float16) \
        if is_lora else \
        cast(Ptuning, user_model).get_inference_ptuning_kv_cache_with_convert(torch.float16)
    user_data = [
        [
            (DeviceTensorCarrier.from_data(k.cpu()), DeviceTensorCarrier.from_data(v.cpu()))
            for (k, v) in layer_blocks
        ]
        for layer_blocks in user_data
    ]

    def slice(range: Tuple[int, int]) ->\
        Tuple[
            List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]],
            Optional[LoraRenderConfig]
    ]:
        return user_data[range[0]:range[1]], lora_cfg
    return (user_model_name, slice), cfg.build_prompt


class InferInstances:
    def __init__(self,
                 instances: List[InferControl],
                 prompt_token_ids: List[List[int]],
                 with_detail_traces: bool,
                 ) -> None:
        self.instances = instances
        self.prompt_token_ids = prompt_token_ids
        self.with_detail_traces = with_detail_traces
        self.detail_traces_record: List[List[List[Tuple[str, int]]]] = [[] for _ in range(len(instances))]
        self.output_time_record: List[List[int]] = [[] for _ in range(len(instances))]

    async def run_prefill(self):
        outs = await asyncio.gather(*[
            instance.infer_step(
                inp=prompt_token_ids.copy(),
                is_prefill=True,
                detail_trace=self.with_detail_traces
            )
            for instance, prompt_token_ids in zip(self.instances, self.prompt_token_ids)
        ])
        self._init_tokens = [token for _, token, _, _ in outs]
        return outs

    async def run_generating(self):
        queue: asyncio.Queue[Tuple[int, bool, int, Dict[int, float], Optional[List[Tuple[str, int]]]]] = asyncio.Queue()

        async def run_each(seq_i: int, instance: InferControl, next_token: int):
            stopped = False
            while not stopped:
                stopped, next_token, logprobs, detail_traces =\
                    await instance.infer_step(
                        inp=next_token,
                        is_prefill=False,
                        detail_trace=self.with_detail_traces
                    )
                queue.put_nowait((seq_i,  stopped, next_token, logprobs, detail_traces))
        running_tasks = [
            asyncio.create_task(run_each(i, instance, init_token))
            for i, (instance, init_token) in enumerate(zip(self.instances, self._init_tokens))
        ]
        unfinished_count = len(self.instances)
        while unfinished_count > 0:
            out = await queue.get()
            if out[1]:
                unfinished_count -= 1
            yield out
        del running_tasks


def record_time_to_current_speed(record_time: List[int]):
    if len(record_time) < 31:
        return 0
    dt = record_time[-1]-record_time[-31]
    return 30/(dt/(10**9))


async def run_benchmark_multi(
    tokenizer: ExtendTokenizer,
    create_instance: Callable[[str, List[int], int], InferControl],
    remove_instance: Callable[[InferControl], None],
    sample_string: str = "once upon a time",
    max_len: int = 20*1000,
    multi_task_count: int = 8,
    with_detail_trace: bool = False,
):

    # do warm up
    print("warm up")
    prompt_tokens = tokenizer.encode(sample_string)
    instances = [
        create_instance(f"warmup-{i}#", prompt_tokens.copy(), 32)
        for i in range(multi_task_count)
    ]
    multi_instances = InferInstances(instances, [prompt_tokens.copy() for _ in range(multi_task_count)], with_detail_traces=with_detail_trace)
    seqs = [
        Sequence(SequenceCore(StopConfig(max_tokens=max_len), prompt_token_ids=prompt_tokens.copy()))
        for _ in range(multi_task_count)
    ]

    outs = await multi_instances.run_prefill()
    for seq_i, (_, token_id, _, trace) in enumerate(outs):
        seqs[seq_i].core.append_token_id(token_id)
        if with_detail_trace:
            assert trace is not None
    for seq in seqs:
        tokenizer.incrementally_decode_seq(seq)

    time_record: List[int] = []
    async for out in multi_instances.run_generating():
        seq_i, _, token_id, _, _ = out
        time_record.append(time.time_ns())
        seqs[seq_i].core.append_token_id(token_id)
        tokenizer.incrementally_decode_seq(seqs[seq_i])
        for seq in seqs:
            print(seq.output_text)
        print(f"TPS quick glance: total {record_time_to_current_speed(time_record):.2f}", end=f"\033[{multi_task_count}F")
    print("\n"*multi_task_count)

    for instance in instances:
        remove_instance(instance)
    await asyncio.sleep(0.2)

    print("benchmarking")
    detail_time_traces: List[List[List[Tuple[str, int]]]] = [[] for _ in range(multi_task_count)]
    instances = [
        create_instance(f"bench-{i}#", prompt_tokens.copy(), max_len)
        for i in range(multi_task_count)
    ]
    multi_instances = InferInstances(instances, [
        prompt_tokens.copy()
        for _ in range(multi_task_count)
    ], with_detail_traces=with_detail_trace)
    outs = await multi_instances.run_prefill()
    for seq_i, (_, _, _, trace) in enumerate(outs):
        if with_detail_trace and trace is not None:
            assert trace is not None
            detail_time_traces[seq_i].append(trace)

    time_record_multi: List[Tuple[int, int]] = []
    t_window: List[int] = []

    st = time.time_ns()
    async for out in multi_instances.run_generating():
        seq_i = out[0]
        t = time.time_ns()
        time_record_multi.append((seq_i, t))
        t_window.append(t)
        trace = out[-1]
        if with_detail_trace and trace is not None:
            detail_time_traces[seq_i].append(trace)
        if len(t_window) > 8:
            print(f"\r{8/((t_window[-1]-t_window[0])/(10**9)):.2f}; {len(time_record_multi)/1000:.2f}K", end="")
            t_window = t_window[1:]
    print()
    for instance in instances:
        remove_instance(instance)
    await asyncio.sleep(0.2)

    print(f"{len(time_record_multi)/((time.time_ns()-st)/1e9)}")

    analyzer = Analyzer()
    analyzer.add_record(f"Multi-{multi_task_count}", time_record_multi)
    analyzer.speed_info()
    return time_record_multi, detail_time_traces
    # with open(output_file, "w") as f:
    #     json.dump(time_record_multi, f)


async def run_benchmark(client_node: ClientNode, profile: Any, output_dir: str):
    # print(inspected)
    model_info = models.models_storage[profile.MODEL_NAME]
    tokenizer = ExtendTokenizer.load_tokenier(
        path=model_info.tokenizer_path,
        cfg=model_info.config.tokenizer_special_tokens,
    )
    extra_info = await req_extra_info(client_node.base_node)
    base_models = extra_info.models[profile.MODEL_NAME].models
    keypoints = await profile.GEN_INIT_PATH(base_models, client_node)
    print("keypoints: ", keypoints)

    user_model, build_prompt = gen_user_model(
        user_model_name=profile.USER_MODEL_DATA_NAME,
        base_model_config=model_info.config,
        user_model_config=profile.USER_MODEL_DATA_CFG,
        user_model_data_file=profile.USER_MODEL_DATA_FILE
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
            retrieve=profile.RETRIEVE
        )
        return infer_ctrl

    time_record_multi, detail_time_traces = await run_benchmark_multi(
        tokenizer=tokenizer,
        create_instance=create_instance,
        remove_instance=lambda instance: client_node.remove_infer_control(instance.ctrl_id),
        sample_string=build_prompt(profile.SAMPLE_STR),
        max_len=profile.SEQ_LEN,
        multi_task_count=profile.PARALLEL_SEQS,
        with_detail_trace=profile.DETAIL_TRACE
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(time_record_multi, f)
    if len(detail_time_traces[0]) > 0:
        with open(os.path.join(output_dir, "traces.json"), "w") as f:
            json.dump(detail_time_traces, f)
    print("to quick review result, use the following cmd:")
    print(f"python analysis_infer.py {output_dir}")

    await asyncio.sleep(0.3)
    exit(0)


def benchmark_node(ctx: Context):
    async def coro(ctx: Context):
        ctx.report_ready()
        net_extend = ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        ctrl = ClientNode(base_node)
        asyncio.create_task(run_benchmark(ctrl, ctx.args.profile, ctx.args.output_dir))

        while True:
            packet = await ctx.recv()
            base_node.on_packet(packet)
    asyncio.get_event_loop().create_task(coro(ctx))


def run_satellite(role: str, orbit_addr: str, profile: Any, output_dir: str):
    def init_fn():
        def on_except(_: asyncio.Loop, e: Exception):
            raise e
        asyncio.config_impl(*profile.ASYNC_IMPLEMENT)
        loop = asyncio.get_event_loop()
        loop.on_except(on_except)
        return loop
    launch_pad = LaunchPad(init_fn)

    async def coro():
        satellite = Satellite(launch_pad=launch_pad, functions_dict={}, host_id=role, network_hostname="", stub_remote_log=True)
        satellite.start(orbit_addr)
        await satellite.ready()
        satellite.launch(benchmark_node, profile=profile, output_dir=output_dir)
        await asyncio.Future()

    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="7B", required=True, help="profile file to use (under profiles/inference dir)")
    parser.add_argument("--output", type=str, default="", help="detail results output file")
    parser.add_argument("--setup", type=str, default="", help="setup task to run before benchmark")
    parser.add_argument("--orbit", type=str, default="localhost:7241", help="orbit addr", required=False)
    args, unknown = parser.parse_known_args()
    profile_file = args.profile
    output_dir = args.output
    profile = importlib.import_module(f"scripts.online.benchmark.profiles.inference.{profile_file}")
    if output_dir == "":
        output_dir = os.path.join(os.path.dirname(__file__), "results", "inference", f"{profile_file}")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "profiles", "inference", f"{profile_file}.py"), os.path.join(output_dir, "profile.py"))
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(sys.argv[1:], f)

    setup_task = args.setup
    if setup_task != "":
        print(f"setup task is set, profile async implement will be ignored, extra args: {profile.SETUP_EXTRA} will effect setup")

        script = importlib.import_module(f"scripts.online.benchmark.setups.{setup_task}")
        shutil.copyfile(os.path.join(os.path.dirname(__file__), "setups", setup_task+".py"), os.path.join(output_dir, "setup.py"))
        launch_pad = LaunchPad(script.PROC_INIT)
        unknown += profile.SETUP_EXTRA

        async def amain():
            orbit = await script.SETUP(launch_pad, *unknown)
            print("Setup Done")
            await asyncio.sleep(1)
            await orbit.launch_local("benchmark", benchmark_node, profile=profile, output_dir=output_dir)
            await asyncio.Future()

        launch_pad.loop.create_task(amain())
        launch_pad.loop.run_forever()
    else:
        print(f"setup task is not set, profile async implement: {profile.ASYNC_IMPLEMENT} will effect, extra args will be ignored")
        run_satellite("benchmark", orbit_addr=args.orbit, profile=profile, output_dir=output_dir)
