import json
import time
from typing import Dict, List, Optional, Tuple, Union

import torch

import extend_asyncio as asyncio

from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig
from operators.core.tokenizer.common import SequenceCore
from operators.core.tokenizer.common import Sequence, SequenceCore, StopConfig
from operators.core.sampler.common import SamplingParams
from scripts.offline.testbed.infer_instance import InferInstance
from scripts.offline.testbed import InferTestConfig, TestBed
from scripts.offline.benchmark.inference.analysis_utils import Analyzer


def record_time_to_current_speed(record_time: List[int]):
    if len(record_time) < 31:
        return 0
    dt = record_time[-1]-record_time[-31]
    return 30/(dt/(10**9))


async def make_and_put_user_model(
    testbed: TestBed,
    cfg: Union[LoraConfig, PtuningConfig],
    state_file: str,
    user_model_name: str,
):
    testbed.put_model_from_config(user_model_name, cfg, state_file)
    return


async def create_infer(
    testbed: TestBed,
    groups: List[List[Tuple[str, str]]],
    prompt: Optional[List[int]],
    test_config: InferTestConfig,
    max_len: int,
    user_model_name: Optional[str],
):
    return await testbed.create_infer_instance(
        groups=groups,
        seq_core=SequenceCore(
            stop_config=StopConfig(
                max_tokens=max_len,
                ignore_eos=prompt is None,
            ),
            prompt_token_ids=[2] if prompt is None else prompt.copy(),
        ),
        sampling_params=SamplingParams(
            temperature=0.0, logprobs=1,
        ),
        test_config=test_config,
        user_model_name=user_model_name,
    )


class InferInstances:
    def __init__(self, instances: List[InferInstance]) -> None:
        self.instances = instances

    async def run_prefill(self):
        outs = await asyncio.gather(*[
            instance.prefill()
            for instance in self.instances
        ])
        return outs

    async def run_generating(self):
        queue: asyncio.Queue[Tuple[int, bool, int, Dict[int, float]]] = asyncio.Queue()

        async def run_each(seq_i: int, instance: InferInstance):
            stopped = False
            while not stopped:
                out = await instance.generating_step()
                queue.put_nowait((seq_i, *out))
                stopped = out[0]
        running_tasks = [
            asyncio.create_task(run_each(i, instance))
            for i, instance in enumerate(self.instances)
        ]
        unfinished_count = len(self.instances)
        while unfinished_count > 0:
            out = await queue.get()
            if out[1]:
                unfinished_count -= 1
            yield out
        del running_tasks

    def remove(self):
        for i in self.instances:
            i.remove()


async def create_multi_infer(
    testbed: TestBed,
    groups: List[List[Tuple[str, str]]],
    prompt: Optional[List[int]],
    test_config: InferTestConfig,
    parallel_num: int,
    max_len: int,
    user_model_name: Optional[str]
):
    instances: List[InferInstance] = [
        await create_infer(
            testbed,
            groups,
            prompt.copy() if prompt is not None else None,
            test_config, max_len, user_model_name)
        for _ in range(parallel_num)
    ]
    return InferInstances(instances)


async def run_benchmark_single(
    testbed: TestBed,
    groups: List[List[Tuple[str, str]]],
    output_file: str,
    test_config: InferTestConfig = InferTestConfig(block_retrieve=False, recover=(0, False)),
    max_len: int = 20*1000,
    user_model_name: Optional[str] = None,
    sample_string: str = "once upon a time",
):

    # do warm up
    time_record: List[int] = []
    prompt_tokens = testbed.tokenizer.encode(sample_string)
    instance = await create_infer(testbed, groups, prompt_tokens, test_config, 32, user_model_name=user_model_name)
    seq = Sequence(SequenceCore(StopConfig(max_tokens=max_len), prompt_token_ids=prompt_tokens.copy()))
    async for (_, token, _) in instance.infer():
        time_record.append(time.time_ns())
        seq.core.append_token_id(token)
        testbed.tokenizer.incrementally_decode_seq(seq)
        print(f"{record_time_to_current_speed(time_record):.2f}; {seq.output_text}", end="\r")
    print()
    instance.remove()
    # warm up done, do actual metric
    torch.cuda.synchronize()
    time_record: List[int] = []
    print(max_len)
    instance = await create_infer(testbed, groups, None, test_config, max_len, user_model_name=user_model_name)
    async for (_, token, _) in instance.infer():
        time_record.append(time.time_ns())
        if len(time_record) % 8 == 0:
            print(f"\r{record_time_to_current_speed(time_record):.2f}; {len(time_record)/1000:.2f}K", end="")
    print()
    instance.remove()
    torch.cuda.synchronize()
    analyzer = Analyzer()
    analyzer.add_record("Single", time_record)
    analyzer.speed_info()
    with open(output_file, "w") as f:
        json.dump(time_record, f)


async def run_benchmark_multi(
    testbed: TestBed,
    groups: List[List[Tuple[str, str]]],
    output_file: str,
    test_config: InferTestConfig = InferTestConfig(block_retrieve=False, recover=(0, False)),
    max_len: int = 20*1000, multi_task_count: int = 8,
    user_model_name: Optional[str] = None,
    sample_string: str = "once upon a time",
):

    # do warm up
    print("warm up")
    prompt_tokens = testbed.tokenizer.encode(sample_string)
    multi_instances = await create_multi_infer(testbed, groups, prompt_tokens.copy(), test_config, multi_task_count, 32, user_model_name)

    seqs = [
        Sequence(SequenceCore(StopConfig(max_tokens=max_len), prompt_token_ids=prompt_tokens.copy()))
        for _ in range(multi_task_count)
    ]

    outs = await multi_instances.run_prefill()
    for seq_i, (_, token_id, _) in enumerate(outs):
        seqs[seq_i].core.append_token_id(token_id)
    for seq in seqs:
        testbed.tokenizer.incrementally_decode_seq(seq)

    time_record: List[int] = []
    async for out in multi_instances.run_generating():
        seq_i, _, token_id, _ = out
        time_record.append(time.time_ns())
        seqs[seq_i].core.append_token_id(token_id)
        testbed.tokenizer.incrementally_decode_seq(seqs[seq_i])
        for seq in seqs:
            print(seq.output_text)
        print(f"TPS quick glance: total {record_time_to_current_speed(time_record):.2f}", end=f"\033[{multi_task_count}F")
    print("\n"*multi_task_count)

    multi_instances.remove()
    del multi_instances
    torch.cuda.synchronize()

    print("benchmarking")

    multi_instances = await create_multi_infer(testbed, groups, None, test_config, multi_task_count, max_len, user_model_name)
    await multi_instances.run_prefill()
    torch.cuda.synchronize()
    time_record_multi: List[Tuple[int, int]] = []
    t_window: List[int] = []

    st = time.time_ns()
    async for out in multi_instances.run_generating():
        seq_i = out[0]
        t = time.time_ns()
        time_record_multi.append((seq_i, t))
        t_window.append(t)
        if len(t_window) > 8:
            print(f"\r{8/((t_window[-1]-t_window[0])/(10**9)):.2f}; {len(time_record_multi)/1000:.2f}K", end="")
            t_window = t_window[1:]
    print()
    multi_instances.remove()

    print(f"{len(time_record_multi)/((time.time_ns()-st)/1e9)}")

    analyzer = Analyzer()
    analyzer.add_record(f"Multi-{multi_task_count}", time_record_multi)
    analyzer.speed_info()
    with open(output_file, "w") as f:
        json.dump(time_record_multi, f)

     # warm up done, do actual metric
    #  time_record_multi: List[Tuple[int, int]] = []
    #   t_window: List[int] = []
    #    async for (seq_i, _) in create_multi_infer(base_model, None, cfg.test_config, parallel_num=multi_task_count, max_len=cfg.max_len):
    #         t = time.time_ns()
    #         time_record_multi.append((seq_i, t))
    #         t_window.append(t)
    #         if len(t_window) > 8:
    #             print(f"\r{8/((t_window[-1]-t_window[0])/(10**9)):.2f}; {len(time_record_multi)/1000:.2f}K", end="")
    #             t_window = t_window[1:]
    #     return time_record_multi


# run_speed_test_single(SpeedTestConfig(), "default2.json")
# run_speed_test_multi(SpeedTestConfig(), "default3.json", multi_task_count=8)
