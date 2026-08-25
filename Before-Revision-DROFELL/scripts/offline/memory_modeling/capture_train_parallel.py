from asyncio import Task
import os
import pickle
from typing import Callable, List, Tuple, Union, cast

import torch
from torch.utils.data import DataLoader

import cache
import data
import extend_asyncio as asyncio
from operators.helpers.model_lib.common import ModelInfo
import split_qllm_models
from dataset.advertise_gen import make_dataset_adv_gen
from dataset.common import ProcessedRecord
from dealers.analysis.frame import AnalysisFrame
from dealers.analysis.recorder import ActualIORecorder
from dealers.common.base_types import balance_divide
from dealers.common.brain import OptimizerConfig, UserModelDataInitConfig
from dealers.memory_schedule.brain import Brain
from dealers.memory_schedule.restriction import MemoryConfigStatic
from node.compute_node.train_meta import TrainHeadInput
from operators.container import get_thread_context
from operators.container.base_container import BaseContainer
from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig
from scripts.offline.testbed import ModelConfig, TestBed
from scripts.offline.testbed.train_instance import TrainTestConfig
from utils.minibatch_feeder import AsyncMinibatchFeeder

# flags.DECODERS_TRAIN_METHOD="first.2pass"
# flags.USER_MODEL_ENABLE_CHECKPOTIN=False


async def instance(
    instance_name: str,
    gen_sample: Callable[[], List[ProcessedRecord]],
    batch_size: int,
    brain: AnalysisFrame,
    model_config: ModelConfig,
    user_model_config: Union[PtuningConfig, LoraConfig],
    optimizer_config: OptimizerConfig,
):
    feeder = AsyncMinibatchFeeder(
        get_sample=gen_sample,
        batch_size=batch_size,
        max_concurrent=batch_size*2
    )
    testbed = TestBed(brain, model_config)  # type:ignore
    groups = balance_divide(
        base_models=testbed.base_model_names,
        with_head=True, with_tail=True, decoders_groups=4, user_model_prefix=instance_name
    )
    train_instance = await testbed.create_train_instance(
        groups=groups,
        user_model_config=user_model_config,
        optimizer_config=optimizer_config,
        scheduler_config=None,
        data_init_method=UserModelDataInitConfig(init_method="reinit"),
        test_config=TrainTestConfig(backup_period=0, recover=(0, False)),
    )

    async def do_one_batch(sample: torch.Tuple[List[str], int, str, List[ProcessedRecord]]):
        batch_ids, req_version, minibatch_id, records = sample
        is_bfloat16 = False
        _input = TrainHeadInput.from_records(records, is_bfloat16=is_bfloat16)
        await train_instance.do_mini_batch(req_version, full_batch_ids=batch_ids, minibatch_id=minibatch_id, sample=_input)
        _input = TrainHeadInput.from_records(records, is_bfloat16=False)

    running_tasks: Tuple[Task[None], ...]
    _running_tasks: List[Task[None]] = []
    for _ in range(2):
        sample = await feeder.get_next(wait=False)
        assert sample is not None
        t = asyncio.create_task(do_one_batch(sample))
        _running_tasks.append(t)

    running_tasks = tuple(_running_tasks)

    for i in range(20):
        sample = await feeder.get_next(wait=False)
        if sample is None:
            break
        task: Task[None] = asyncio.create_task(do_one_batch(sample))
        running_tasks = (*running_tasks, task)
        done, pending = await asyncio.wait_first(running_tasks)
        running_tasks = tuple(pending)
        print(f"{instance_name}: {i}")
    while running_tasks:
        done, pending = await asyncio.wait_first(running_tasks)
        running_tasks = tuple(pending)
    train_instance.remove()
    await asyncio.sleep(0.1)


def gen_config(is_ptuning: bool):
    if is_ptuning:
        user_model_config = PtuningConfig(
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.1,
            prefix_len=128
        )
        optimizer_config = OptimizerConfig(
            optimizer_name="AdamW", lr=1e-2, eps=1e-4
        )
    else:
        user_model_config = LoraConfig(
            r_q=-1,
            r_k=0,
            r_v=-1,
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.0,
            len_limit=128,
        )
        optimizer_config = OptimizerConfig(
            optimizer_name="AdamW", lr=1e-3, eps=1e-4
        )
    return user_model_config, optimizer_config


def gen_sample_iter(model_info: ModelInfo, build_prompt_str: str):
    train_dataset = make_dataset_adv_gen(
        json_file=data.advertise_gen_train_path,
        build_prompt_str=build_prompt_str,
        tokenizer_path=model_info.tokenizer_path,
        eos_token_id=model_info.config.tokenizer_special_tokens.eos_token_id,
        max_query_tokens=-1,
        max_response_tokens=-1,
        cache_dir=cache.dataset_cache_path
    )
    minibatch_size = 4
    print(f"minibatch_size size: {minibatch_size}")

    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=lambda samples: samples,
        batch_size=minibatch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True
    )
    get_thread_context().switch_cpu()
    train_iter = train_loader.__iter__()

    def gen_sample():
        get_thread_context().switch_cpu()
        records: List[ProcessedRecord] = train_iter.__next__()
        # sample = Sample.create_from_records(samples=records, is_bfloat16=False)
        return records
    return gen_sample


async def amain():
    device = torch.zeros((0,), device="cuda:0").device
    stream = cast(torch.cuda.Stream, torch.cuda.Stream(device))
    container = BaseContainer(stream)

    brain = Brain(
        models_path=split_qllm_models.lib_path,
        device=device,
        container=container,
        individual_train_thread=False,
        memory_config=MemoryConfigStatic(
            preallocate=False,
        )
    )
    recorder = ActualIORecorder(brain)
    brain = AnalysisFrame(brain, recorder)
    model_cfg = ModelConfig()
    scheduler_config = None
    data_init_method_config = UserModelDataInitConfig(
        init_method="reinit"
    )
    test_config = TrainTestConfig(
        backup_period=0,
        recover=(0, False)
    )

    user_model_config, optimizer_config = gen_config(True)
    gen_sample = gen_sample_iter(model_cfg.model_info, user_model_config.build_prompt_str)
    print("data load complete")

    async def wrapper():
        results = await asyncio.gather(*[
            instance(instance_name=f"instance-{instance_i}",  gen_sample=gen_sample, batch_size=4, brain=brain, model_config=model_cfg, user_model_config=user_model_config, optimizer_config=optimizer_config)
            for instance_i in range(1)
        ])
        await asyncio.sleep(0.1)

    await wrapper()
    print("dump state")
    with open(os.path.join(os.path.dirname(__file__), "memory_train_parallel.pkl"), "wb") as f:
        pickle.dump(recorder.recorded, f)

if __name__ == "__main__":
    # force swap out/in to check correctness
    # train_binding.FORCE_SWAP_IN_SWAP_OUT=True
    # flags.DEOCDERS_TRAIN_METHOD = "first.2pass"
    asyncio.config_impl("golang", True)
    asyncio.run(amain())
