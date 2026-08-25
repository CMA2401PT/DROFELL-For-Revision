import os
from typing import Dict, List, TypeVar, cast
import torch
from dataset.common import ProcessedRecord
from dataset.advertise_gen import make_dataset_adv_gen
from dealers.memory_schedule.brain import Brain
from node.compute_node.train_meta import TrainHeadInput
from operators.helpers.cuda_utils.memory import current_memory_status
from utils.minibatch_feeder import MinibatchFeeder
from dealers.common.base_types import balance_divide
from dealers.common.brain import OptimizerConfig, UserModelDataInitConfig
from operators.core.lora.common import LoraConfig
import split_qllm_models
from dealers.real_basic.basic import BasicBrain
from operators.container.base_container import BaseContainer
from operators.core.ptuning.common import PtuningConfig
from torch.utils.data import DataLoader
from operators.container import get_thread_context
import data
import cache
from scripts.base.train_state_maintainer import TrainStatusMaintainer, StepSize
from testbed import ModelConfig, TestBed
from scripts.offline.testbed.train_instance import TrainTestConfig
import extend_asyncio as asyncio

SampleData = TypeVar("SampleData")


async def train_main(
        events: Dict[str, int],
        step_size: StepSize,
        test_config: TrainTestConfig,
        bfloat16_end: int,
):
    device = torch.zeros((0,), device="cuda:0").device
    stream = cast(torch.cuda.Stream, torch.cuda.Stream(device))
    container = BaseContainer(stream)
    brain = Brain(
        models_path=split_qllm_models.lib_path,
        device=device,
        container=container
    )

    cfg = ModelConfig()
    testbed = TestBed(brain, cfg)

    ptuning = True
    if ptuning:
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

    train_dataset = make_dataset_adv_gen(
        json_file=data.advertise_gen_train_path,
        build_prompt_str=user_model_config.build_prompt_str,
        tokenizer_path=cfg.model_info.tokenizer_path,
        eos_token_id=cfg.model_info.config.tokenizer_special_tokens.eos_token_id,
        max_query_tokens=-1,
        max_response_tokens=-1,
        cache_dir=cache.dataset_cache_path
    )

    minibatch_size = step_size.step_batch_size
    batch_size = step_size.steps_in_each_update
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
    print(f"batch size: {batch_size}")
    feeder = MinibatchFeeder(gen_sample, batch_size)
    print("data load complete")

    scheduler_config = None
    data_init_method_config = UserModelDataInitConfig(
        init_method="reinit"
    )
    test_config = TrainTestConfig(
        backup_period=2,
        recover=[(10, False), (15, True), (30, False)]
    )
    groups = balance_divide(
        base_models=testbed.base_model_names,
        with_head=True, with_tail=True, decoders_groups=1, user_model_prefix=f"user_model"
    )
    train_instance = await testbed.create_train_instance(
        groups=groups,
        user_model_config=user_model_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        data_init_method=data_init_method_config,
        test_config=test_config,
    )

    print("ready to train ")

    train_status = TrainStatusMaintainer(
        step_size=step_size,
        total_samples=len(train_dataset),
        events=events,
    )

    while True:
        batch_ids, req_version, minibatch_id, records = feeder.get_next()
        is_bfloat16 = train_status.steps < bfloat16_end
        _input = TrainHeadInput.from_records(records, is_bfloat16=is_bfloat16)
        loss = await train_instance.do_mini_batch(req_version, full_batch_ids=batch_ids, minibatch_id=minibatch_id, sample=_input)
        print(train_status.update_loss(loss))
        if train_status.should("memory"):
            m = current_memory_status()
            sm = {}
            for k, v in m.items():
                print(f"{k} {str(v)}")
                sm[k] = (v.allocated, v.approximate, v.requested)
            train_status.record_event("memory", sm)
        if train_status.should("record"):
            train_status.dump(os.path.join(os.path.dirname(__file__), "train_with_recover_correct_record.json"))

backup_period = 10
test_config = TrainTestConfig(
    backup_period=backup_period,
    recover=[
        (i, False)
        for i in [
            backup_period*2+1, backup_period*4-1, backup_period*7,
            backup_period*12+1, backup_period*14-1, backup_period*17,
            backup_period*32+1, backup_period*34-1, backup_period*37,
            backup_period*42+1, backup_period*44-1, backup_period*47
        ]
    ]
)

asyncio.config_impl("golang", True)
asyncio.run(train_main(
    events={
        "record": 10,
        "memory": 20,
    },
    step_size=StepSize(1, 4),
    test_config=test_config,
    bfloat16_end=backup_period*20,
))
