from typing import List, Literal, Tuple, cast
import torch
from dataset.common import ProcessedRecord
from dataset.advertise_gen import make_dataset_adv_gen
from dealers.common.base_types import balance_divide
from dealers.common.brain import OptimizerConfig, UserModelDataInitConfig
from dealers.memory_schedule.brain import Brain
from dealers.memory_schedule.restriction import MemoryConfigStatic
from node.compute_node import TrainHeadInput
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
from scripts.offline.testbed import ModelConfig, PipelineParallelTestbed
from scripts.offline.testbed.train_instance import TrainTestConfig
import extend_asyncio as asyncio
from utils.minibatch_feeder import AsyncMinibatchFeeder

TEST: Literal["swap", "recover", "normal"] = "normal"


async def amain():
    device = torch.zeros((0,), device="cuda:0").device
    stream = torch.cuda.Stream(device=device)
    container = BaseContainer(stream)

    swap_memory_config = MemoryConfigStatic()
    if TEST == "swap":
        swap_memory_config.root_memory = int(6.3*(2**30))
        swap_memory_config.major_reserved = (2**20)*64
    cfg = ModelConfig()
    brain = Brain(
        models_path=split_qllm_models.lib_path,
        device=device,
        container=container,
        max_seq_len=cfg.max_seq_len,
        max_concurrency=8,
        decoder_implement=cfg.decoder_backend,
        memory_config=swap_memory_config
    )
    testbed = PipelineParallelTestbed([brain], cfg)
    ptuning = True
    if ptuning:
        user_model_config = PtuningConfig(
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.1,
            prefix_len=128
        )
        optimizer_config = OptimizerConfig(
            optimizer_name="AdamW", lr=1e-4, eps=1e-4
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

    scheduler_config = None
    data_init_method_config = UserModelDataInitConfig(
        init_method="reinit"
    )
    test_config = TrainTestConfig(
        backup_period=0,
        recover=(
            [(2, False), (15, True), (30, False)]
            if TEST == "recover" else
            []
        ),
    )
    groups = balance_divide(
        base_models=testbed.base_model_names,
        with_head=True, with_tail=True, decoders_groups=2, user_model_prefix=f"user_model"
    )
    train_instance = await testbed.create_train_instance(
        groups=groups,
        user_model_config=user_model_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        data_init_method=data_init_method_config,
        test_config=test_config,
    )

    minibatch_size = 1
    batch_size = 1

    train_status_maintainer = TrainStatusMaintainer(
        step_size=StepSize(
            steps_in_each_update=batch_size,
            step_batch_size=minibatch_size
        ),
        total_samples=len(train_dataset),
        events=None,
    )

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

    feeder = AsyncMinibatchFeeder(gen_sample, batch_size, batch_size*2)

    async def do_one_batch(sample: Tuple[List[str], int, str, List[ProcessedRecord]]):
        batch_ids, req_version, minibatch_id, records = sample
        _input = TrainHeadInput.from_records(records, is_bfloat16=False)
        loss = await train_instance.do_mini_batch(req_version, full_batch_ids=batch_ids, minibatch_id=minibatch_id, sample=_input)
        total_tokens = sum(len(r.input_ids) for r in records)
        print(train_status_maintainer.update_loss(loss, total_tokens))
        feeder.consume_one()

    while True:
        sample = await feeder.get_next()
        asyncio.create_task(do_one_batch(sample))

if __name__ == "__main__":
    # force swap out/in to check correctness
    # train_binding.FORCE_SWAP_IN_SWAP_OUT=True
    # flags.DEOCDERS_TRAIN_METHOD = "first.2pass"
    asyncio.config_impl("golang", True)
    asyncio.run(amain())
