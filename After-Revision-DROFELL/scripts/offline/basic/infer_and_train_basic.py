from typing import List, Tuple

import torch


import cache
from dataset.advertise_gen import make_dataset_adv_gen
from node.compute_node.train_meta import TrainHeadInput
from utils.minibatch_feeder import AsyncMinibatchFeeder
from dataset.common import ProcessedRecord
from dealers.common.base_types import balance_divide
from dealers.common.brain import OptimizerConfig, UserModelDataInitConfig
from dealers.memory_schedule.brain import Brain
from operators.container.run_context import get_thread_context
from operators.core.lora.common import LoraConfig
from scripts.base.train_state_maintainer import TrainStatusMaintainer, StepSize
from scripts.offline.testbed.train_instance import TrainTestConfig
import split_qllm_models
from torch.utils.data import DataLoader
from operators.container.base_container import BaseContainer
from operators.core.ptuning.common import PtuningConfig
from operators.core.tokenizer.common import Sequence, SequenceCore, StopConfig
from operators.core.sampler.common import SamplingParams
import extend_asyncio as asyncio
import data

from testbed import ModelConfig, PipelineParallelTestbed
from testbed.infer_instance import InferTestConfig


async def amain():
    device = torch.zeros((0,), device="cuda:0").device
    stream = torch.cuda.Stream(device=device)
    container = BaseContainer(stream)

    cfg = ModelConfig(
        group_fn=lambda layers: [8 for _ in range(layers//8)],
    )
    brain = Brain(
        models_path=split_qllm_models.lib_path,
        device=device,
        container=container,
        max_seq_len=cfg.max_seq_len,
        max_concurrency=8,
        decoder_implement=cfg.decoder_backend,
        individual_train_thread=True,
    )
    testbed = PipelineParallelTestbed([brain], cfg)

    ptuning = True
    user_model_name = "advgen-ptuning"
    state_file = ""
    if ptuning:
        user_model_config = PtuningConfig(
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.1,
            prefix_len=128
        )
        state_file = data.trained_model_path_of("ptuning_7B.pt")
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
        state_file = data.trained_model_path_of("lora_7B.pt")
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
    test_config = TrainTestConfig()
    train_groups = balance_divide(
        base_models=testbed.base_model_names,
        with_head=True, with_tail=True, decoders_groups=2, user_model_prefix=f"user_model_train"
    )

    train_instance = await testbed.create_train_instance(
        groups=train_groups,
        user_model_config=user_model_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        data_init_method=data_init_method_config,
        test_config=test_config,
    )

    minibatch_size = 2
    batch_size = 2

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

    async def train_coro():
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
            print(train_status_maintainer.update_loss(loss))
            feeder.consume_one()

        while True:
            sample = await feeder.get_next()
            if sample is None:
                break
            asyncio.create_task(do_one_batch(sample))

    async def infer_coro():
        testbed.put_model_from_config(
            name=user_model_name, cfg=user_model_config, state_file=state_file
        )
        prompt = "once upon a day, a girl"
        prompt = user_model_config.build_prompt("类型#裤*风格#英伦*风格#简约")
        prompt_ids = testbed.tokenizer.encode(prompt)

        async def infer_instance(i: int):
            sampling_params = SamplingParams(temperature=0.0, logprobs=1)
            stop_config = StopConfig(max_tokens=256, ignore_eos=False)
            groups = balance_divide(
                base_models=testbed.base_model_names,
                with_head=True, with_tail=True, decoders_groups=1,
                user_model_prefix=f"user_model-{i}"
            )

            instance_orig = await testbed.create_infer_instance(
                groups=groups,
                seq_core=SequenceCore(
                    stop_config=stop_config,
                    prompt_token_ids=prompt_ids.copy(),
                ),
                sampling_params=sampling_params,
                test_config=InferTestConfig(),
                user_model_name=user_model_name,
            )
            seq = Sequence(core=SequenceCore(
                stop_config=stop_config,
                prompt_token_ids=prompt_ids,
            ))
            async for (_, new_token, _) in instance_orig.infer():
                seq.core.append_token_id(new_token)
                testbed.tokenizer.incrementally_decode_seq(seq)
            instance_orig.remove()
            print(instance_orig.seq_id, "> ", seq.output_text)
        tasks = [
            infer_instance(i)
            for i in range(8)
        ]
        await asyncio.gather(*tasks)
    asyncio.create_task(train_coro())
    asyncio.create_task(infer_coro())
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.config_impl("golang", True)
    asyncio.run(amain())
