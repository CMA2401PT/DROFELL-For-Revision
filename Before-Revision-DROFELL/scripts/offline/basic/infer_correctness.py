import json
import os
import time
from typing import List, cast

import torch


from dealers.common.base_types import balance_divide
import split_qllm_models

from dealers.real_basic.basic import BasicBrain
from dealers.memory_schedule.brain import Brain
from operators.container.base_container import BaseContainer
from operators.core.ptuning.common import PtuningConfig
from operators.core.tokenizer.common import Sequence, SequenceCore, StopConfig
from operators.core.sampler.common import SamplingParams
import extend_asyncio as asyncio
import data

from testbed import ModelConfig, TestBed
from testbed.infer_instance import InferTestConfig

model_id = "teared-Llama-2-7B-GPTQ"


async def infer_main():
    device = torch.zeros((0,), device="cuda:0").device
    stream = cast(torch.cuda.Stream, torch.cuda.Stream(device))
    container = BaseContainer(stream)

    brain = Brain(
        models_path=split_qllm_models.lib_path,
        device=device,
        container=container
    )

    cfg = ModelConfig(
        group_fn=lambda layers: [8 for _ in range(layers//8)],
    )
    testbed = TestBed(brain, cfg)
    groups = balance_divide(
        base_models=testbed.base_model_names,
        with_head=True, with_tail=True, decoders_groups=1,
        user_model_prefix=f"user_model"
    )

    user_model_name = "advgen-ptuning"
    user_model_config = PtuningConfig(
        build_prompt_str="[instruction]{}\n[generation]",
        dropout_p=0.1,
        prefix_len=128
    )
    state_file = data.trained_model_path_of("ptuning_7B.pt")
    testbed.put_model_from_config(
        name=user_model_name, cfg=user_model_config, state_file=state_file
    )
    prompt = user_model_config.build_prompt("类型#裤*风格#英伦*风格#简约")
    prompt_ids = testbed.tokenizer.encode(prompt)

    client_side_seq = Sequence(core=SequenceCore(
        stop_config=StopConfig(max_tokens=256, ignore_eos=False),
        prompt_token_ids=prompt_ids,
    ))

    # normal inference (warm up)
    seq = client_side_seq.clone()
    server_side_instance = await testbed.create_infer_instance(
        groups=groups,
        seq_core=seq.core.clone(),
        sampling_params=SamplingParams(temperature=0.0, logprobs=1),
        test_config=InferTestConfig(),
        user_model_name=user_model_name,
    )

    async for (_, new_token, _) in server_side_instance.infer():
        seq.core.append_token_id(new_token)
        testbed.tokenizer.incrementally_decode_seq(seq)
    print(server_side_instance.seq_id, "> ", seq.output_text)
    server_side_instance.remove()
    torch.cuda.synchronize()

    # normal inference (common, no recover)
    common_record: List[int] = []
    seq = client_side_seq.clone()
    server_side_instance = await testbed.create_infer_instance(
        groups=groups,
        seq_core=seq.core.clone(),
        sampling_params=SamplingParams(temperature=0.0, logprobs=1),
        test_config=InferTestConfig(
            block_retrieve=True,
        ),
        user_model_name=user_model_name,
    )

    async for (_, new_token, _) in server_side_instance.infer():
        seq.core.append_token_id(new_token)
        testbed.tokenizer.incrementally_decode_seq(seq)
        common_record.append(time.time_ns())
    print(server_side_instance.seq_id, "> ", seq.output_text)
    server_side_instance.remove()
    torch.cuda.synchronize()

    # normal inference (common, no recover)
    fast_record: List[int] = []
    seq = client_side_seq.clone()
    server_side_instance = await testbed.create_infer_instance(
        groups=groups,
        seq_core=seq.core.clone(),
        sampling_params=SamplingParams(temperature=0.0, logprobs=1),
        test_config=InferTestConfig(
            block_retrieve=False,
        ),
        user_model_name=user_model_name,
    )

    async for (_, new_token, _) in server_side_instance.infer():
        seq.core.append_token_id(new_token)
        testbed.tokenizer.incrementally_decode_seq(seq)
        fast_record.append(time.time_ns())
    print(server_side_instance.seq_id, "> ", seq.output_text)
    server_side_instance.remove()
    torch.cuda.synchronize()

    # inference with recover
    recover_record: List[int] = []
    seq = client_side_seq.clone()
    server_side_instance = await testbed.create_infer_instance(
        groups=groups,
        seq_core=seq.core.clone(),
        sampling_params=SamplingParams(temperature=0.0, logprobs=1),
        test_config=InferTestConfig(
            block_retrieve=True,
            recover=[
                (step, False) for step in
                [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
            ]
        ),
        user_model_name=user_model_name,
    )

    async for (_, new_token, _) in server_side_instance.infer():
        seq.core.append_token_id(new_token)
        testbed.tokenizer.incrementally_decode_seq(seq)
        recover_record.append(time.time_ns())
    print(server_side_instance.seq_id, "> ", seq.output_text)
    server_side_instance.remove()
    torch.cuda.synchronize()

    with open(os.path.join(os.path.dirname(__file__), "infer_with_recover_record.json"), "w") as f:
        json.dump({
            "fast": fast_record,
            "common": common_record,
            "recover": recover_record
        }, f)

asyncio.config_impl("golang", True)
asyncio.run(infer_main())
