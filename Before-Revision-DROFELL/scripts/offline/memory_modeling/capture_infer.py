import os
import pickle
from typing import cast

import torch

from dealers.analysis.frame import AnalysisFrame
from dealers.analysis.recorder import ActualIORecorder
from dealers.common.base_types import balance_divide
from dealers.memory_schedule.brain import Brain
from dealers.memory_schedule.restriction import MemoryConfig, MemoryConfigStatic
from dealers.real_basic.memory import MemoryConfigBasic
from operators.core.lora.common import LoraConfig
import split_qllm_models

from dealers.real_basic.basic import BasicBrain

from operators.container.base_container import BaseContainer
from operators.core.ptuning.common import PtuningConfig
from operators.core.tokenizer.common import Sequence, SequenceCore, StopConfig
from operators.core.sampler.common import SamplingParams
import extend_asyncio as asyncio
import data

from scripts.offline.testbed import ModelConfig, TestBed
from scripts.offline.testbed.infer_instance import InferTestConfig

# flags.ALLOW_2WAY_DISPATCH = False


async def amain():
    device = torch.zeros((0,), device="cuda:0").device
    stream = cast(torch.cuda.Stream, torch.cuda.Stream(device))
    container = BaseContainer(stream)

    brain = Brain(
        models_path=split_qllm_models.lib_path,
        device=device,
        container=container,
        memory_config=MemoryConfigStatic(),
        max_concurrency=1
    )
    recorder = ActualIORecorder(brain)
    brain = AnalysisFrame(brain, recorder)
    cfg = ModelConfig()
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

    async def wrapper():
        testbed = TestBed(brain, cfg)  # type:ignore
        testbed.put_model_from_config(
            name=user_model_name, cfg=user_model_config, state_file=state_file
        )
        prompt = "once upon a day, a girl"
        prompt = user_model_config.build_prompt("类型#裤*风格#英伦*风格#简约")
        prompt_ids = testbed.tokenizer.encode(prompt)

        sampling_params = SamplingParams(temperature=0.0, logprobs=1)
        stop_config = StopConfig(max_tokens=256, ignore_eos=False)
        groups = balance_divide(
            base_models=testbed.base_model_names,
            with_head=True, with_tail=True, decoders_groups=1,
            user_model_prefix=f"user_model"
        )

        instance = await testbed.create_infer_instance(
            groups=groups,
            seq_core=SequenceCore(
                stop_config=stop_config,
                prompt_token_ids=prompt_ids.copy(),
            ),
            sampling_params=sampling_params,
            test_config=InferTestConfig(
                block_retrieve=False,
                recover=(0, False),
            ),
            user_model_name=user_model_name,
        )
        seq = Sequence(core=SequenceCore(
            stop_config=stop_config,
            prompt_token_ids=prompt_ids,
        ))
        async for (_, new_token, _) in instance.infer():
            seq.core.append_token_id(new_token)
            testbed.tokenizer.incrementally_decode_seq(seq)
        print(instance.seq_id, "> ", seq.output_text)
        torch.cuda.synchronize()
        instance.remove()
    await wrapper()
    print("dump state")
    with open(os.path.join(os.path.dirname(__file__), "memory_infer.pkl"), "wb") as f:
        pickle.dump(recorder.recorded, f)

if __name__ == "__main__":
    asyncio.config_impl("golang", True)
    asyncio.run(amain())
