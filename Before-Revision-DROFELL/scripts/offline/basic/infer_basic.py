from typing import Literal, cast

import torch


from dealers.common.base_types import balance_divide
from dealers.memory_schedule.brain import Brain
from dealers.memory_schedule.restriction import MemoryConfigStatic
from operators.core.lora.common import LoraConfig
import split_qllm_models

from dealers.real_basic.basic import BasicBrain

from operators.container.base_container import BaseContainer
from operators.core.ptuning.common import PtuningConfig
from operators.core.tokenizer.common import Sequence, SequenceCore, StopConfig
from operators.core.sampler.common import SamplingParams
import extend_asyncio as asyncio
import data

from testbed import ModelConfig, TestBed
from testbed.infer_instance import InferTestConfig

TEST: Literal["swap", "recover", "normal"] = "normal"


async def amain():
    device = torch.zeros((0,), device="cuda:0").device
    stream = cast(torch.cuda.Stream, torch.cuda.Stream(device))
    container = BaseContainer(stream)

    swap_memory_config = MemoryConfigStatic()
    if TEST == "swap":
        swap_memory_config.root_memory = int(6.5*(2**30)) - int(198*(2**20))
        swap_memory_config.major_reserved = (2**20)*64
    brain = Brain(
        models_path=split_qllm_models.lib_path,
        device=device,
        container=container,
        memory_config=swap_memory_config
    )

    cfg = ModelConfig(
        model_name="teared-Llama-2-7B-GPTQ",
        group_fn=lambda layers: [8 for _ in range(layers//8)],
    )
    testbed = TestBed(brain, cfg)

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
            test_config=InferTestConfig(
                block_retrieve=True,
                recover=(
                    [(20, False), (30, False), (80, True), (100, False)]
                    if TEST == "recover" else
                    (0, False)
                ),
            ),
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

if __name__ == "__main__":
    asyncio.config_impl("golang", True)
    asyncio.run(amain())
