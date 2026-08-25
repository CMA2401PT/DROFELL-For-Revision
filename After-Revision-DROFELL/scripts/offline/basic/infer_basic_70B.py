from typing import Literal

import torch


from dealers.common.base_types import balance_divide
from dealers.memory_schedule.brain import Brain
from dealers.memory_schedule.restriction import MemoryConfigPercentage
from operators.core.lora.common import LoraConfig

import split_qllm_models


from operators.container.base_container import SharedBaseContainer
from operators.core.ptuning.common import PtuningConfig
from operators.core.tokenizer.common import Sequence, SequenceCore, StopConfig
from operators.core.sampler.common import SamplingParams
import extend_asyncio as asyncio
import data

from scripts.offline.testbed import ModelConfig, PipelineParallelTestbed
from scripts.offline.testbed.infer_instance import InferTestConfig

TEST: Literal["swap", "recover", "normal"] = "swap"


async def amain():
    swap_memory_config = MemoryConfigPercentage()
    cfg = ModelConfig(
        model_name="teared-Llama-2-70B-GPTQ",
        group_fn=lambda layers: [40 for _ in range(layers//40)],
    )
    devices = 2
    brains = [
        Brain(
            models_path=split_qllm_models.lib_path,
            device=torch.zeros((0,), device=f"cuda:{cuda_i}").device,
            container=SharedBaseContainer(
                stream=torch.cuda.Stream(torch.zeros(
                    (0,), device=f"cuda:{cuda_i}").device)
            ),
            max_seq_len=cfg.max_seq_len,
            max_concurrency=(1 if TEST == "swap" else 8),
            decoder_implement=cfg.decoder_backend,
            memory_config=swap_memory_config
        )

        for cuda_i in range(devices)
    ]
    testbed = PipelineParallelTestbed(brains, cfg)
    if TEST == "swap":
        # Limit only the modeled pool above the resident base model. Real GPU
        # capacity remains untouched, so swap is exercised before the physical
        # device approaches OOM.
        kv_pool = 512 * (2**20)
        base_memory = max(
            brain.memory_modeling.pools[swap_memory_config.base_models_name]
            for brain in brains
        )
        swap_memory_config.root_memory = (
            base_memory
            + swap_memory_config.operate_reserved
            + swap_memory_config.graph_budget
            + swap_memory_config.unmanaged_reserved
            + kv_pool
        )
        for brain in brains:
            brain.reinit_restriction()
        print(
            f"swap test modeled KV pool: {kv_pool/(2**20):.0f} MiB, "
            f"root: {swap_memory_config.root_memory/(2**30):.2f} GiB"
        )

    ptuning = True
    user_model_name = "advgen-ptuning"
    state_file = ""
    if ptuning:
        user_model_config = PtuningConfig(
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.1,
            prefix_len=128
        )
        state_file = data.trained_model_path_of("ptuning_70B.pt")
    else:
        user_model_config = LoraConfig(
            r_q=-1,
            r_k=0,
            r_v=-1,
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.0,
            len_limit=128,
        )
        state_file = data.trained_model_path_of("lora_70B.pt")
    testbed.put_model_from_config(
        name=user_model_name, cfg=user_model_config, state_file=state_file
    )
    # prompt = "once upon a day, a girl"
    query = "类型#裤*风格#英伦*风格#简约"
    if TEST == "swap":
        query *= 12
    prompt = user_model_config.build_prompt(query)
    prompt_ids = testbed.tokenizer.encode(prompt)

    async def infer_instance(i: int):
        sampling_params = SamplingParams(temperature=0.0, logprobs=1)
        stop_config = StopConfig(max_tokens=256, ignore_eos=False)
        groups = balance_divide(
            base_models=testbed.base_model_names,
            with_head=True, with_tail=True, decoders_groups=devices,
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
                    []
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
