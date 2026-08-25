import os
import data
from dealers.memory_schedule.restriction import MemoryConfigPercentage
from operators.core.ptuning.common import PtuningConfig
from operators.container.base_container import (SharedBaseContainer)
from dealers.memory_schedule.brain import Brain
from dealers.common.base_types import balance_divide, group_model
from scripts.offline.testbed import ModelConfig, TestBed
from scripts.offline.testbed.infer_instance import InferTestConfig
import split_qllm_models
import torch
from typing import Literal, cast


print("profile: 70B")
ASYNC_IMPLEMENT = ("golang", True)
NUM_DEVICES = 2
MODEL_CFG = ModelConfig(
    model_name="teared-Llama-2-70B-GPTQ",
    group_fn=lambda layers: [40 for _ in range(layers//40)],
)
SEQ_LEN = 2100
PARALLEL_SEQS = int(os.getenv("SEQS", 1))


def GROUPS(prefix: str):
    return balance_divide(
        base_models=group_model(
            base_model_name=MODEL_CFG.model_name,
            decoders=MODEL_CFG.layers,
            group_fn=MODEL_CFG.group_fn
        ),
        with_head=True,
        with_tail=True,
        decoders_groups=NUM_DEVICES,
        user_model_prefix=prefix
    )


USER_MODEL_DATA_TYPE: Literal["ptuning", "lora", ""] = ""
if USER_MODEL_DATA_TYPE == "ptuning":
    USER_MODEL_DATA_NAME = "adv-gen"
    USER_MODEL_DATA_FILE = data.trained_model_path_of("ptuning_70B.pt")
    USER_MODEL_DATA_CFG = PtuningConfig(
        build_prompt_str="[instruction]{}\n[generation]",
        dropout_p=0.1,
        prefix_len=128
    )
    SAMPLE_STR = USER_MODEL_DATA_CFG.build_prompt("类型#裤*风格#英伦*风格#简约")

elif USER_MODEL_DATA_TYPE == "lora":
    raise NotImplementedError
    # USER_MODEL_DATA_NAME = "adv-gen"
    # USER_MODEL_DATA_FILE = data.trained_model_path_of("lora_7B.pt")
    # USER_MODEL_DATA_CFG = LoraConfig(
    #     r_q=-1,
    #     r_k=0,
    #     r_v=-1,
    #     build_prompt_str="[instruction]{}\n[generation]",
    #     dropout_p=0.0,
    #     len_limit=128,
    # ).check_or_set_decoders_config(
    #     num_layers=cfg.layers, head_dim=cfg.base_model_config.head_dim,
    #     num_key_value_heads=cfg.base_model_config.num_key_value_heads, num_query_heads=cfg.base_model_config.num_query_heads
    # )
    # SAMPLE_STR = USER_MODEL_DATA_CFG.build_prompt("类型#裤*风格#英伦*风格#简约")
else:
    USER_MODEL_DATA_NAME = ""
    SAMPLE_STR = "once upon a time"
    USER_MODEL_DATA_CFG = None
    USER_MODEL_DATA_FILE = ""

TEST_CFG = InferTestConfig(
    block_retrieve=False,
    recover=(0, False),
)


async def GEN_TESTBED():
    testbed = TestBed(brains=[
        Brain(
            models_path=split_qllm_models.lib_path,
            device=torch.zeros((0,), device=f"cuda:{cuda_i}").device,
            container=SharedBaseContainer(
                stream=cast(torch.cuda.Stream, torch.cuda.Stream(torch.zeros((0,), device=f"cuda:{cuda_i}").device))
            ),
            memory_config=MemoryConfigPercentage(
                operate_reserved=100 * (2 ** 20),
                unmanaged_reserved=10*(2**20),  # 10MiB
                major_percentage=0.5,
                minor_maximum=256*(2**30)
            )
        )
        for cuda_i in range(NUM_DEVICES)
    ], cfg=MODEL_CFG)
    return testbed
