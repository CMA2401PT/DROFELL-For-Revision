import data
from dealers.memory_schedule.restriction import MemoryConfigStatic
from operators.core.ptuning.common import PtuningConfig
from operators.core.lora.common import LoraConfig
from operators.container.base_container import (SharedBaseContainer)
from dealers.memory_schedule.brain import Brain
from dealers.common.base_types import balance_divide, group_model
from scripts.offline.testbed import ModelConfig, TestBed
from scripts.offline.testbed.infer_instance import InferTestConfig
import split_qllm_models
import torch
from typing import Literal, Optional, Union, cast


print("profile: 7B")
ASYNC_IMPLEMENT = ("golang", True)
DEVICE = "cuda:0"
MODEL_CFG = ModelConfig(
    model_name="teared-Llama-2-7B-GPTQ",
    group_fn=lambda i: [i]
)
SEQ_LEN = 1050
PARALLEL_SEQS = 8


def GROUPS(prefix: str):
    return balance_divide(
        base_models=group_model(
            base_model_name=MODEL_CFG.model_name,
            decoders=MODEL_CFG.layers,
            group_fn=MODEL_CFG.group_fn
        ),
        with_head=True,
        with_tail=True,
        decoders_groups=1,
        user_model_prefix=prefix
    )


USER_MODEL_DATA_TYPE: Literal["ptuning", "lora", ""] = ""
if USER_MODEL_DATA_TYPE == "ptuning":
    USER_MODEL_DATA_NAME = "adv-gen"
    USER_MODEL_DATA_FILE = data.trained_model_path_of("ptuning_7B.pt")
    USER_MODEL_DATA_CFG = PtuningConfig(
        build_prompt_str="[instruction]{}\n[generation]",
        dropout_p=0.1,
        prefix_len=128
    )
    SAMPLE_STR = USER_MODEL_DATA_CFG.build_prompt("类型#裤*风格#英伦*风格#简约")

elif USER_MODEL_DATA_TYPE == "lora":
    USER_MODEL_DATA_NAME = "adv-gen"
    USER_MODEL_DATA_FILE = data.trained_model_path_of("lora_7B.pt")
    USER_MODEL_DATA_CFG = LoraConfig(
        r_q=-1,
        r_k=0,
        r_v=-1,
        build_prompt_str="[instruction]{}\n[generation]",
        dropout_p=0.0,
        len_limit=128,
    )
    SAMPLE_STR = USER_MODEL_DATA_CFG.build_prompt("类型#裤*风格#英伦*风格#简约")
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
    device = torch.zeros((0,), device=DEVICE).device
    stream = cast(torch.cuda.Stream, torch.cuda.Stream(device))
    container = SharedBaseContainer(stream)

    brain = Brain(
        models_path=split_qllm_models.lib_path,
        device=device,
        container=container,
        memory_config=MemoryConfigStatic()
    )

    testbed = TestBed(brain, MODEL_CFG)
    return testbed
