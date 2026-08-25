import os
from typing import List
from typing import Literal
import data
from node.client.client import ClientNode
from operators.core.ptuning.common import PtuningConfig
from operators.core.lora.common import LoraConfig
from orbit import *

print("profile: 7B")
ASYNC_IMPLEMENT = ("golang", True)
SETUP_EXTRA = []
MODEL_NAME = model_name = "teared-Llama-2-7B-GPTQ"


async def GEN_INIT_PATH(
    base_models: List[str],
    client_node: ClientNode
):
    inspected = await client_node.inspect_fastest_n_path(
        type="infer",
        base_models=base_models,
        allow_reuse=False,
        max_n=1,
        providers=None
    )
    assert inspected
    keypoints = inspected.keypoints
    return keypoints


SEQ_LEN = int(os.getenv("SEQLEN", 2100))
PARALLEL_SEQS = int(os.getenv("SEQS", 16))
DETAIL_TRACE = bool(os.getenv("DETAIL", False))
RETRIEVE = bool(os.getenv("RETRIEVE", False))

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
