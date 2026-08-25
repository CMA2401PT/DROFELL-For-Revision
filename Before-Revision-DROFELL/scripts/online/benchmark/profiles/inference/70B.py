import os
from typing import List
from typing import Literal
import data
from node.client.client import ClientNode
from node.path_utils import paths_to_keypoints
from operators.core.ptuning.common import PtuningConfig
from orbit import *

print("profile: 70B")
ASYNC_IMPLEMENT = ("golang", True)
SETUP_EXTRA = []
MODEL_NAME = model_name = "teared-Llama-2-70B-GPTQ"


async def GEN_INIT_PATH(
    base_models: List[str],
    client_node: ClientNode
):
    inspected = await client_node.inspect_fastest_path("infer", base_models, None)
    assert inspected
    keypoints = paths_to_keypoints(paths=[inspected.best])
    return keypoints

SEQ_LEN = 2050
PARALLEL_SEQS = int(os.getenv("SEQS", 1))
DETAIL_TRACE = True
RETRIEVE = False

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
