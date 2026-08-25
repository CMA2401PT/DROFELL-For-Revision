import os
import data
from node.path_utils import paths_to_keypoints
from node.client.client import ClientNode
from operators.core.ptuning.common import PtuningConfig
from operators.core.lora.common import LoraConfig
from dealers.common.brain import OptimizerConfig
from operators.core.ptuning.common import PtuningConfig
from operators.core.lora.common import LoraConfig
from dealers.common.brain import OptimizerConfig, UserModelDataInitConfig
from typing import List, Literal, cast


print("profile: 7B")
ASYNC_IMPLEMENT = ("golang", True)
SETUP_EXTRA = ["--trace-time"]
ORBIT_ADDR = "localhost:7241"
MODEL_NAME = "teared-Llama-2-7B-GPTQ"
SAMPLES_EACH_INSTANCE = 256
PARALLEL_INSTANCES = int(os.getenv("INSTANCES", 1))
MINIBATCH_SIZE = int(os.getenv("MINIBATCH_SIZE", 4))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))  # 4*4=16 samples in each batch
SAMPLE_LEN = 128
IS_BFLOAT16 = False
DETAIL_TRACE = True
CKPT_PERIOD = 16
PROMPT_LENTTH = int(os.getenv("TRAINABLE", 32))


async def GEN_INIT_PATH(
    base_models: List[str],
    client_node: ClientNode
):
    inspected = await client_node.inspect_fastest_path("train", base_models, None)
    assert inspected
    keypoints = paths_to_keypoints(paths=[inspected.best])
    return keypoints


USER_MODEL_DATA_TYPE: Literal["ptuning", "lora", ""] = cast(Literal["ptuning", "lora", ""], os.getenv("METHOD", "ptuning"))
if USER_MODEL_DATA_TYPE == "ptuning":
    USER_MODEL_DATA_CFG = PtuningConfig(
        build_prompt_str="[instruction]{}\n[generation]",
        dropout_p=0.1,
        prefix_len=PROMPT_LENTTH
    )

    def USER_MODEL_INIT_DATA_PROVIDER(src: str, dst: str):
        return UserModelDataInitConfig(
            init_method="reinit",
            ckpt_bytes=None
        )
    OPTIMIZER_CFG = OptimizerConfig(
        optimizer_name="AdamW", lr=1e-2, eps=1e-4
    )
elif USER_MODEL_DATA_TYPE == "lora":
    USER_MODEL_DATA_FILE = data.trained_model_path_of("lora_7B.pt")
    USER_MODEL_DATA_CFG = LoraConfig(
        r_q=-1,
        r_k=0,
        r_v=-1,
        build_prompt_str="[instruction]{}\n[generation]",
        dropout_p=0.0,
        len_limit=PROMPT_LENTTH,
    )

    def USER_MODEL_INIT_DATA_PROVIDER(src: str, dst: str):
        return UserModelDataInitConfig(
            init_method="reinit",
            ckpt_bytes=None
        )
    OPTIMIZER_CFG = OptimizerConfig(
        optimizer_name="AdamW", lr=1e-3, eps=1e-4
    )
else:
    raise ValueError(USER_MODEL_DATA_TYPE)
SCHEDULER_CFG = None
DATA_INIT_CFG = UserModelDataInitConfig(
    init_method="reinit",
    ckpt_bytes=None
)
