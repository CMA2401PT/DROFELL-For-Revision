import os
import extend_asyncio as asyncio
from typing import List
from typing import Literal
import data
from node.aux_node.protocols import req_providers
from node.client.client import ClientNode
from operators.core.ptuning.common import PtuningConfig
from operators.core.lora.common import LoraConfig
from node import flags
from orbit import *

print("profile: 7B")
ASYNC_IMPLEMENT = ("golang", True)
SETUP_EXTRA = []
MODEL_NAME = model_name = "teared-Llama-2-7B-GPTQ"

_reseted = False
DROP_PROB: float = float(os.getenv("DROP_PROB", default=0.01))
MULTI_PATH = int(os.getenv("MULTI_PATH", default=1))
POSTPONE = float(os.getenv("POSTPONE", default=0.3))


async def InspectPostpone(kind: Literal[
        "infer",
        "train",
        "backbone.infer",
        "backbone.train"
]):
    await asyncio.sleep(POSTPONE)

flags.InspectPostpone = InspectPostpone


async def _setup(client_node: ClientNode):
    dealer_nodes = set(p for _, ps in await req_providers(client_node.base_node, None) for p, _ in ps)
    for dealer_node in dealer_nodes:
        tags = await client_node.base_node.req_tags(dealer_node)
        print(f"{dealer_node} {tags}")
        if 'infer_drop_reset' in tags:
            await client_node.base_node.rpc_with_args(
                node=dealer_node, api="infer_drop_reset", args=DROP_PROB, T=float)


async def GEN_INIT_PATH(
    base_models: List[str],
    client_node: ClientNode
):
    global _reseted
    if not _reseted:
        _reseted = True
        await _setup(client_node)
    inspected = await client_node.inspect_fastest_n_path(
        type="infer",
        base_models=base_models,
        allow_reuse=False,
        max_n=MULTI_PATH,
        providers=None
    )
    assert inspected
    keypoints = inspected.keypoints
    return keypoints

SEQ_LEN = 1024
PARALLEL_SEQS = 1
DETAIL_TRACE = True
RETRIEVE = False

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
