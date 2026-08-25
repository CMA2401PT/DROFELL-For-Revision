import json
import os
from dealers.analysis.recorder import TimeRecorder
from dealers.analysis.frame import AnalysisFrame
from dealers.common.brain import OptimizerConfig
from dealers.memory_schedule.restriction import MemoryConfigStatic
from operators.container.base_container import SharedBaseContainer
from operators.core.ptuning.common import PtuningConfig
from operators.core.lora.common import LoraConfig
from dealers.memory_schedule.brain import Brain
from dealers.common.brain import OptimizerConfig, UserModelDataInitConfig
from dealers.common.base_types import balance_divide, group_model
from scripts.offline.testbed import ModelConfig, TestBed
from scripts.offline.testbed.train_instance import TrainTestConfig
import split_qllm_models
import torch
from typing import Literal, cast


print("profile: 7B")
ASYNC_IMPLEMENT = ("golang", True)
DEVICE = "cuda:0"
MODEL_CFG = ModelConfig(
    model_name="teared-Llama-2-7B-GPTQ",
    group_fn=lambda i: [i]
)
SAMPLES_EACH_INSTANCE = 100
MINIBATCH_SIZE = 2
BATCH_SIZE = 2  # 8*8=64 samples in each batch
SAMPLE_LEN = 128
IS_BFLOAT16 = False
PARALLEL_INSTANCES = 2


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


USER_MODEL_DATA_TYPE: Literal["ptuning", "lora"] = "ptuning"
if USER_MODEL_DATA_TYPE == "ptuning":
    USER_MODEL_DATA_CFG = PtuningConfig(
        build_prompt_str="[instruction]{}\n[generation]",
        dropout_p=0.1,
        prefix_len=128
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
        len_limit=128,
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
TEST_CFG = TrainTestConfig(
    backup_period=0,
    recover=(0, False),
)


async def GEN_TESTBED():
    device = torch.zeros((0,), device=DEVICE).device
    stream = cast(torch.cuda.Stream, torch.cuda.Stream(device))
    container = SharedBaseContainer(stream)
    time_recorder = TimeRecorder(sync_cuda=False)

    brain = AnalysisFrame.wrap_brain(
        brain=Brain(
            models_path=split_qllm_models.lib_path,
            device=device,
            container=container,
            memory_config=MemoryConfigStatic()
        ),
        recorder=time_recorder
    )

    def dump_record(dir: str):
        with open(os.path.join(dir, "brain.json"), "w") as f:
            json.dump({
                "1": time_recorder.dump_info()
            }, f)
    testbed = TestBed(brain, MODEL_CFG)
    return testbed, dump_record
