import json
import os
from dealers.analysis.frame import AnalysisFrame
from dealers.analysis.recorder import TimeRecorder
from dealers.common.brain import OptimizerConfig
from dealers.memory_schedule.restriction import MemoryConfigPercentage
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


print("profile: 70B")
ASYNC_IMPLEMENT = ("golang", True)
NUM_DEVICES = 2
MODEL_CFG = ModelConfig(
    model_name="teared-Llama-2-70B-GPTQ",
    # avoid oom, reduce backward memory requirement
    group_fn=lambda layers: [40 for _ in range(layers//40)],
)
SAMPLES_EACH_INSTANCE = 64
MINIBATCH_SIZE = int(os.getenv("MINIBATCH_SIZE", 4))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))  # 4*4=16 samples in each batch
SAMPLE_LEN = 128
IS_BFLOAT16 = False
PARALLEL_INSTANCES = int(os.getenv("INSTANCES", 4))


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


USER_MODEL_DATA_TYPE: Literal["ptuning", "lora"] = "ptuning"
if USER_MODEL_DATA_TYPE == "ptuning":
    USER_MODEL_DATA_CFG = PtuningConfig(
        build_prompt_str="[instruction]{}\n[generation]",
        dropout_p=0.1,
        prefix_len=32
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
        len_limit=16,
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
    brains = [
        Brain(
            models_path=split_qllm_models.lib_path,
            device=torch.zeros((0,), device=f"cuda:{cuda_i}").device,
            container=SharedBaseContainer(
                stream=cast(torch.cuda.Stream, torch.cuda.Stream(torch.zeros((0,), device=f"cuda:{cuda_i}").device))
            ),
            memory_config=MemoryConfigPercentage(
                operate_reserved=3 * (2 ** 30),
                major_percentage=0.5,
                unmanaged_reserved=500*(2**20),  # 500MiB
                minor_maximum=1024*(2**20),
                minor_minimum=512*(2**20),
            )
        )

        for cuda_i in range(NUM_DEVICES)
    ]
    time_recorders = [
        TimeRecorder(sync_cuda=False)
        for _ in brains
    ]
    brains_with_frame = [
        AnalysisFrame.wrap_brain(
            brain=brain,
            recorder=time_recorder
        )
        for brain, time_recorder in zip(brains, time_recorders)
    ]

    def dump_record(dir: str):
        with open(os.path.join(dir, "brain.json"), "w") as f:
            json.dump({
                f"{i}": time_recorder.dump_info()
                for i, time_recorder in enumerate(time_recorders)
            }, f)
    testbed = TestBed(brains_with_frame, MODEL_CFG)
    return testbed, dump_record
