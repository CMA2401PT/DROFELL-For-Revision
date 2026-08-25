import os
from typing import Literal, cast

import torch
import data
from dealers.common.base_types import balance_divide
from dealers.memory_schedule.brain import Brain
from operators.container.base_container import BaseContainer, SharedBaseContainer
from dealers.real_basic.basic import BasicBrain
from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig
from scripts.offline.speed_test.speed_tester import run_speed_test_multi, make_and_put_user_model
from scripts.offline.testbed import ModelConfig, TestBed
from scripts.offline.testbed.infer_instance import InferTestConfig
import split_qllm_models
import extend_asyncio as asyncio


async def amain():
    device = torch.zeros((0,), device="cuda:0").device
    stream = cast(torch.cuda.Stream, torch.cuda.Stream(device))
    container = SharedBaseContainer(stream)

    brain = Brain(
        models_path=split_qllm_models.lib_path,
        device=device,
        container=container
    )

    cfg = ModelConfig(
        # group_fn=lambda i: [i]
    )
    testbed = TestBed(brain, cfg)
    # await run_speed_test_single(testbed, "test.json", max_len=1024)
    groups = balance_divide(
        base_models=testbed.base_model_names,
        with_head=True,
        with_tail=True,
        decoders_groups=1,
        user_model_prefix="test-model"
    )

    user_model: Literal["ptuning", "lora", ""] = ""
    user_model_name = None
    sample_string = "once upon a time"
    if user_model == "ptuning":
        user_model_name = "adv-gen"
        user_model_config = PtuningConfig(
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.1,
            prefix_len=128
        ).check_or_set_decoders_config(
            num_layers=cfg.layers,
            num_key_value_heads=cfg.base_model_config.num_key_value_heads,
            head_dim=cfg.base_model_config.head_dim
        )
        sample_string = user_model_config.build_prompt("类型#裤*风格#英伦*风格#简约")
        await make_and_put_user_model(testbed, user_model_config, data.trained_model_path_of("ptuning_7B.pt"), user_model_name)
    elif user_model == "lora":
        user_model_name = "adv-gen"
        user_model_config = LoraConfig(
            r_q=-1,
            r_k=0,
            r_v=-1,
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.0,
            len_limit=128,
        ).check_or_set_decoders_config(
            num_layers=cfg.layers, head_dim=cfg.base_model_config.head_dim,
            num_key_value_heads=cfg.base_model_config.num_key_value_heads, num_query_heads=cfg.base_model_config.num_query_heads
        )
        sample_string = user_model_config.build_prompt("类型#裤*风格#英伦*风格#简约")
        await make_and_put_user_model(testbed, user_model_config, data.trained_model_path_of("lora_7B.pt"), user_model_name)

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    await run_speed_test_multi(
        testbed=testbed,
        groups=groups,
        test_config=InferTestConfig(
            block_retrieve=False,
            recover=(0, False),
        ),
        output_file=os.path.join(os.path.dirname(__file__), "results", f"fast.json"),
        max_len=1024, multi_task_count=8,
        user_model_name=user_model_name,
        sample_string=sample_string
    )

asyncio.config_impl("golang", True)
asyncio.run(amain())
