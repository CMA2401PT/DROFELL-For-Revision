import os
from typing import cast
import torch

from operators.container.container import NonAsyncContainer
from operators.core.ptuning.common import PtuningConfig
from scripts.base.train import run, TrainConfig
import extend_asyncio as asyncio

cfg = TrainConfig(
    user_model_config=PtuningConfig(
        build_prompt_str="[instruction]{}\n[generation]",
        dropout_p=0.1,
        prefix_len=128
    ),
    lr=1e-2,
    final_lr=1e-5,
    eps=1e-8,
    max_query_tokens=-1,
    max_response_tokens=-1,
    bfloat16_end=0,
    nan_steps=-1,
    model_name="teared-Llama-2-7B-GPTQ",
    event_dir=os.path.join(os.path.dirname(__file__), "ptuning-llama-2-7B")
)


async def main():
    device = torch.cuda.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    container = NonAsyncContainer(stream)
    # loop = asyncio.get_event_loop()
    # from operators.metrics.base_container.containers.fast import FastContainer
    # container = FastContainer(stream)
    # loop.set_blocking_scheduler(container)
    train_task = run(cfg, container, device=container.device)
    await train_task

asyncio.config_impl("golang", True)
asyncio.run(main())
