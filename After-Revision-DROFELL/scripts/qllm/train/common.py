import os

import torch
from operators import flags
from operators.container.container import NonAsyncContainer
from train_testbed import TrainConfig, run
import extend_asyncio as asyncio


async def main():
    flags.USER_MODEL_ENABLE_CHECKPOTIN = True
    flags.DECODERS_TRAIN_METHOD = "each_decoder"
    cfg = TrainConfig(
        lr=1e-2,
        eps=1e-4,
        event_dir=os.path.join(os.path.dirname(__file__), "records", "common"),
        max_query_tokens=-1,
        max_response_tokens=-1,
        # force_pad_tokens=256,
        batch_size=4,
    )
    cfg.ptuning_config.prefix_len = 128
    container = NonAsyncContainer(torch.cuda.current_stream())
    train_task = run(cfg, container, device=container.device)
    await train_task


asyncio.config_impl("golang", True)
asyncio.run(main())
