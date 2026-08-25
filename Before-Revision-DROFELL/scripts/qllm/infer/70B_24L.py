import os
import pickle

import torch
from scripts.qllm.infer.infer_testbed import run, InferConfig
from operators.container.container import NonAsyncContainer
import extend_asyncio as asyncio
import data


async def amain():
    container = NonAsyncContainer(torch.cuda.current_stream())
    cfg = InferConfig(
        model_name="teared-Llama-2-70B-GPTQ",
        blocks_count=128*8,
        max_layers_to_load=24,
        encoder_data=data.trained_model_path_of("ptuning_70B_24Layers.pt"),
        cache_kernel_implement="dual_table"
    )
    result = await run(cfg, container)
    outdir = os.path.join(os.path.dirname(__file__), "cache_kernel")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "kernel_dual_table_70B_24L.pkl"), "wb") as f:
        pickle.dump(result, f)

asyncio.config_impl("golang", True)
asyncio.run(amain())
