import os
import pickle

import torch
from scripts.qllm.infer.infer_testbed import run, InferConfig
from operators.container.container import NonAsyncContainer
import extend_asyncio as asyncio


async def amain():
    container = NonAsyncContainer(torch.cuda.current_stream())
    cfg = InferConfig(blocks_count=640)
    cfg.cache_kernel_implement = "vllm"
    result = await run(cfg, container, do_memory=False, total_tokens=2100)
    outdir = os.path.join(os.path.dirname(__file__), "cache_kernel")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "kernel_vllm_2k.pkl"), "wb") as f:
        pickle.dump(result, f)

asyncio.config_impl("golang", True)
asyncio.run(amain())
