import os
import pickle

import torch
from scripts.qllm.infer.infer_testbed import run, InferConfig
from operators.container.container import NonAsyncContainer
import extend_asyncio as asyncio


async def amain():
    container = NonAsyncContainer(torch.cuda.current_stream())
    cfg = InferConfig(blocks_count=640)
    cfg.cache_kernel_implement = "dual_table"
    result = await run(cfg, container, total_tokens=128, do_memory=True)
    outdir = os.path.join(os.path.dirname(__file__), "cache_kernel")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "kernel_dual_table_memory.pkl"), "wb") as f:
        pickle.dump(result, f)

asyncio.config_impl("golang", True)
asyncio.run(amain())
