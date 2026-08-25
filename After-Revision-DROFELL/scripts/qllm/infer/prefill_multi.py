import json
import os
import torch
from scripts.qllm.infer.infer_testbed import InferTestBed, InferConfig, run_prefill
from operators.container.container import NonAsyncContainer
import extend_asyncio as asyncio
from operators import flags


async def amain():
    flags.MULTI_PREFILL_STREAM = True
    container = NonAsyncContainer(torch.cuda.current_stream())
    cfg = InferConfig(blocks_count=256)
    test_bed = InferTestBed(cfg)
    summary = {}
    test_bed.fill_user_model(128)
    ret = await run_prefill(container, test_bed, 8, True)
    summary["memory"] = ret["memory"]
    summary["128+256 8"] = ret["256 8"]
    summary["128+512 8"] = ret["512 8"]
    summary["128+1024 8"] = ret["1024 8"]

    ret = await run_prefill(container, test_bed, 2, False)
    summary["128+256 2"] = ret["256 2"]
    summary["128+512 2"] = ret["512 2"]
    summary["128+1024 2"] = ret["1024 2"]

    test_bed.fill_user_model(256)
    ret = await run_prefill(container, test_bed, 8)
    summary["256+256 8"] = ret["256 8"]
    summary["256+512 8"] = ret["512 8"]
    summary["256+1024 8"] = ret["1024 8"]
    ret = await run_prefill(container, test_bed, 2)
    summary["256+256 2"] = ret["256 2"]
    summary["256+512 2"] = ret["512 2"]
    summary["256+1024 2"] = ret["1024 2"]

    test_bed.fill_user_model(512)
    ret = await run_prefill(container, test_bed, 8)
    summary["512+256 8"] = ret["256 8"]
    summary["512+512 8"] = ret["512 8"]
    summary["512+1024 8"] = ret["1024 8"]
    ret = await run_prefill(container, test_bed, 2)
    summary["512+256 2"] = ret["256 2"]
    summary["512+512 2"] = ret["512 2"]
    summary["512+1024 2"] = ret["1024 2"]

    outdir = os.path.join(os.path.dirname(__file__), "prefill")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "multi.json"), "w") as f:
        json.dump(summary, f)

asyncio.config_impl("golang", True)
asyncio.run(amain())
