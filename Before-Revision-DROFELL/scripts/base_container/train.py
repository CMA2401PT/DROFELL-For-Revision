import asyncio
import json
import os
from typing import cast

import torch
from containers.fast import FastContainer
from containers.measure import MeasureContainer
from containers.pressure import PressureReleaseContainer
from containers.solve import SolveContainer
# from operators.container.container import LeakDetectContainer
from train_base import TrainConfig, prepare
import extend_asyncio as asyncio
# import asyncio
asyncio.config_impl("golang", True)
cfg = TrainConfig(
    lr=1e-2,
    eps=1e-8,
    max_query_tokens=-1,
    max_response_tokens=-1,
    force_query_tokens=128,
    force_response_tokens=256,
    bfloat16_end=0,
    model_name="teared-Llama-2-7B-GPTQ"
)

device = torch.zeros((0,), device="cuda").device
task = prepare(cfg=cfg, device=device)
stream = cast(torch.cuda.Stream, torch.cuda.Stream())


async def amain():
    mode = "leak_detect"
    # if mode == "leak_detect":
    #     container = LeakDetectContainer(stream)
    if mode == "fast":
        container = FastContainer(stream)
    elif mode == "measure":
        container = MeasureContainer(stream)
    elif mode == "solve":
        operate_load_file = os.path.join(os.path.dirname(__file__), "train_profile.json")
        with open(operate_load_file, "r") as f:
            operate_load = json.load(f)
        container = SolveContainer(stream, operate_load)
    elif mode == "pressure":
        operate_load_file = os.path.join(os.path.dirname(__file__), "train_profile.json")
        with open(operate_load_file, "r") as f:
            operate_load = json.load(f)
        container = PressureReleaseContainer(stream, operate_load)
    else:
        raise ValueError(mode)
    loop = asyncio.get_event_loop()
    loop.set_blocking_scheduler(container)
    await task(50, container)
    print(f"blocking time: {loop.blocking_calltime_ms:.4f}ms")
    print(f"nonblocking time: {loop.non_blocking_calltime_ms:.4f}ms")
    profile_file = os.path.join(os.path.dirname(__file__), "results", f"{mode}_train.json")
    container.dump_profile(profile_file)
    if isinstance(container, MeasureContainer):
        profile_out = os.path.join(os.path.dirname(__file__), "train_profile.json")
        with open(profile_out, "w") as f:
            json.dump(container.compute_stats(), f)
    if isinstance(container, SolveContainer):
        alpha, beta = container.estimate_model()
        print(f"alpha={alpha}, beta={beta}")
        with open(os.path.join(os.path.dirname(__file__), "model_solve.json"), "w") as f:
            json.dump(container.pressure_records, f)
asyncio.run(amain())
