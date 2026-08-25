import os
import pickle
import time
from typing import List

import torch
from torch import optim
from operators.container.container import NonAsyncContainer
from operators.core.swappable.real import CanSwapObject
from operators.core.ptuning.real import Ptuning, PtuningConfig
from operators.assemble.decoders.real import DecodersImplement, make_decoders_from_files
from operators.assemble.fwd_ctx import forward, backward
from operators.helpers.model_lib import ModelsLib
from operators import flags
import split_qllm_models

from plot import analysis_forward_backward_memory
import matplotlib.pyplot as plt
import seaborn as sns

import extend_asyncio as asyncio

target_model = "teared-Llama-2-7B-GPTQ"
decoder_implement: DecodersImplement = "our_fuse"
layers_to_load = 8
batch_size = 4
seq_len = 512
prompt_seq_len = 256
device = torch.zeros((0,)).cuda().device

prefix_encoder_config = PtuningConfig(
    build_prompt_str="[instruction]{}\n[generation]",
    prefix_len=prompt_seq_len
)

models = ModelsLib(split_qllm_models.lib_path)
model_info = models.models_storage[target_model]


with device:
    decoders_group = make_decoders_from_files(
        implement=decoder_implement,
        config=model_info.config.quant_decoder_config,
        state_files=[
            model_info.decoder_path(layer_i)
            for layer_i in range(layers_to_load)
        ]
    )
    prefix_encoder_config.num_key_value_heads = model_info.config.num_key_value_heads
    prefix_encoder_config.head_dim = model_info.config.head_dim
    prefix_encoder_config.num_layers = layers_to_load
    prefix_encoder = Ptuning(prefix_encoder_config)

optimizer = optim.AdamW(prefix_encoder.parameters(), lr=1e-2, eps=1e-4)

container = NonAsyncContainer(torch.cuda.current_stream(device))


async def record_forward_and_backward(do_swap: bool):
    async def do_forward():
        def create_input():
            hidden_states = torch.empty(
                size=(batch_size*seq_len, model_info.config.hidden_size),
                device=device,
                dtype=torch.float16
            )
            position_ids_tensor = torch.arange(0, seq_len, dtype=torch.int32).repeat(batch_size)
            return hidden_states, position_ids_tensor
        hidden_states, position_ids_tensor = await container.step("create_input", create_input)
        input_lens = [seq_len for _ in range(batch_size)]
        _, forward_ctx = await forward(
            container=container,
            hidden_states=hidden_states,
            position_ids=position_ids_tensor,
            input_lens=input_lens,
            user_model=prefix_encoder,
            decoders=decoders_group
        )
        return forward_ctx

    async def do_backward(forward_ctx: CanSwapObject):
        hidden_states_grad = await container.step(
            "create_fake_grad", torch.empty, size=(batch_size*seq_len, model_info.config.hidden_size),
            device=device,
            dtype=torch.float16
        )
        await backward(
            container=container,
            grad=hidden_states_grad,
            ctx=forward_ctx,
            user_model=prefix_encoder,
            decoders=decoders_group
        )
        optimizer.zero_grad()

    async def warmup(do_swap: bool):
        for _ in range(5):
            forward_ctx = await do_forward()

            def do_swap_out_in():
                forward_ctx.swap_out(None)
                forward_ctx.swap_in(device=device, stream=None)
                torch.cuda.synchronize()
            if do_swap:
                await container.step("do_swap_out_in", do_swap_out_in)
            await do_backward(forward_ctx)

    async def time_metrics(do_swap: bool):
        forward_time: List[int] = []
        backward_time: List[int] = []
        swap_time: List[int] = []
        for _ in range(100):
            forward_start = time.time_ns()
            forward_ctx = await do_forward()
            torch.cuda.synchronize()
            forward_complete = time.time_ns()

            if do_swap:
                def do_swap_out_in():
                    swap_start = forward_complete
                    forward_ctx.swap_out(None)
                    forward_ctx.swap_in(device=device, stream=None)
                    torch.cuda.synchronize()
                    swap_complete = time.time_ns()
                    swap_time.append(swap_complete-swap_start)
                await container.step("do_swap_out_in", do_swap_out_in)
            backward_start = time.time_ns()
            await do_backward(forward_ctx)
            torch.cuda.synchronize()
            backward_complete = time.time_ns()
            forward_time.append(forward_complete-forward_start)
            backward_time.append(backward_complete-backward_start)
        return forward_time, swap_time, backward_time

    async def memory_metrics(do_swap: bool):
        forward_ctx = await do_forward()
        snapshot_after_forward = torch.cuda.memory._snapshot()
        snapshot_after_swapped_out = None
        snapshot_after_swapped_in = None
        if do_swap:
            def do_swap_out_in():
                forward_ctx.swap_out(None)
                torch.cuda.synchronize()
                snapshot_after_swapped_out = torch.cuda.memory._snapshot()
                forward_ctx.swap_in(device=device, stream=None)
                torch.cuda.synchronize()
                snapshot_after_swapped_in = torch.cuda.memory._snapshot()
                return snapshot_after_swapped_out, snapshot_after_swapped_in
            snapshot_after_swapped_out, snapshot_after_swapped_in = await container.step("do_swap_out_in", do_swap_out_in)
        await do_backward(forward_ctx)
        # make ctx free
        return snapshot_after_forward, snapshot_after_swapped_out, snapshot_after_swapped_in

    print("test start")
    await warmup(do_swap)  # optimizer, cudablasWorkSpace
    print("warm up complete, do time metrics")
    forward_time, swap_time, backward_time = await time_metrics(do_swap)
    print("metrics complete, do memory record")
    torch.cuda.memory._record_memory_history()
    snapshot_after_forward, snapshot_after_swap_out, snapshot_after_swapped_in, = await memory_metrics(do_swap)
    # make ctx free
    snapshot_after_backward = torch.cuda.memory._snapshot()
    print("all done")
    return {
        "forward_memory": snapshot_after_forward,
        "swap_out_memory": snapshot_after_swap_out,
        "swap_in_memory": snapshot_after_swapped_in,
        "backward_memory": snapshot_after_backward,
        "forward_time": forward_time,
        "swap_time": swap_time,
        "backward_time": backward_time,
    }


async def test_and_save_result(name: str, do_swap: bool):
    dir_name = os.path.join(os.path.dirname(__file__), name)
    os.makedirs(dir_name, exist_ok=True)
    record = await record_forward_and_backward(do_swap)
    with open(os.path.join(dir_name, "record.pkl"), "wb") as fp:
        pickle.dump(record, fp)
    ret = analysis_forward_backward_memory(
        forward_trace=record["forward_memory"]['device_traces'][0],
        swap_out_trace=record["swap_out_memory"]['device_traces'][0] if record["swap_out_memory"] is not None else None,
        swap_in_trace=record["swap_in_memory"]['device_traces'][0] if record["swap_in_memory"] is not None else None,
        backward_trace=record["backward_memory"]['device_traces'][0],
        forward_time=record["forward_time"],
        swap_time=record["swap_time"],
        backward_time=record["backward_time"],
    )

    plt.figure(figsize=(8, 5), dpi=330)
    sns.set_theme(font="Times New Roman")
    ret.plot({})
    plt.savefig(os.path.join(dir_name, "summary.png"), dpi=330)


flags.RECONSTRUCTE_WEIGTH_FLOAT16 = False
flags.RECONSTRUCTE_WEIGTH_BFLOAT16 = False
flags.REUSE_RECONSTRUCT_CACHE_WHEN_GRAD = False


async def test_all():
    for user_model_checkpoint in [True, False]:
        for decoders_impl in ["grad", "each_decoder", "first.grad", "first.2pass"]:

            flags.USER_MODEL_ENABLE_CHECKPOTIN = user_model_checkpoint
            flags.DECODERS_TRAIN_METHOD = decoders_impl

            name = ""
            if flags.RECONSTRUCTE_WEIGTH_FLOAT16:
                name = "float16"
            if flags.RECONSTRUCTE_WEIGTH_BFLOAT16:
                if flags.RECONSTRUCTE_WEIGTH_FLOAT16:
                    name = "float16_"
                name += "bfloat16"
            if name == "":
                name = "quant"
            if flags.REUSE_RECONSTRUCT_CACHE_WHEN_GRAD:
                name += "_cache_reuse"
            name += "_user_"+("seed" if flags.USER_MODEL_ENABLE_CHECKPOTIN else "grad")
            name += "_base_"+flags.DECODERS_TRAIN_METHOD
            print(f"run: {name}")
            await test_and_save_result(f"results/{name}", True)

asyncio.config_impl("golang", True)
asyncio.run(test_all())
