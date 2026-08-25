from typing import cast

import torch

from dealers.common.provider import Provider
import split_qllm_models

from dealers.real_basic.basic import BasicBrain

from operators.container.base_container import BaseContainer
import extend_asyncio as asyncio


async def amain():
    device = torch.zeros((0,), device="cuda:0").device
    stream = cast(torch.cuda.Stream, torch.cuda.Stream(device))
    container = BaseContainer(stream)

    brain = BasicBrain(
        models_path=split_qllm_models.lib_path,
        device=device,
        container=container
    )

    provider = Provider(brain)
    dealer = provider.get_model_decoders_dealer(
        f"teared-Llama-2-7B-GPTQ:decoders-{0:0>3}-{32:0>3}"
    )
    # tps = await dealer.underlay_model.get_prefill_through_output(
    #     per_batch_size=(1, 1024),
    #     batch_count=10
    # )
    # print(tps)

    # tps = await dealer.underlay_model.get_forward_backward_output(
    #     per_batch_size=(1, 1024),
    #     batch_count=10
    # )
    # print(tps)

    tps = await dealer.underlay_model.get_generating_throughoutput(
        parallel_count=32,
        seq_len=128,
        repeat_times=1,
    )
    print(tps)

if __name__ == "__main__":
    asyncio.config_impl("golang", True)
    asyncio.run(amain())
