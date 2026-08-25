import argparse
from typing import Optional, cast

import torch


from dealers.common.base_types import balance_divide
from dealers.memory_schedule.brain import Brain
from dealers.memory_schedule.restriction import MemoryConfigStatic
from operators.core.lora.common import LoraConfig
import split_qllm_models

from dealers.real_basic.basic import BasicBrain

from operators.container.base_container import BaseContainer
from operators.core.ptuning.common import PtuningConfig
from operators.core.tokenizer.common import Sequence, SequenceCore, StopConfig
from operators.core.sampler.common import SamplingParams
import extend_asyncio as asyncio
import data

from scripts.offline.testbed import ModelConfig, PipelineParallelTestbed
from scripts.offline.testbed.infer_instance import InferTestConfig


async def amain(args: argparse.Namespace):
    device = torch.zeros((0,), device="cuda:0").device
    stream = torch.cuda.Stream(device=device)
    container = BaseContainer(stream)

    swap_memory_config = MemoryConfigStatic(
        graph_budget=args.graph_budget_mib * 2**20
    )
    if args.mode == "swap":
        swap_memory_config.root_memory = int(6.5*(2**30)) - int(198*(2**20))
        swap_memory_config.major_reserved = (2**20)*64
    cfg = ModelConfig(
        model_name="teared-Llama-2-7B-GPTQ",
        decoder_backend=args.decoder_backend,
        enable_cuda_graph=args.cuda_graph,
        max_seq_len=args.graph_max_context_len,
        max_layers_to_load=args.layers,
        group_fn=lambda layers: [min(8, layers-start)
                                 for start in range(0, layers, 8)],
    )
    brain = Brain(
        models_path=split_qllm_models.lib_path,
        device=device,
        container=container,
        max_seq_len=cfg.max_seq_len,
        max_concurrency=8,
        decoder_implement=cfg.decoder_backend,
        memory_config=swap_memory_config
    )
    testbed = PipelineParallelTestbed([brain], cfg)

    user_model_name: Optional[str] = None
    user_model_config = None
    if args.adapter == "ptuning":
        user_model_name = "advgen-ptuning"
        user_model_config = PtuningConfig(
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.1,
            prefix_len=128
        )
        state_file = data.trained_model_path_of("ptuning_7B.pt")
    elif args.adapter == "lora":
        user_model_name = "advgen-lora"
        user_model_config = LoraConfig(
            r_q=-1,
            r_k=0,
            r_v=-1,
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.0,
            len_limit=128,
        )
        state_file = data.trained_model_path_of("lora_7B.pt")
    if user_model_config is not None:
        testbed.put_model_from_config(
            name=cast(str, user_model_name), cfg=user_model_config, state_file=state_file
        )
        prompt = user_model_config.build_prompt("类型#裤*风格#英伦*风格#简约")
    else:
        prompt = "once upon a day, a girl"
    prompt_ids = testbed.tokenizer.encode(prompt)

    async def infer_instance(i: int):
        sampling_params = SamplingParams(temperature=0.0, logprobs=1)
        stop_config = StopConfig(
            max_tokens=args.max_tokens, ignore_eos=args.ignore_eos)
        groups = balance_divide(
            base_models=testbed.base_model_names,
            with_head=True, with_tail=True, decoders_groups=1,
            user_model_prefix=f"user_model-{i}"
        )

        instance_orig = await testbed.create_infer_instance(
            groups=groups,
            seq_core=SequenceCore(
                stop_config=stop_config,
                prompt_token_ids=prompt_ids.copy(),
            ),
            sampling_params=sampling_params,
            test_config=InferTestConfig(
                block_retrieve=True,
                recover=(
                    [(20, False), (30, False), (80, True), (100, False)]
                    if args.mode == "recover" else
                    []
                ),
            ),
            user_model_name=user_model_name,
        )
        seq = Sequence(core=SequenceCore(
            stop_config=stop_config,
            prompt_token_ids=prompt_ids,
        ))
        async for (_, new_token, _) in instance_orig.infer():
            seq.core.append_token_id(new_token)
            testbed.tokenizer.incrementally_decode_seq(seq)
        instance_orig.remove()
        print(instance_orig.seq_id, "> ", seq.output_text)
    tasks = [
        infer_instance(i)
        for i in range(args.parallel)
    ]
    await asyncio.gather(*tasks)
    if args.cuda_graph:
        print("decode graph:", brain.query_extra_info("decode_graph"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite offline Marlin inference smoke test")
    parser.add_argument(
        "--adapter", choices=["base", "lora", "ptuning"], default="base")
    parser.add_argument(
        "--decoder-backend", choices=["marlin", "exllama", "exllama_fuse"], default="marlin")
    parser.add_argument(
        "--mode", choices=["normal", "swap", "recover"], default="normal")
    parser.add_argument("--layers", type=int, default=-1)
    parser.add_argument("--parallel", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--graph-budget-mib", type=int, default=0)
    parser.add_argument("--graph-max-context-len", type=int, default=256)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--async-backend",
                        choices=["py", "chimera", "golang"], default="golang")
    args = parser.parse_args()
    if args.graph_budget_mib < 0:
        parser.error("--graph-budget-mib must be non-negative")
    if args.cuda_graph and args.graph_budget_mib <= 0:
        parser.error("--cuda-graph requires a positive --graph-budget-mib")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    asyncio.config_impl(parsed.async_backend, True)
    asyncio.run(amain(parsed))
