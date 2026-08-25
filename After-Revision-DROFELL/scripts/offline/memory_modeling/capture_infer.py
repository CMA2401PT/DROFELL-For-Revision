import argparse
from pathlib import Path

import torch

from dealers.analysis.frame import AnalysisFrame
from dealers.analysis.recorder import ActualIORecorder
from dealers.common.base_types import balance_divide
from dealers.memory_schedule.brain import Brain
from operators.core.lora.common import LoraConfig
import split_qllm_models

from operators.container.base_container import BaseContainer
from operators.core.ptuning.common import PtuningConfig
from operators.core.tokenizer.common import Sequence, SequenceCore, StopConfig
from operators.core.sampler.common import SamplingParams
import extend_asyncio as asyncio
import data

from scripts.offline.testbed import PipelineParallelTestbed
from scripts.offline.testbed.infer_instance import InferTestConfig
from scripts.offline.memory_modeling.capture_common import (
    add_capture_runtime_arguments,
    capture_metadata,
    dump_capture,
    make_memory_config,
    make_model_config,
    validate_capture_runtime_arguments,
)

# flags.ALLOW_2WAY_DISPATCH = False


async def amain(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    device = torch.zeros((0,), device="cuda:0").device
    stream = torch.cuda.Stream(device=device)
    container = BaseContainer(stream)

    cfg = make_model_config(args)
    memory_config = make_memory_config(args)
    brain = Brain(
        models_path=split_qllm_models.lib_path,
        device=device,
        container=container,
        memory_config=memory_config,
        max_seq_len=cfg.max_seq_len,
        max_concurrency=args.max_concurrency,
        decoder_implement=cfg.decoder_backend,
    )
    recorder = ActualIORecorder(brain)
    brain = AnalysisFrame.wrap_brain(brain, recorder)
    ptuning = args.method == "ptuning"
    user_model_name = f"advgen-{args.method}"
    state_file = ""
    if ptuning:
        user_model_config = PtuningConfig(
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.1,
            prefix_len=128
        )
        state_file = data.trained_model_path_of("ptuning_7B.pt")
    else:
        user_model_config = LoraConfig(
            r_q=-1,
            r_k=0,
            r_v=-1,
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.0,
            len_limit=128,
        )
        state_file = data.trained_model_path_of("lora_7B.pt")

    async def wrapper():
        testbed = PipelineParallelTestbed([brain], cfg)
        testbed.put_model_from_config(
            name=user_model_name, cfg=user_model_config, state_file=state_file
        )
        prompt = "once upon a day, a girl"
        prompt = user_model_config.build_prompt("类型#裤*风格#英伦*风格#简约")
        prompt_ids = testbed.tokenizer.encode(prompt)

        sampling_params = SamplingParams(temperature=0.0, logprobs=1)
        stop_config = StopConfig(max_tokens=args.max_tokens, ignore_eos=True)
        groups = balance_divide(
            base_models=testbed.base_model_names,
            with_head=True, with_tail=True,
            decoders_groups=args.pipeline_decoder_groups,
            user_model_prefix=f"user_model"
        )

        instance = await testbed.create_infer_instance(
            groups=groups,
            seq_core=SequenceCore(
                stop_config=stop_config,
                prompt_token_ids=prompt_ids.copy(),
            ),
            sampling_params=sampling_params,
            test_config=InferTestConfig(
                block_retrieve=False,
            ),
            user_model_name=user_model_name,
        )
        seq = Sequence(core=SequenceCore(
            stop_config=stop_config,
            prompt_token_ids=prompt_ids,
        ))
        async for (_, new_token, _) in instance.infer():
            seq.core.append_token_id(new_token)
            testbed.tokenizer.incrementally_decode_seq(seq)
        print(instance.seq_id, "> ", seq.output_text)
        instance.remove()
        await asyncio.sleep(0.1)
        for provider in testbed.providers:
            provider.remove()
        testbed.providers.clear()
        torch.cuda.synchronize(device)
    await wrapper()
    print("dump state")
    dump_capture(
        args.output,
        recorder,
        capture_metadata(
            cfg=cfg,
            memory_config=memory_config,
            max_concurrency=args.max_concurrency,
            decoder_group_size=args.decoder_group_size,
            random_seed=args.seed,
            scenario={
                "kind": "infer",
                "method": args.method,
                "max_tokens": args.max_tokens,
                "pipeline_decoder_groups": args.pipeline_decoder_groups,
            },
        ),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--pipeline-decoder-groups", type=int, default=1,
        help="number of decoder route groups in the inference instance",
    )
    parser.add_argument(
        "--method", choices=["ptuning", "lora"], default="ptuning")
    add_capture_runtime_arguments(
        parser, default_max_concurrency=1, default_preallocate=False
    )
    parsed = parser.parse_args()
    validate_capture_runtime_arguments(parser, parsed)
    if parsed.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if parsed.pipeline_decoder_groups <= 0:
        parser.error("--pipeline-decoder-groups must be positive")
    asyncio.config_impl("golang", True)
    asyncio.run(amain(parsed))
