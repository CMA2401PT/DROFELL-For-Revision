"""General offline inference benchmark configured only through CLI options.

See ``README.md`` for complete 7B/70B and single/multi-GPU examples. Unknown
arguments are rejected; the effective values are written to ``args.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import extend_asyncio as asyncio
import torch
from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig
from scripts.offline.benchmark.common import (
    add_execution_arguments,
    add_memory_arguments,
    add_recovery_argument,
    make_groups,
    make_model_config,
    make_testbed,
    positive_int,
    validate_execution_arguments,
    validate_recovery_arguments,
    write_effective_args,
)
from scripts.offline.benchmark.inference.runner import (
    make_and_put_user_model,
    run_decode_benchmark,
    run_benchmark_multi,
)
from scripts.offline.testbed.infer_instance import InferTestConfig
from scripts.offline.testbed.model_config import ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline node-form inference benchmark without Orbit networking"
    )
    add_execution_arguments(parser)
    add_memory_arguments(parser)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--measurement",
        choices=("request", "decode"),
        default="request",
        help=(
            "request writes per-token completion times for one workload; "
            "decode measures fixed post-prefill decode rounds."
        ),
    )
    parser.add_argument(
        "--num-seqs",
        type=positive_int,
        nargs="+",
        required=True,
        help=(
            "Concurrent sequence counts. request mode accepts exactly one; "
            "decode mode accepts one or more."
        ),
    )
    parser.add_argument("--max-tokens", type=positive_int)
    parser.add_argument("--warmup-tokens", type=positive_int)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--warmup-decode-tokens", type=int, default=8)
    parser.add_argument("--decode-tokens", type=positive_int, default=200)
    parser.add_argument(
        "--max-concurrency",
        type=positive_int,
        help=(
            "Brain merge/concurrency limit. Defaults to the largest "
            "--num-seqs value for backward compatibility."
        ),
    )
    parser.add_argument("--prompt", default="once upon a time")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--block-retrieve", action="store_true")
    add_recovery_argument(parser)
    parser.add_argument(
        "--adapter",
        choices=("none", "ptuning", "lora"),
        default="none",
    )
    parser.add_argument("--adapter-name", default="user-model")
    parser.add_argument("--adapter-state")
    parser.add_argument(
        "--adapter-prompt-template",
        default="[instruction]{}\n[generation]",
    )
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--ptuning-prefix-len", type=positive_int, default=128)
    parser.add_argument("--lora-r-q", type=int, default=-1)
    parser.add_argument("--lora-r-k", type=int, default=0)
    parser.add_argument("--lora-r-v", type=int, default=-1)
    parser.add_argument("--lora-len-limit", type=positive_int, default=128)
    args = parser.parse_args()
    validate_execution_arguments(parser, args)
    validate_recovery_arguments(parser, args.recover, allow_step_zero=False)
    if args.cuda_graph and args.decoder_backend != "marlin":
        parser.error("--cuda-graph requires --decoder-backend marlin")
    if args.adapter != "none" and not args.adapter_state:
        parser.error("--adapter-state is required when --adapter is not none")
    if args.adapter == "none" and args.adapter_state:
        parser.error("--adapter-state requires --adapter ptuning or lora")
    if not 0.0 <= args.adapter_dropout <= 1.0:
        parser.error("--adapter-dropout must be in [0, 1]")
    if len(set(args.num_seqs)) != len(args.num_seqs):
        parser.error("--num-seqs values must be unique")
    if args.warmup_requests < 0 or args.warmup_decode_tokens < 0:
        parser.error("decode warmup counts must be non-negative")
    if args.measurement == "request":
        if len(args.num_seqs) != 1:
            parser.error("request mode requires exactly one --num-seqs value")
        if args.max_tokens is None:
            parser.error("request mode requires --max-tokens")
        if args.warmup_tokens is None:
            args.warmup_tokens = 32
    else:
        if args.max_tokens is not None or args.warmup_tokens is not None:
            parser.error(
                "decode mode uses --warmup-decode-tokens and "
                "--decode-tokens, not --max-tokens/--warmup-tokens"
            )
        if args.adapter != "none":
            parser.error("decode mode does not support adapters")
        if args.block_retrieve:
            parser.error("decode mode requires retrieval to remain disabled")
        if args.recover:
            parser.error("decode mode does not support recovery events")
    return args


def make_adapter(args: argparse.Namespace):
    if args.adapter == "none":
        return None
    if args.adapter == "ptuning":
        return PtuningConfig(
            build_prompt_str=args.adapter_prompt_template,
            dropout_p=args.adapter_dropout,
            prefix_len=args.ptuning_prefix_len,
        )
    return LoraConfig(
        r_q=args.lora_r_q,
        r_k=args.lora_r_k,
        r_v=args.lora_r_v,
        build_prompt_str=args.adapter_prompt_template,
        dropout_p=args.adapter_dropout,
        len_limit=args.lora_len_limit,
    )


async def amain(args: argparse.Namespace, model_config: ModelConfig) -> None:
    output_dir = Path(args.output)
    testbed = make_testbed(
        args,
        model_config,
        max_concurrency=args.max_concurrency or max(args.num_seqs),
    )
    adapter_config = make_adapter(args)
    adapter_name = None
    prompt = args.prompt
    if adapter_config is not None:
        adapter_name = args.adapter_name
        await make_and_put_user_model(
            testbed=testbed,
            cfg=adapter_config,
            state_file=args.adapter_state,
            user_model_name=adapter_name,
        )
        prompt = adapter_config.build_prompt(prompt)
    prompt_tokens = testbed.tokenizer.encode(prompt)
    warmup_prompt_tokens = len(prompt_tokens)
    if args.measurement == "request":
        if warmup_prompt_tokens + args.warmup_tokens > model_config.max_seq_len:
            raise ValueError(
                "warmup prompt plus --warmup-tokens exceeds the effective "
                "maximum sequence length"
            )
    else:
        request_tokens = (
            1 + args.warmup_decode_tokens + args.decode_tokens
        )
        if warmup_prompt_tokens + request_tokens > model_config.max_seq_len:
            raise ValueError(
                "prompt plus decode benchmark output exceeds the effective "
                "maximum sequence length"
            )
    write_effective_args(
        output_dir,
        args,
        extra={
            "effective_max_seq_len": model_config.max_seq_len,
            "model_decoder_layers": model_config.layers,
            "effective_prompt": prompt,
            "warmup_prompt_tokens": warmup_prompt_tokens,
        },
    )
    groups = make_groups(
        model_config,
        num_devices=len(args.devices),
        user_model_prefix="user-infer",
    )
    test_config = InferTestConfig(
        block_retrieve=args.block_retrieve,
        recover=args.recover,
    )
    if args.measurement == "request":
        await run_benchmark_multi(
            testbed=testbed,
            groups=groups,
            test_config=test_config,
            output_file=str(output_dir / "result.json"),
            max_len=args.max_tokens,
            multi_task_count=args.num_seqs[0],
            user_model_name=adapter_name,
            sample_string=prompt,
            warmup_tokens=args.warmup_tokens,
        )
        print("to review the result, run:")
        print(
            "python scripts/offline/benchmark/inference/analysis_utils.py "
            f"{output_dir}"
        )
        return

    cases = await run_decode_benchmark(
        testbed=testbed,
        groups=groups,
        prompt_tokens=prompt_tokens,
        test_config=test_config,
        num_seqs_values=args.num_seqs,
        runs=args.runs,
        warmup_requests=args.warmup_requests,
        warmup_decode_tokens=args.warmup_decode_tokens,
        decode_tokens=args.decode_tokens,
    )
    gpu_info = []
    decode_graph = []
    for brain in testbed.brains:
        properties = torch.cuda.get_device_properties(brain.device)
        gpu_info.append(
            {
                "device": str(brain.device),
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
            }
        )
        decode_graph.append(
            {
                "device": str(brain.device),
                "stats": brain.query_extra_info("decode_graph"),
            }
        )
    report = {
        "schema_version": 1,
        "engine": "drofell-offline",
        "model": model_config.model_name,
        "execution": "graph" if args.cuda_graph else "eager",
        "decoder_backend": args.decoder_backend,
        "execution_runtime": {
            "async_backend": args.async_backend,
            "busy_wait": args.busy_wait,
            "container": args.container,
        },
        "measurement": {
            "mode": "decode",
            "runs": args.runs,
            "warmup_requests": args.warmup_requests,
            "warmup_decode_tokens_per_seq": args.warmup_decode_tokens,
            "decode_tokens_per_seq": args.decode_tokens,
            "max_model_len": model_config.max_seq_len,
            "timing_scope": "post_prefill_offline_decode_end_to_end",
        },
        "topology": {
            "brains": len(testbed.brains),
            "devices": args.devices,
            "decoder_sizes": args.decoder_splits,
            "max_concurrency": args.max_concurrency or max(args.num_seqs),
            "retrieve": False,
        },
        "gpu": gpu_info,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cases": cases,
        "decode_graph": decode_graph,
    }
    (output_dir / "result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    model_config = make_model_config(args, enable_cuda_graph=args.cuda_graph)
    if (
        args.measurement == "request"
        and 1 + args.max_tokens > model_config.max_seq_len
    ):
        raise ValueError(
            "the measured one-token prompt plus --max-tokens exceeds the "
            "effective maximum sequence length"
        )
    asyncio.config_impl(args.async_backend, args.busy_wait)
    asyncio.run(amain(args, model_config))


if __name__ == "__main__":
    main()
