"""Deterministic offline inference/recovery correctness runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

import data
from dealers.common.base_types import balance_divide
from dealers.memory_schedule.brain import Brain
import extend_asyncio as asyncio
from operators.container.base_container import BaseContainer
from operators.core.ptuning.common import PtuningConfig
from operators.core.sampler.common import SamplingParams
from operators.core.tokenizer.common import Sequence, SequenceCore, StopConfig
from scripts.offline.benchmark.common import (
    add_recovery_argument,
    positive_int,
    validate_recovery_arguments,
)
from scripts.offline.testbed import (
    InferTestConfig,
    ModelConfig,
    PipelineParallelTestbed,
)
import split_qllm_models


MODEL_ID = "teared-Llama-2-7B-GPTQ"
DEFAULT_OUTPUT = Path(__file__).with_name("infer_with_recover_record.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline inference recovery correctness test",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--decoder-backend",
        choices=("marlin", "exllama", "exllama_fuse"),
        default="marlin",
    )
    parser.add_argument("--max-tokens", type=positive_int, default=256)
    parser.add_argument("--warmup-tokens", type=positive_int, default=32)
    parser.add_argument(
        "--groups",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="Number of local InferNode groups across the four decoder shards.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Generate exactly --max-tokens even if EOS is sampled.",
    )
    parser.add_argument(
        "--block-retrieve",
        action="store_true",
        help="Enable periodic KV block retrieval for the configured run.",
    )
    add_recovery_argument(parser)
    parser.add_argument(
        "--recovery-replay",
        choices=("sequential", "chunk"),
        default="chunk",
        help="Replay uncovered KV-checkpoint inputs individually or as one chunk.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--async-backend",
        choices=("py", "chimera", "golang"),
        default="golang",
    )
    parser.add_argument(
        "--busy-wait",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    validate_recovery_arguments(
        parser, args.recover, allow_step_zero=False)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        parser.error("--device must be an indexed CUDA device such as cuda:0")
    if device.index >= torch.cuda.device_count():
        parser.error(
            "requested CUDA device is unavailable; "
            f"visible device count={torch.cuda.device_count()}"
        )
    if args.recover and max(step for step, _ in args.recover) >= args.max_tokens:
        parser.error("every --recover step must be smaller than --max-tokens")
    return args


async def run_case(
    *,
    name: str,
    testbed: PipelineParallelTestbed,
    groups: list[list[tuple[str, str]]],
    client_sequence: Sequence,
    max_tokens: int,
    ignore_eos: bool,
    test_config: InferTestConfig,
    user_model_name: str,
) -> dict[str, Any]:
    sequence = client_sequence.clone()
    sequence.core.stop_config = StopConfig(
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
    )
    instance = await testbed.create_infer_instance(
        groups=groups,
        seq_core=sequence.core.clone(),
        sampling_params=SamplingParams(temperature=0.0, logprobs=1),
        test_config=test_config,
        user_model_name=user_model_name,
    )
    completion_ns: list[int] = []
    try:
        async for _, new_token, _ in instance.infer():
            sequence.core.append_token_id(new_token)
            testbed.tokenizer.incrementally_decode_seq(sequence)
            completion_ns.append(time.time_ns())
        print(instance.seq_id, name, ">", sequence.output_text)
        return {
            "completion_ns": completion_ns,
            "token_ids": sequence.core.output_token_ids.copy(),
            "output_text": sequence.output_text,
            "recovery_records": instance.recovery_records.copy(),
        }
    finally:
        instance.remove()
        for brain in testbed.brains:
            torch.cuda.synchronize(brain.device)


def assert_same_tokens(cases: dict[str, dict[str, Any]]) -> None:
    expected_name = "fast"
    expected = cases[expected_name]["token_ids"]
    for name, case in cases.items():
        actual = case["token_ids"]
        if actual == expected:
            continue
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(expected, actual))
                if left != right
            ),
            min(len(expected), len(actual)),
        )
        raise AssertionError(
            f"token mismatch: {expected_name} vs {name} at output index "
            f"{mismatch}; lengths={len(expected)},{len(actual)}"
        )


async def infer_main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    stream = torch.cuda.Stream(device=device)
    container = BaseContainer(stream)
    cfg = ModelConfig(
        model_name=MODEL_ID,
        decoder_backend=args.decoder_backend,
        group_fn=lambda layers: [
            min(8, layers - start) for start in range(0, layers, 8)
        ],
    )
    brain = Brain(
        models_path=split_qllm_models.lib_path,
        device=device,
        container=container,
        max_seq_len=cfg.max_seq_len,
        max_concurrency=8,
        decoder_implement=cfg.decoder_backend,
    )
    testbed = PipelineParallelTestbed([brain], cfg)
    groups = balance_divide(
        base_models=testbed.base_model_names,
        with_head=True,
        with_tail=True,
        decoders_groups=args.groups,
        user_model_prefix="user_model",
    )

    user_model_name = "advgen-ptuning"
    user_model_config = PtuningConfig(
        build_prompt_str="[instruction]{}\n[generation]",
        dropout_p=0.1,
        prefix_len=128,
    )
    testbed.put_model_from_config(
        name=user_model_name,
        cfg=user_model_config,
        state_file=data.trained_model_path_of("ptuning_7B.pt"),
    )
    prompt = user_model_config.build_prompt(
        "类型#裤*风格#英伦*风格#简约")
    prompt_ids = testbed.tokenizer.encode(prompt)
    longest_output = max(args.max_tokens, args.warmup_tokens)
    if len(prompt_ids) + longest_output > cfg.max_seq_len:
        raise ValueError(
            "prompt plus requested output tokens exceeds model max_seq_len")
    client_sequence = Sequence(core=SequenceCore(
        stop_config=StopConfig(
            max_tokens=args.max_tokens,
            ignore_eos=args.ignore_eos,
        ),
        prompt_token_ids=prompt_ids,
    ))

    await run_case(
        name="warmup",
        testbed=testbed,
        groups=groups,
        client_sequence=client_sequence,
        max_tokens=args.warmup_tokens,
        ignore_eos=True,
        test_config=InferTestConfig(),
        user_model_name=user_model_name,
    )

    cases = {
        "fast": await run_case(
            name="fast",
            testbed=testbed,
            groups=groups,
            client_sequence=client_sequence,
            max_tokens=args.max_tokens,
            ignore_eos=args.ignore_eos,
            test_config=InferTestConfig(block_retrieve=False),
            user_model_name=user_model_name,
        ),
        "common": await run_case(
            name="common",
            testbed=testbed,
            groups=groups,
            client_sequence=client_sequence,
            max_tokens=args.max_tokens,
            ignore_eos=args.ignore_eos,
            test_config=InferTestConfig(block_retrieve=True),
            user_model_name=user_model_name,
        ),
        "configured": await run_case(
            name="configured",
            testbed=testbed,
            groups=groups,
            client_sequence=client_sequence,
            max_tokens=args.max_tokens,
            ignore_eos=args.ignore_eos,
            test_config=InferTestConfig(
                block_retrieve=args.block_retrieve,
                recover=args.recover,
                record_recovery=True,
                recovery_replay=args.recovery_replay,
            ),
            user_model_name=user_model_name,
        ),
    }
    assert_same_tokens(cases)
    triggered_steps = {
        record["trigger_step"]
        for record in cases["configured"]["recovery_records"]
    }
    expected_steps = {step for step, _ in args.recover}
    if triggered_steps != expected_steps:
        raise AssertionError(
            "not every requested recovery was triggered; "
            f"expected={sorted(expected_steps)}, actual={sorted(triggered_steps)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 2,
        "config": {
            **vars(args),
            "output": str(args.output),
            "prompt_tokens": len(prompt_ids),
        },
        "tokens_match": True,
        "cases": cases,
        # Keep the original timing keys for existing ad-hoc analysis.
        "fast": cases["fast"]["completion_ns"],
        "common": cases["common"]["completion_ns"],
        "recover": cases["configured"]["completion_ns"],
    }
    report["recovery_gap_ms"] = {
        str(step): (
            cases["configured"]["completion_ns"][step]
            - cases["configured"]["completion_ns"][step - 1]
        ) / 1e6
        for step, _ in args.recover
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"token correctness: PASS ({len(cases['fast']['token_ids'])} tokens)")
    print(f"report: {args.output}")


def main() -> None:
    args = parse_args()
    asyncio.config_impl(args.async_backend, args.busy_wait)
    asyncio.run(infer_main(args))


if __name__ == "__main__":
    main()
