"""Decode-only benchmark for DROFELL's base Llama-2-7B-GPTQ path.

The timing protocol matches nano-vLLM's ``bench.py`` decode benchmark:

1. Run complete requests outside the measured runs for warmup.
2. For each measured run, execute prefill outside the decode timer.
3. Execute a fixed number of decode rounds outside the timer for warmup.
4. Time a fixed number of decode rounds and synchronize CUDA at both edges.

Run from the repository root, for example::

    PYTHONPATH="$PWD" python \
        scripts/base/bench.py --num-seqs 1 8 --output result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence


# Invalid values make OpenMP fail while importing torch/transformers.
for thread_env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    thread_env_value = os.environ.get(thread_env_name)
    try:
        thread_env_valid = thread_env_value is None or int(
            thread_env_value) > 0
    except ValueError:
        thread_env_valid = False
    if not thread_env_valid:
        os.environ[thread_env_name] = "1"


import torch  # noqa: E402

import extend_asyncio as asyncio  # noqa: E402
import split_qllm_models  # noqa: E402
from operators import flags  # noqa: E402
from operators.assemble.decoders.real import DecodersImplement, make_decoders_from_files  # noqa: E402
from operators.container import NonAsyncContainer  # noqa: E402
from operators.core.block.real import LayerBlocksHolder  # noqa: E402
from operators.core.head.real import ModelHead  # noqa: E402
from operators.core.sampler.common import SamplingParams  # noqa: E402
from operators.core.tail.real import ModelTail  # noqa: E402
from operators.core.tokenizer.real import ExtendTokenizer  # noqa: E402
from operators.helpers.model_lib import ModelsLib  # noqa: E402
from scripts.base.infer import CacheAllocator, InferRunner, SeqComplex  # noqa: E402


MODEL_NAME = "teared-Llama-2-7B-GPTQ"
DEFAULT_PROMPT = "Explain why the sky appears blue in three sentences."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark decode throughput of DROFELL's base Llama-2-7B-GPTQ."
        )
    )
    parser.add_argument(
        "--num-seqs",
        type=int,
        nargs="+",
        default=[1, 8],
        help="Batch sizes to benchmark (default: 1 8).",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--warmup-decode-tokens", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=200)
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--decoder-backend",
        choices=("marlin", "exllama_fuse"),
        default="marlin",
        help="Quantized decode GEMM backend (default: marlin).",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture and replay the greedy decode path by exact batch size.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=32*1024,
        help="Fixed max seq len (at model build & graph capture, not actual benchmark seq len).",
    )
    parser.add_argument(
        "--async-backend",
        choices=("py", "chimera", "golang"),
        default="golang",
        help="extend_asyncio implementation (default: golang).",
    )
    parser.add_argument(
        "--busy-wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable busy-waiting in the selected extend_asyncio backend; "
            "use --no-busy-wait to disable it."
        ),
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    if not args.num_seqs or any(value <= 0 for value in args.num_seqs):
        parser.error("--num-seqs values must be positive")
    if len(set(args.num_seqs)) != len(args.num_seqs):
        parser.error("--num-seqs values must be unique")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be non-negative")
    if args.warmup_decode_tokens < 0:
        parser.error("--warmup-decode-tokens must be non-negative")
    if args.decode_tokens <= 0:
        parser.error("--decode-tokens must be positive")
    if args.num_blocks <= 0:
        parser.error("--num-blocks must be positive")
    if args.max_seq_len <= 0:
        parser.error("--graph-max-context-len must be positive")
    if args.cuda_graph and args.decoder_backend != "marlin":
        parser.error(
            "--cuda-graph currently requires --decoder-backend marlin")
    return args


class DecodeBenchmark:
    def __init__(
        self,
        *,
        device: str,
        num_blocks: int,
        decoder_backend: DecodersImplement,
        cuda_graph: bool,
        max_seq_len: int,
        max_seqs_num: int,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("this benchmark requires a CUDA device")
        torch.cuda.set_device(self.device)

        models = ModelsLib(split_qllm_models.lib_path)
        model = models.models_storage[MODEL_NAME]
        model.config.max_seq_len = max_seq_len
        self.model = model
        self.tokenizer = ExtendTokenizer.load_tokenier(
            model.tokenizer_path,
            model.config.tokenizer_special_tokens,
        )
        self.num_blocks = num_blocks

        with torch.device(self.device):
            head = ModelHead(model.config.model_head_config)
            head.load_from_data_dir(model.model_head_path)
            tail = ModelTail(model.config.model_tail_config)
            tail.load_from_data_dir(model.model_tail_path)
            decoders = make_decoders_from_files(
                implement=decoder_backend,
                config=model.config.quant_decoder_config,
                state_files=[
                    model.decoder_path(index)
                    for index in range(model.config.num_decoders)
                ],
            )
            layer_cachers = [
                LayerBlocksHolder.create_contigious_cacher(
                    block_info=model.config.block_info,
                    num_blocks=num_blocks,
                    device=self.device,
                    dtype=torch.float16,
                )
                for _ in range(model.config.num_decoders)
            ]

        self.runner = InferRunner(
            head,
            tail,
            decoders,
            layer_cachers,
            tokenizer=self.tokenizer,
            enable_cuda_graph=cuda_graph,
            max_seqs_num=max_seqs_num,
            max_seq_len=max_seq_len,
        )
        stream = torch.cuda.Stream(device=self.device)
        self.container = NonAsyncContainer(stream)

    def make_seqs(
        self,
        *,
        num_seqs: int,
        prompt: str,
        max_tokens: int,
    ) -> list[SeqComplex]:
        prompt_tokens = self.tokenizer.encode(prompt)
        blocks_per_seq = (
            len(prompt_tokens) + max_tokens + flags.BLOCK_SIZE - 1
        ) // flags.BLOCK_SIZE
        required_blocks = blocks_per_seq * num_seqs
        if required_blocks > self.num_blocks:
            raise ValueError(
                f"workload requires {required_blocks} cache blocks, but "
                f"--num-blocks={self.num_blocks}"
            )

        available_blocks = list(range(self.num_blocks))
        seqs: list[SeqComplex] = []
        for _ in range(num_seqs):
            seq = self.tokenizer.init_seq(
                prompt=prompt,
                max_tokens=max_tokens,
                ignore_eos=True,
                stop_str=None,
            )
            seqs.append(
                SeqComplex(
                    seq=seq,
                    allocator=CacheAllocator(available_blocks.pop),
                    sampling_params=SamplingParams(temperature=0.0),
                    user_model=None,
                )
            )
        return seqs

    @staticmethod
    def append_tokens(seqs: Sequence[SeqComplex], tokens: Sequence[int]) -> None:
        if len(seqs) != len(tokens):
            raise RuntimeError(
                f"decoder returned {len(tokens)} tokens for {len(seqs)} sequences"
            )
        for seq, token in zip(seqs, tokens):
            seq.seq.core.append_token_id(int(token))

    async def run_prefill(self, seqs: list[SeqComplex]) -> list[int]:
        with self.container.stage("prefilling_step"):
            tokens = await self.runner.prefilling_step(self.container, seqs)
            await self.container.notify("prefilling_step_end")
        self.append_tokens(seqs, tokens)
        return tokens

    async def run_decode_rounds(
        self,
        seqs: list[SeqComplex],
        next_tokens: list[int],
        rounds: int,
    ) -> list[int]:
        for _ in range(rounds):
            with self.container.stage("generating_step"):
                next_tokens = await self.runner.generating_step(
                    self.container,
                    seqs,
                    next_tokens,
                )
                await self.container.notify("generating_step_end")
            self.append_tokens(seqs, next_tokens)
        return next_tokens

    async def run_complete_request(
        self,
        *,
        num_seqs: int,
        prompt: str,
        output_tokens: int,
    ) -> None:
        seqs = self.make_seqs(
            num_seqs=num_seqs,
            prompt=prompt,
            max_tokens=output_tokens,
        )
        with self.container.mode("inference"), torch.inference_mode():
            next_tokens = await self.run_prefill(seqs)
            await self.run_decode_rounds(
                seqs,
                next_tokens,
                output_tokens - 1,
            )

    async def benchmark_case(
        self,
        *,
        num_seqs: int,
        prompt: str,
        runs: int,
        warmup_requests: int,
        warmup_decode_tokens: int,
        decode_tokens: int,
    ) -> dict[str, Any]:
        request_output_tokens = 1 + warmup_decode_tokens + decode_tokens
        for _ in range(warmup_requests):
            await self.run_complete_request(
                num_seqs=num_seqs,
                prompt=prompt,
                output_tokens=request_output_tokens,
            )
            torch.cuda.synchronize(self.device)

        raw_runs: list[dict[str, Any]] = []
        for run_index in range(runs):
            seqs = self.make_seqs(
                num_seqs=num_seqs,
                prompt=prompt,
                max_tokens=request_output_tokens,
            )
            with self.container.mode("inference"), torch.inference_mode():
                next_tokens = await self.run_prefill(seqs)
                next_tokens = await self.run_decode_rounds(
                    seqs,
                    next_tokens,
                    warmup_decode_tokens,
                )

                torch.cuda.synchronize(self.device)
                started = time.perf_counter()
                await self.run_decode_rounds(
                    seqs,
                    next_tokens,
                    decode_tokens,
                )
                torch.cuda.synchronize(self.device)
                seconds = time.perf_counter() - started

            measured_tokens = decode_tokens * num_seqs
            token_ids = [seq.seq.core.output_token_ids for seq in seqs]
            raw_runs.append(
                {
                    "run_index": run_index,
                    "decode": {
                        "tokens": measured_tokens,
                        "rounds": decode_tokens,
                        "seconds": round(seconds, 6),
                        "tok_s": round(measured_tokens / seconds, 4),
                        "avg_ms_per_token": round(
                            seconds * 1000 / measured_tokens, 4
                        ),
                        "avg_ms_per_logical_round": round(
                            seconds * 1000 / decode_tokens, 4
                        ),
                    },
                    "output_token_ids_sha256": [
                        hashlib.sha256(
                            json.dumps(ids, separators=(
                                ",", ":")).encode("ascii")
                        ).hexdigest()
                        for ids in token_ids
                    ],
                }
            )

        rates = [run["decode"]["tok_s"] for run in raw_runs]
        round_times = [
            run["decode"]["avg_ms_per_logical_round"] for run in raw_runs
        ]
        return {
            "num_seqs": num_seqs,
            "prompt_tokens_per_seq": len(self.tokenizer.encode(prompt)),
            "runs": raw_runs,
            "summary": {
                "decode_tok_s_median": round(statistics.median(rates), 4),
                "decode_tok_s_min": round(min(rates), 4),
                "decode_tok_s_max": round(max(rates), 4),
                "decode_round_ms_median": round(
                    statistics.median(round_times), 4
                ),
            },
        }


async def amain(args: argparse.Namespace) -> dict[str, Any]:
    benchmark = DecodeBenchmark(
        device=args.device,
        num_blocks=args.num_blocks,
        decoder_backend=args.decoder_backend,
        cuda_graph=args.cuda_graph,
        max_seq_len=args.max_seq_len,
        max_seqs_num=max(args.num_seqs),
    )
    cases: dict[str, Any] = {}
    for num_seqs in args.num_seqs:
        print(f"[info] benchmarking num_seqs={num_seqs}", file=sys.stderr)
        cases[f"seqs_{num_seqs}"] = await benchmark.benchmark_case(
            num_seqs=num_seqs,
            prompt=args.prompt,
            runs=args.runs,
            warmup_requests=args.warmup_requests,
            warmup_decode_tokens=args.warmup_decode_tokens,
            decode_tokens=args.decode_tokens,
        )

    gpu_index = benchmark.device.index
    if gpu_index is None:
        gpu_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(gpu_index)
    return {
        "schema_version": 1,
        "model": MODEL_NAME,
        "execution": "graph" if args.cuda_graph else "eager",
        "decoder_backend": args.decoder_backend,
        "graph": benchmark.runner.graph_stats(),
        "measurement": {
            "runs": args.runs,
            "warmup_requests": args.warmup_requests,
            "warmup_decode_tokens_per_seq": args.warmup_decode_tokens,
            "decode_tokens_per_seq": args.decode_tokens,
        },
        "execution_runtime": {
            "async_backend": args.async_backend,
            "busy_wait": args.busy_wait,
        },
        "gpu": {
            "device": str(benchmark.device),
            "name": props.name,
            "total_memory_bytes": props.total_memory,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    asyncio.config_impl(args.async_backend, args.busy_wait)
    report = asyncio.run(amain(args))
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"[info] results written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
