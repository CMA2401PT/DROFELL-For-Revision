"""Standalone vLLM Llama-2-7B-GPTQ decode-only benchmark.

The script uses vLLM's native engine API.  It submits a request batch, advances
the engine through prefill, performs decode warmup steps, and only then starts
the timer.  It intentionally does not share a benchmark implementation with
DROFELL's base/offline/online layers.
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
from typing import Any, cast


for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _value = os.environ.get(_thread_variable)
    try:
        _valid = _value is None or int(_value) > 0
    except ValueError:
        _valid = False
    if not _valid:
        os.environ[_thread_variable] = "1"

# Keep engine stepping synchronous so the prefill/decode timing boundary is
# observable in this process.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


DEFAULT_MODEL = "/home/dai/projects/LLM/dflash_related/models/Llama-2-7B-GPTQ"
DEFAULT_PROMPT = "Explain why the sky appears blue in three sentences."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM Llama-2-7B-GPTQ decode-only benchmark"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-seqs", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--warmup-decode-tokens", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable vLLM CUDA graph capture; otherwise enforce eager mode.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.num_seqs or any(value <= 0 for value in args.num_seqs):
        parser.error("--num-seqs must contain positive values")
    if len(set(args.num_seqs)) != len(args.num_seqs):
        parser.error("--num-seqs values must be unique")
    if args.runs <= 0 or args.decode_tokens <= 0:
        parser.error("--runs and --decode-tokens must be positive")
    if args.warmup_requests < 0 or args.warmup_decode_tokens < 0:
        parser.error("warmup counts must be non-negative")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    return args


class VllmDecodeBenchmark:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        import vllm
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams, TokensPrompt
        from vllm.sampling_params import RequestOutputKind

        self.args = args
        self.torch = torch
        self.vllm = vllm
        self.tokens_prompt_type = TokensPrompt
        self.request_output_kind = RequestOutputKind
        self.sampling_params_type = SamplingParams
        self.request_counter = 0
        self.prompt_ids = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
        ).encode(args.prompt)
        request_tokens = (
            1 + args.warmup_decode_tokens + args.decode_tokens
        )
        if len(self.prompt_ids) + request_tokens > args.max_model_len:
            raise ValueError("prompt plus generated tokens exceeds max model length")

        capture_sizes = sorted(set(args.num_seqs))
        llm_options = dict(
            model=args.model,
            quantization="gptq_marlin",
            dtype="float16",
            trust_remote_code=True,
            enforce_eager=not args.cuda_graph,
            max_model_len=args.max_model_len,
            max_num_seqs=max(args.num_seqs),
            gpu_memory_utilization=args.gpu_memory_utilization,
            disable_log_stats=True,
            async_scheduling=False,
        )
        if args.cuda_graph:
            llm_options["compilation_config"] = {
                "cudagraph_capture_sizes": capture_sizes,
            }
        self.llm = LLM(**llm_options)
        self.engine = self.llm.llm_engine
        compilation = self.engine.vllm_config.compilation_config
        self.graph_metadata = {
            "enforce_eager": not args.cuda_graph,
            "cudagraph_mode": str(compilation.cudagraph_mode),
            "cudagraph_capture_sizes": list(
                compilation.cudagraph_capture_sizes or []
            ),
            "max_cudagraph_capture_size": (
                compilation.max_cudagraph_capture_size
            ),
        }

    def _add_requests(self, prefix: str, num_seqs: int) -> dict[str, list[int]]:
        request_tokens = (
            1
            + self.args.warmup_decode_tokens
            + self.args.decode_tokens
        )
        sampling_params = self.sampling_params_type(
            temperature=0.0,
            max_tokens=request_tokens,
            ignore_eos=True,
            detokenize=False,
            output_kind=self.request_output_kind.DELTA,
        )
        outputs = {}
        for index in range(num_seqs):
            request_id = f"{prefix}-{index}"
            outputs[request_id] = []
            self.engine.add_request(
                request_id,
                self.tokens_prompt_type(
                    prompt_token_ids=self.prompt_ids.copy()
                ),
                sampling_params,
            )
        return outputs

    def _step(self, output_tokens: dict[str, list[int]]) -> int:
        visible_tokens = 0
        for request_output in self.engine.step():
            request_id = str(request_output.request_id)
            for completion in cast(Any, request_output.outputs):
                token_ids = [int(token) for token in completion.token_ids]
                visible_tokens += len(token_ids)
                output_tokens[request_id].extend(token_ids)
        return visible_tokens

    def _run_until_tokens(
        self,
        output_tokens: dict[str, list[int]],
        target_tokens: int,
    ) -> int:
        generated = 0
        while generated < target_tokens and self.engine.has_unfinished_requests():
            generated += self._step(output_tokens)
        if generated != target_tokens:
            raise RuntimeError(
                "vLLM engine step crossed the requested decode boundary: "
                f"target={target_tokens}, actual={generated}"
            )
        return generated

    def _run_prefill(self, output_tokens: dict[str, list[int]]) -> int:
        steps = 0
        while any(not tokens for tokens in output_tokens.values()):
            self._step(output_tokens)
            steps += 1
        if any(len(tokens) != 1 for tokens in output_tokens.values()):
            raise RuntimeError(
                "vLLM produced more than one visible token at the prefill boundary"
            )
        return steps

    def run_request(self, num_seqs: int, measured: bool) -> dict[str, Any]:
        prefix = f"request-{self.request_counter}"
        self.request_counter += 1
        output_tokens = self._add_requests(prefix, num_seqs)
        prefill_steps = self._run_prefill(output_tokens)
        self._run_until_tokens(
            output_tokens,
            self.args.warmup_decode_tokens * num_seqs,
        )
        if not measured:
            self._run_until_tokens(
                output_tokens,
                self.args.decode_tokens * num_seqs,
            )
            return {}

        self.torch.cuda.synchronize()
        started = time.perf_counter()
        measured_tokens = self._run_until_tokens(
            output_tokens,
            self.args.decode_tokens * num_seqs,
        )
        self.torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if self.engine.has_unfinished_requests():
            while self.engine.has_unfinished_requests():
                self._step(output_tokens)
        expected_tokens = (
            1
            + self.args.warmup_decode_tokens
            + self.args.decode_tokens
        )
        if any(len(tokens) != expected_tokens for tokens in output_tokens.values()):
            raise RuntimeError("vLLM request produced an unexpected token count")
        return {
            "prefill_engine_steps": prefill_steps,
            "tokens": measured_tokens,
            "logical_rounds": self.args.decode_tokens,
            "seconds": seconds,
            "tok_s": measured_tokens / seconds,
            "avg_ms_per_logical_round": (
                seconds * 1000 / self.args.decode_tokens
            ),
            "output_token_ids_sha256": [
                hashlib.sha256(
                    json.dumps(tokens, separators=(",", ":")).encode("ascii")
                ).hexdigest()
                for tokens in output_tokens.values()
            ],
        }

    def run(self) -> dict[str, Any]:
        cases = []
        for num_seqs in self.args.num_seqs:
            for _ in range(self.args.warmup_requests):
                self.run_request(num_seqs, measured=False)
            runs = []
            for run_index in range(self.args.runs):
                result = self.run_request(num_seqs, measured=True)
                result["run_index"] = run_index
                runs.append(result)
            rates = [run["tok_s"] for run in runs]
            cases.append(
                {
                    "num_seqs": num_seqs,
                    "prompt_tokens_per_seq": len(self.prompt_ids),
                    "runs": runs,
                    "summary": {
                        "decode_tok_s_median": statistics.median(rates),
                        "decode_tok_s_min": min(rates),
                        "decode_tok_s_max": max(rates),
                    },
                }
            )
        props = self.torch.cuda.get_device_properties(
            self.torch.cuda.current_device()
        )
        return {
            "schema_version": 1,
            "engine": "vllm",
            "model": Path(self.args.model).name,
            "quantization": "gptq_marlin",
            "execution": "graph" if self.args.cuda_graph else "eager",
            "measurement": {
                "runs": self.args.runs,
                "warmup_requests": self.args.warmup_requests,
                "warmup_decode_tokens_per_seq": (
                    self.args.warmup_decode_tokens
                ),
                "decode_tokens_per_seq": self.args.decode_tokens,
                "max_model_len": self.args.max_model_len,
                "timing_scope": "post_prefill_engine_step_decode",
            },
            "graph": self.graph_metadata,
            "gpu": {
                "name": props.name,
                "total_memory_bytes": props.total_memory,
            },
            "software": {
                "python": sys.version.split()[0],
                "torch": self.torch.__version__,
                "cuda": self.torch.version.cuda,
                "vllm": self.vllm.__version__,
            },
            "cases": cases,
        }

    def close(self) -> None:
        engine_core = getattr(self.engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if callable(shutdown):
            shutdown()
        del self.engine
        del self.llm
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
        self.torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    benchmark = VllmDecodeBenchmark(args)
    try:
        report = benchmark.run()
    finally:
        benchmark.close()
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
