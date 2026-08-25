"""Benchmark one GPTQ projection with inference and PEFT backward backends."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, cast

import torch
from safetensors import safe_open  # type: ignore

from cuda_extends.exllama import CacheV2, TuningParams
from cuda_extends.marlin import MarlinDenseCache, MarlinTuningParams
from operators import flags
from operators.core.linear.real import Linear
from operators.core.linear.real_auto_gptq import (
    AutoGPTQQuantLinearsType,
    QuantLinearHolder as AutoGPTQLinearHolder,
)
from operators.core.linear.real_marlin import (
    MarlinQuantLinear,
    MarlinQuantLinearMaker,
)
from operators.core.linear.real_quant import QuantLinearMaker
from operators.helpers.configs import LlamaConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = REPO_ROOT / "scripts" / "converts" / "qlinear_data"
Method = Literal[
    "our_cache",
    "our_no_cache",
    "marlin",
    "marlin_dense_no_cache",
    "marlin_dense_cache",
    "exllama_v1",
    "exllama_v2",
    "triton",
    "cuda",
    "cuda_old",
]
METHODS: tuple[Method, ...] = (
    "our_cache",
    "our_no_cache",
    "marlin",
    "marlin_dense_no_cache",
    "marlin_dense_cache",
    "exllama_v1",
    "exllama_v2",
    "triton",
    "cuda",
    "cuda_old",
)
PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--projection", choices=PROJECTIONS, default="q_proj")
    parser.add_argument(
        "--seq-lens", nargs="+", type=int, default=[1, 16, 128, 512, 2048]
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if any(seq_len <= 0 for seq_len in args.seq_lens):
        parser.error("--seq-lens values must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats positive")
    return args


def projection_shape(config: LlamaConfig, projection: str) -> tuple[int, int]:
    hidden = config.hidden_size
    q_size = config.num_query_heads * config.head_dim
    kv_size = config.num_key_value_heads * config.head_dim
    intermediate = config.decoder_config.intermediate_size
    return {
        "q_proj": (hidden, q_size),
        "k_proj": (hidden, kv_size),
        "v_proj": (hidden, kv_size),
        "o_proj": (q_size, hidden),
        "gate_proj": (hidden, intermediate),
        "up_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }[projection]


def load_weight(path: Path, module: torch.nn.Module) -> None:
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as weights:
        for key in weights.keys():
            value = weights.get_tensor(key)
            if key.endswith("bias"):
                if bool(torch.any(value != 0).item()):
                    raise ValueError(f"non-zero bias is unsupported: {path}:{key}")
                continue
            state_dict[key] = value
    module.load_state_dict(state_dict=state_dict)


def make_qlinear(
    config: LlamaConfig,
    input_dir: Path,
    projection: str,
    method: Method,
    device: torch.device,
    max_seq_len: int,
) -> tuple[Linear, int, int, list[Any], bool]:
    in_features, out_features = projection_shape(config, projection)
    persist: list[Any] = []
    supports_backward = True
    with torch.device(device):
        if method in ("our_cache", "our_no_cache"):
            cache = CacheV2(device)
            maker = QuantLinearMaker(
                quantization_group_size=config.quantization_config.group_size,
                use_act_order=config.quantization_config.desc_act,
            )
            module = maker.make_linear(in_features, out_features, bias=False)
            load_weight(input_dir / f"{projection}.safetensors", module)
            flags.REUSE_RECONSTRUCT_CACHE_WHEN_GRAD = method == "our_cache"
            maker.post_init(TuningParams(), cache)
            cache.init_cache()
            persist.extend((maker, cache))
        elif method.startswith("marlin"):
            dense_forward = method != "marlin"
            cache = MarlinDenseCache(device)
            tuning_params = MarlinTuningParams(
                reconstruct_threshold=1 if dense_forward else None,
                grad_reconstruct_threshold=1 if dense_forward else None,
                reuse_reconstruct_cache_when_grad=(
                    method == "marlin_dense_cache"
                ),
            )
            maker = MarlinQuantLinearMaker(
                quantization_group_size=config.quantization_config.group_size,
                use_act_order=config.quantization_config.desc_act,
            )
            module = maker.make_linear(in_features, out_features, bias=False)
            load_weight(input_dir / f"{projection}.safetensors", module)
            maker.post_init(tuning_params, cache)
            cache.init_cache()
            persist.extend((maker, cache))
        else:
            auto_method = cast(AutoGPTQQuantLinearsType, method)
            maker = AutoGPTQLinearHolder(
                group_size=config.quantization_config.group_size,
                bits=config.quantization_config.bits,
                desc_act=config.quantization_config.desc_act,
                quant_linear_type=auto_method,
            )
            module = maker.make_linear(in_features, out_features, bias=False)
            load_weight(input_dir / f"{projection}.safetensors", module)
            persist.extend(maker.post_init(max_input_length=max_seq_len))
            maker.warmup(seqlen=max_seq_len, warmup_history={})
            persist.append(maker)
            supports_backward = method not in ("exllama_v1", "exllama_v2")
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return module, in_features, out_features, persist, supports_backward


def elapsed_ms(operation: Callable[[], Any]) -> tuple[float, Any]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), result


def forward_samples(
    module: Linear,
    seq_len: int,
    in_features: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    generator = torch.Generator(device=device)
    generator.manual_seed(0x4D41524C + seq_len)
    x = torch.randn(
        (seq_len, in_features),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    module.eval()
    values: list[float] = []
    with torch.inference_mode():
        for iteration in range(warmup + repeats):
            duration, output = elapsed_ms(lambda: module(x))
            if iteration >= warmup:
                values.append(duration)
            del output
    return values


def peft_samples(
    module: Linear,
    seq_len: int,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[list[float], list[float], float]:
    generator = torch.Generator(device=device)
    generator.manual_seed(0x50454654 + seq_len)
    x = torch.randn(
        (seq_len, in_features),
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    grad = torch.randn(
        (seq_len, out_features),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    module.train()
    forward_values: list[float] = []
    backward_values: list[float] = []
    checksum = 0.0
    for iteration in range(warmup + repeats):
        x.grad = None
        forward_ms, output = elapsed_ms(lambda: module(x))
        backward_ms, _ = elapsed_ms(lambda: output.backward(grad))
        if x.grad is None or not bool(torch.all(torch.isfinite(x.grad)).item()):
            raise RuntimeError("PEFT backward produced a missing or non-finite input gradient")
        if iteration >= warmup:
            forward_values.append(forward_ms)
            backward_values.append(backward_ms)
        checksum = float(x.grad.float().sum().item())
        del output
    return forward_values, backward_values, checksum


def marlin_component_samples(
    module: MarlinQuantLinear,
    seq_len: int,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[list[float], list[float]]:
    """Measure reconstruction and pre-reconstructed dense GEMM separately."""

    dense_weight = torch.empty(
        (in_features, out_features), dtype=torch.float16, device=device
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(0x434F4D50 + seq_len)
    x = torch.randn(
        (seq_len, in_features),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    reconstruction: list[float] = []
    dense_forward: list[float] = []
    module.eval()
    with torch.inference_mode():
        for iteration in range(warmup + repeats):
            duration, _ = elapsed_ms(
                lambda: module.reconstruct_weight(dense_weight)
            )
            if iteration >= warmup:
                reconstruction.append(duration)
        # Keep a valid, fully reconstructed weight fixed while measuring only
        # the GEMM portion of the alternative forward path.
        module.reconstruct_weight(dense_weight)
        for iteration in range(warmup + repeats):
            duration, output = elapsed_ms(
                lambda: module.dense_forward(x, dense_weight)
            )
            if iteration >= warmup:
                dense_forward.append(duration)
            del output
    return reconstruction, dense_forward


def memory_sample(
    module: Linear,
    seq_len: int,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, int]:
    gc.collect()
    torch.cuda.empty_cache()
    x = torch.randn(
        (seq_len, in_features), dtype=dtype, device=device, requires_grad=True
    )
    grad = torch.randn((seq_len, out_features), dtype=dtype, device=device)
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    output = module(x)
    torch.cuda.synchronize(device)
    after_forward = torch.cuda.memory_allocated(device)
    peak_forward = torch.cuda.max_memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    output.backward(grad)
    torch.cuda.synchronize(device)
    after_backward = torch.cuda.memory_allocated(device)
    peak_backward = torch.cuda.max_memory_allocated(device)
    if x.grad is None:
        raise RuntimeError("memory sample did not produce an input gradient")

    record = {
        "allocated_before_bytes": int(before),
        "retained_after_forward_bytes": int(after_forward - before),
        "forward_peak_delta_bytes": int(peak_forward - before),
        "backward_peak_delta_from_forward_bytes": int(
            peak_backward - after_forward
        ),
        "total_peak_delta_bytes": int(max(peak_forward, peak_backward) - before),
        "retained_after_backward_bytes": int(after_backward - before),
    }
    del output, grad, x
    gc.collect()
    torch.cuda.empty_cache()
    return record


def stats(values: Sequence[float]) -> dict[str, Any]:
    return {
        "samples_ms": list(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    with (input_dir / "config.json").open("r", encoding="utf-8") as stream:
        config = LlamaConfig(**json.load(stream))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    dtype = getattr(torch, args.dtype)
    module, in_features, out_features, persist, supports_backward = make_qlinear(
        config=config,
        input_dir=input_dir,
        projection=args.projection,
        method=args.method,
        device=device,
        max_seq_len=max(args.seq_lens),
    )

    records: list[dict[str, Any]] = []
    for seq_len in args.seq_lens:
        print(f"[{args.method}] seq_len={seq_len}", flush=True)
        forward = forward_samples(
            module, seq_len, in_features, dtype, device, args.warmup, args.repeats
        )
        record: dict[str, Any] = {
            "seq_len": seq_len,
            "forward": stats(forward),
        }
        if supports_backward:
            peft_forward, backward, checksum = peft_samples(
                module,
                seq_len,
                in_features,
                out_features,
                dtype,
                device,
                args.warmup,
                args.repeats,
            )
            record["peft_forward"] = stats(peft_forward)
            record["backward"] = stats(backward)
            record["peft_total"] = stats(
                [fwd + bwd for fwd, bwd in zip(peft_forward, backward)]
            )
            record["input_grad_checksum"] = checksum
            record["memory"] = memory_sample(
                module,
                seq_len,
                in_features,
                out_features,
                dtype,
                device,
            )
        # Run component-only diagnostics after the comparable workload so its
        # extra random input does not change PEFT checksums across backends.
        if isinstance(module, MarlinQuantLinear):
            reconstruction, dense_forward = marlin_component_samples(
                module,
                seq_len,
                in_features,
                out_features,
                dtype,
                device,
                args.warmup,
                args.repeats,
            )
            record["reconstruction"] = stats(reconstruction)
            record["dense_forward_only"] = stats(dense_forward)
        records.append(record)

    props = torch.cuda.get_device_properties(device)
    result = {
        "schema_version": 2,
        "created_at_unix": time.time(),
        "method": args.method,
        "projection": args.projection,
        "input_dir": str(input_dir),
        "dtype": args.dtype,
        "in_features": in_features,
        "out_features": out_features,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "supports_backward": supports_backward,
        "reuse_reconstruct_cache_when_grad": (
            args.method in ("our_cache", "marlin_dense_cache")
            if args.method.startswith(("our_", "marlin"))
            else None
        ),
        "marlin_forward_policy": (
            "reconstruct_dense"
            if args.method.startswith("marlin_dense")
            else "direct_packed"
            if args.method == "marlin"
            else None
        ),
        "marlin_reconstruct_threshold": (
            module.tuning_params.reconstruct_threshold
            if isinstance(module, MarlinQuantLinear)
            else None
        ),
        "marlin_grad_reconstruct_threshold": (
            module.tuning_params.grad_reconstruct_threshold
            if isinstance(module, MarlinQuantLinear)
            else None
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": props.name,
            "gpu_total_memory_bytes": props.total_memory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")
    _ = persist


if __name__ == "__main__":
    main()
