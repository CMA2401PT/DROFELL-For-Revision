"""Generate a placement-independent online benchmark network specification."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.online.benchmark.network_spec import (
    DealerSpec,
    MemorySpec,
    NetworkSpec,
    model_decoder_count,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def dealer_assignment(value: str) -> tuple[str, tuple[int, ...]]:
    name, separator, group_text = value.partition("=")
    if not separator or not name or not group_text:
        raise argparse.ArgumentTypeError(
            "dealer assignment must use NAME=GROUP_ID[,GROUP_ID...]"
        )
    try:
        group_ids = tuple(int(item) for item in group_text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid dealer group list in {value!r}"
        ) from error
    if any(group_id < 0 for group_id in group_ids):
        raise argparse.ArgumentTypeError("dealer group IDs must be non-negative")
    return name, group_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a logical online LLM network JSON file"
    )
    parser.add_argument("--name", required=True, help="Stable logical network name.")
    parser.add_argument("--model", required=True, help="Model name in ModelsLib.")
    parser.add_argument(
        "--decoder-group-sizes",
        nargs="+",
        type=positive_int,
        required=True,
        metavar="LAYERS",
        help="Exact ordered decoder group sizes; their sum must cover the model.",
    )
    parser.add_argument(
        "--dealer",
        action="append",
        type=dealer_assignment,
        required=True,
        metavar="NAME=GROUP_IDS",
        help="Repeatable dealer-to-group assignment, e.g. dealer-a=0,1.",
    )
    parser.add_argument(
        "--decoder-backend",
        choices=("marlin", "exllama", "exllama_fuse"),
        default="marlin",
    )
    parser.add_argument("--max-seq-len", type=positive_int, required=True)
    parser.add_argument("--max-concurrency", type=positive_int, required=True)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--time-trace", action="store_true")
    parser.add_argument("--time-trace-sync-cuda", action="store_true")
    parser.add_argument(
        "--failure-injection",
        action="store_true",
        help="Launch all dealers with benchmark-controlled failure injection support.",
    )
    parser.add_argument(
        "--memory-policy", choices=("static", "percentage"), default="static"
    )
    parser.add_argument("--major-reserved-mib", type=non_negative_int, default=8192)
    parser.add_argument("--major-percentage", type=float, default=0.8)
    parser.add_argument(
        "--unmanaged-reserved-mib", type=non_negative_int, default=32
    )
    parser.add_argument("--minor-min-mib", type=non_negative_int, default=0)
    parser.add_argument("--minor-max-mib", type=non_negative_int, default=4096)
    parser.add_argument("--operate-reserved-mib", type=non_negative_int, default=1024)
    parser.add_argument("--graph-budget-mib", type=non_negative_int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.time_trace_sync_cuda and not args.time_trace:
        parser.error("--time-trace-sync-cuda requires --time-trace")
    if args.cuda_graph and args.decoder_backend != "marlin":
        parser.error("--cuda-graph requires --decoder-backend marlin")
    if args.cuda_graph and args.graph_budget_mib <= 0:
        parser.error("--cuda-graph requires a positive --graph-budget-mib")
    if not 0.0 < args.major_percentage <= 1.0:
        parser.error("--major-percentage must be in (0, 1]")
    if args.minor_min_mib > args.minor_max_mib:
        parser.error("--minor-min-mib cannot exceed --minor-max-mib")
    return args


def main() -> None:
    args = parse_args()
    num_decoders = model_decoder_count(args.model)
    memory = MemorySpec(
        policy=args.memory_policy,
        major_reserved_mib=args.major_reserved_mib,
        major_percentage=args.major_percentage,
        unmanaged_reserved_mib=args.unmanaged_reserved_mib,
        minor_min_mib=args.minor_min_mib,
        minor_max_mib=args.minor_max_mib,
        operate_reserved_mib=args.operate_reserved_mib,
        graph_budget_mib=args.graph_budget_mib,
    )
    dealers = tuple(
        DealerSpec(
            name=name,
            group_ids=group_ids,
            decoder_backend=args.decoder_backend,
            max_seq_len=args.max_seq_len,
            max_concurrency=args.max_concurrency,
            enable_cuda_graph=args.cuda_graph,
            time_trace=args.time_trace,
            time_trace_sync_cuda=args.time_trace_sync_cuda,
            failure_injection=args.failure_injection,
            memory=memory,
        )
        for name, group_ids in args.dealer
    )
    spec = NetworkSpec(
        name=args.name,
        model_name=args.model,
        num_decoders=num_decoders,
        decoder_group_sizes=tuple(args.decoder_group_sizes),
        dealers=dealers,
    )
    spec.write(args.output)
    print(f"wrote {args.output}")
    print(f"network: {spec.name}")
    print(f"spec sha256: {spec.spec_hash()}")
    print(f"decoder groups: {list(spec.decoder_group_sizes)}")
    for dealer in spec.dealers:
        print(f"{dealer.name}: groups={list(dealer.group_ids)}")


if __name__ == "__main__":
    main()
