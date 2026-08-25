from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=REPO_ROOT, check=True,
        stdout=subprocess.PIPE, text=True,
    ).stdout.strip()


def row(result: Any, *, sweep: str) -> dict[str, Any]:
    transfer_wait_ns = sum(
        int(backend["transfer_contention_ns"])
        for backend in result.backends
    )
    device_wait_ns = sum(
        int(backend["device_contention_ns"])
        for backend in result.backends
    )
    backend_operations = sum(
        int(backend["operations"]) for backend in result.backends
    )
    return {
        "sweep": sweep,
        "segments": int(result.topology["segments"]),
        "redundancy": int(result.topology["redundancy"]),
        "logical_nodes": int(result.topology["logical_nodes"]),
        "clients": int(result.config["clients"]),
        "tokens_per_client": int(result.config["tokens_per_client"]),
        "metric_warmup_tokens": int(
            result.config["metric_warmup_tokens"]
        ),
        "steady_tokens_per_second": float(
            result.timing["steady_state_tokens_per_second"]
        ),
        "ttft_ms": result.timing["ttft_ms"],
        "itl_ms": result.timing["itl_ms"],
        "packets": int(result.fabric["packets"]),
        "wire_bytes": int(result.fabric["bytes_total"]),
        "fabric_queue_ms": float(result.fabric["queue_delay_ms_total"]),
        "max_fabric_queue_ms": float(result.fabric["queue_delay_ms_max"]),
        "host_cpu_queue_ms": result.fabric["host_cpu_queue_ms"],
        "device_queue_ms_total": device_wait_ns / 1e6,
        "transfer_queue_ms_total": transfer_wait_ns / 1e6,
        "backend_operations": backend_operations,
        "wall_seconds": float(result.wall_seconds),
        "cost_extrapolations": result.cost_model["extrapolations"],
        "notes": result.notes,
        # Preserve schema-v2 per-client timestamps/intervals and all resource
        # counters so the compact columns above can be independently rebuilt.
        "raw_result": dataclasses.asdict(result),
    }


def deterministic_projection(result: Any) -> dict[str, Any]:
    """Fields that must agree across deterministic reruns."""
    return {
        "topology": result.topology,
        "tokens_produced": result.tokens_produced,
        "virtual_seconds": result.virtual_seconds,
        "aggregate_tokens_per_second": result.aggregate_tokens_per_second,
        "per_client_inter_token_ms": result.per_client_inter_token_ms,
        "timing": result.timing,
        "fabric": result.fabric,
        "backends": result.backends,
        "events": result.events,
        "notes": result.notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--loaded-clients", type=int, default=16)
    parser.add_argument("--max-logical-nodes", type=int, default=256)
    parser.add_argument("--loaded-max-redundancy", type=int, default=8)
    parser.add_argument(
        "--redundancies", nargs="+", type=int, default=(1, 2, 3),
        help=(
            "active replicas per segment; 1/2/3 is the representative range, "
            "larger values are mechanism stress tests"
        ),
    )
    parser.add_argument(
        "--evidence-id",
        default="autodl-l2-l4-multipath-r1-r3-v1",
    )
    args = parser.parse_args()
    if args.tokens <= args.warmup_tokens:
        raise SystemExit("tokens must exceed warm-up tokens")

    profile = json.loads(args.profile.read_text())
    dataset = json.loads(args.dataset.read_text())
    segments = sorted(int(value) for value in dataset["included_segments"])
    if segments != [2, 4]:
        raise SystemExit(f"this experiment requires L2/L4 data, got {segments}")
    if profile["hardware_id"] != dataset["hardware_id"]:
        raise SystemExit("profile and dataset hardware IDs differ")
    if profile["valid_layer_range"] != [8, 16]:
        raise SystemExit(
            "profile is not scoped to the measured L2/L4 shard sizes"
        )

    from simulate.drosim.run import SimConfig, run

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    redundancies_requested = sorted(set(args.redundancies))
    if not redundancies_requested or redundancies_requested[0] <= 0:
        raise SystemExit("redundancies must be positive")
    for segment_count in segments:
        redundancies = [
            value for value in redundancies_requested
            if segment_count * value <= args.max_logical_nodes
        ]
        for redundancy in redundancies:
            print(
                f"single-client L{segment_count} r{redundancy} "
                f"N={segment_count * redundancy}",
                flush=True,
            )
            result = run(SimConfig(
                segments=segment_count,
                redundancy=redundancy,
                clients=1,
                tokens=args.tokens,
                metric_warmup_tokens=args.warmup_tokens,
                cost_profile=args.profile,
                log_level="error",
            ))
            rows.append(row(result, sweep="active_holders_single_client"))

        for redundancy in redundancies:
            if redundancy > args.loaded_max_redundancy:
                continue
            print(
                f"loaded L{segment_count} r{redundancy} "
                f"N={segment_count * redundancy} "
                f"C={args.loaded_clients}",
                flush=True,
            )
            result = run(SimConfig(
                segments=segment_count,
                redundancy=redundancy,
                clients=args.loaded_clients,
                tokens=args.tokens,
                metric_warmup_tokens=args.warmup_tokens,
                cost_profile=args.profile,
                log_level="error",
            ))
            rows.append(row(result, sweep="active_holders_loaded"))

    # G0 reproducibility check: rerun the smallest measured-scope cell and
    # compare every deterministic field, excluding real wall time.
    repeat_config = SimConfig(
        segments=2,
        redundancy=1,
        clients=1,
        tokens=args.tokens,
        metric_warmup_tokens=args.warmup_tokens,
        cost_profile=args.profile,
        log_level="error",
    )
    first = run(repeat_config)
    second = run(repeat_config)
    reproducible = deterministic_projection(first) == deterministic_projection(
        second
    )

    profile_gates = profile["calibration"]["gates"]
    blockers = [
        name for name, passed in profile_gates.items() if not passed
    ]
    blockers.extend([
        "logical-holder counts above four are topology extrapolations",
        "all logical holders are placed on one modelled host",
        "CPU service is per-process; finite host-wide CPU capacity is not modelled",
        "no cross-host or geographic-network validation is available",
        "parameter uncertainty cannot be propagated from one hardware run per cell",
    ])
    if max(redundancies_requested) > 3:
        blockers.append(
            "redundancy above three is a mechanism stress test, not the "
            "expected deployment range"
        )
    if not reproducible:
        blockers.append("deterministic rerun mismatch")

    report = {
        "schema_version": 2,
        "evidence_id": args.evidence_id,
        "scope": {
            "included_measured_segments": segments,
            "excluded_segments": [8],
            "excluded_reason": (
                "L8 receiver/GC/queueing anomaly remains causally unresolved"
            ),
            "placement": "dedicated",
            "experiment_class": (
                "representative_multipath_range"
                if max(redundancies_requested) <= 3
                else "mechanism_stress_test"
            ),
            "redundancy_semantics": (
                "all replicas are active; every holder in one stage sends to "
                "every holder in the next stage"
            ),
            "representative_redundancy_max": 3,
            "placement_interpretation": (
                "control-plane isolation with one modelled device per logical "
                "holder; not a physical single-host capacity claim"
            ),
            "network": "same-host loopback only",
            "host_cpu": (
                "per-process CPU domains; finite host-wide core capacity is "
                "not modelled"
            ),
            "randomness": "none; the clean simulation is deterministic",
        },
        "configuration": {
            "tokens_per_client": args.tokens,
            "metric_warmup_tokens": args.warmup_tokens,
            "loaded_clients": args.loaded_clients,
            "max_logical_nodes": args.max_logical_nodes,
            "loaded_max_redundancy": args.loaded_max_redundancy,
            "redundancies": redundancies_requested,
        },
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_dirty": bool(git_output("status", "--porcelain")),
            "profile": str(args.profile),
            "profile_sha256": sha256(args.profile),
            "dataset": str(args.dataset),
            "dataset_sha256": sha256(args.dataset),
            "hardware_id": profile["hardware_id"],
        },
        "reproducibility": {
            "deterministic_rerun_passed": reproducible,
            "cell": "L2-r1-c1",
        },
        "rows": rows,
        "elapsed_wall_seconds": time.monotonic() - started,
        "validity": {
            "paper_eligible": False,
            "profile_paper_eligible": bool(
                profile["calibration"]["paper_eligible"]
            ),
            "blockers": blockers,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        f"wrote {args.out}: {len(rows)} cells, "
        f"deterministic={str(reproducible).lower()}, "
        "paper_eligible=false",
    )


if __name__ == "__main__":
    main()
