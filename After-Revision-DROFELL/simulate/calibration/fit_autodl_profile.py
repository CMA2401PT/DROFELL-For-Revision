from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATASET = Path(
    "simulate/results/reviewer-grade/autodl8_clean_scale_dataset.json"
)


@dataclass(frozen=True)
class Candidate:
    transfer_fixed_us: float
    invocation_batch1_us: float
    beta_us_per_item: float
    client_turnaround_us: float
    gamma_us_per_layer: float
    host_cpu_fixed_us: float
    head_us: float = 100.0
    tail_us: float = 400.0
    transfer_bandwidth_gbps: float = 12.0

    @property
    def candidate_id(self) -> str:
        return (
            f"xfer{self.transfer_fixed_us:g}-"
            f"inv{self.invocation_batch1_us:g}-"
            f"beta{self.beta_us_per_item:g}-"
            f"turn{self.client_turnaround_us:g}-"
            f"cpu{self.host_cpu_fixed_us:g}"
        )


def candidates(
    target_anchor_itl_us: float,
    target_hop_us: float,
    anchor_segments: int,
) -> list[Candidate]:
    """Constrained family around the fixed-context measured hop term."""
    result: list[Candidate] = []
    # Leave 60 us/hop for loopback wire and residual control work already
    # represented by Fabric and bookkeeping.
    decomposed_hop_us = max(100.0, target_hop_us - 60.0)
    for transfer in (200.0, 300.0, 400.0):
        for host_cpu_fixed in (0.0, 10.0, 20.0, 30.0):
            # One additional segment pays one D2H, one H2D, one invocation and
            # approximately two packets (serialize+deserialize each). The
            # component split remains an ensemble.
            invocation = (
                decomposed_hop_us - 2 * transfer - 4 * host_cpu_fixed
            )
            if invocation < 25:
                continue
            beta_options = tuple(
                value for value in (25.0, 50.0, 100.0, 150.0)
                if value <= invocation
            )
            # The production benchmark immediately calls the next infer_step;
            # keep the independently code-derived client term at zero.
            turnaround = 0.0
            for beta in beta_options:
                # A path with L segments has L decoder invocations, one
                # initial H2D group and L-1 internal D2H+H2D boundaries. The
                # host term approximates two packet directions, each paying
                # serialize+deserialize, per segment. This remains a
                # constrained ensemble rather than a component-level fit.
                fixed_us = (
                    turnaround
                    + (2 * anchor_segments - 1) * transfer
                    + anchor_segments * invocation
                    + 100.0 + 400.0
                    + 4 * anchor_segments * host_cpu_fixed
                )
                gamma = (target_anchor_itl_us - fixed_us) / 32.0
                if gamma <= 0:
                    continue
                result.append(Candidate(
                    transfer_fixed_us=transfer,
                    invocation_batch1_us=invocation,
                    beta_us_per_item=beta,
                    client_turnaround_us=turnaround,
                    gamma_us_per_layer=gamma,
                    host_cpu_fixed_us=host_cpu_fixed,
                ))
    return result


def make_model(candidate: Candidate):
    from simulate.drosim.cost import CostModel, ServiceTerm, TransferTerm

    decoder = ServiceTerm(
        alpha_us=candidate.invocation_batch1_us - candidate.beta_us_per_item,
        beta_us_per_item=candidate.beta_us_per_item,
        gamma_us_per_layer=candidate.gamma_us_per_layer,
        delta_us_per_item_layer=0.0,
    )
    bookkeeping = ServiceTerm(50.0, 0.0)
    return CostModel(
        terms={
            "decoder_generating": decoder,
            "decoder_prefill": decoder,
            "infer_head": ServiceTerm(candidate.head_us, 0.0),
            "infer_tail": ServiceTerm(candidate.tail_us, 0.0),
            "create_infer_binding": bookkeeping,
            "put_user_model": bookkeeping,
            "put_blocks": bookkeeping,
            "get_blocks": bookkeeping,
        },
        transfers={
            "h2d": TransferTerm(
                fixed_us=candidate.transfer_fixed_us,
                bandwidth_gbps=candidate.transfer_bandwidth_gbps,
            ),
            "d2h": TransferTerm(
                fixed_us=candidate.transfer_fixed_us,
                bandwidth_gbps=candidate.transfer_bandwidth_gbps,
            ),
        },
        profile_id=f"autodl8-trial-{candidate.candidate_id}",
        hardware_id="autodl-8gpu-host-2026-08",
        valid_batch_range=(1, 8),
        valid_batch_ranges={"decoder_prefill": (1, 32)},
        valid_layer_range=(4, 32),
        environment={
            "client_turnaround_us": candidate.client_turnaround_us,
            "host_cpu_cost": {
                # One event-loop thread per process domain. Dealer processes are
                # separate domains; persistent Agent shares dealer-g0's domain.
                "cores": 1,
                "serialize_fixed_us": candidate.host_cpu_fixed_us,
                "serialize_bandwidth_gbps": 0.0,
                "deserialize_fixed_us": candidate.host_cpu_fixed_us,
                "deserialize_bandwidth_gbps": 0.0,
            },
        },
        emit_warnings=False,
        provenance=(
            "AutoDL fixed-context raw token timelines; constrained "
            "D2H/H2D/invocation "
            "ensemble, not component-identifiable"
        ),
    )


def simulate(
    row: dict[str, Any],
    model: Any,
    tokens: int,
    metric_warmup_tokens: int | None = None,
) -> dict[str, Any]:
    from simulate.drosim.run import SimConfig, run

    warmup = (
        max(2, min(4, tokens // 3))
        if metric_warmup_tokens is None
        else metric_warmup_tokens
    )
    result = run(SimConfig(
        segments=int(row["segments"]),
        clients=int(row["clients"]),
        tokens=tokens,
        metric_warmup_tokens=warmup,
        deployment=str(row["deployment"]),
        cost_model=model,
        log_level="error",
    ))
    real_tps = float(row["fixed_window_tokens_per_second"])
    sim_tps = float(result.aggregate_tokens_per_second)
    merge_histograms = {
        backend["node"]: backend["merged_batch_histogram"]
        for backend in result.backends
    }
    batch_items = 0
    batch_operations = 0
    for backend in result.backends:
        histogram = backend["merged_batch_histogram"].get(
            "decoder_generating", {}
        )
        for size, count in histogram.items():
            batch_items += int(size) * int(count)
            batch_operations += int(count)
    sim_mean_batch = batch_items / batch_operations
    real_mean_batch = float(
        row["decode_graph"]["observed_mean_merged_batch"]
    )
    return {
        "cell_id": row["cell_id"],
        "split": row["split"],
        "segments": row["segments"],
        "clients": row["clients"],
        "deployment": row["deployment"],
        "real_tokens_per_second": real_tps,
        "sim_tokens_per_second": sim_tps,
        "relative_error_pct": 100.0 * (sim_tps - real_tps) / real_tps,
        "absolute_log_error": abs(math.log(sim_tps / real_tps)),
        "real_mean_merged_batch": real_mean_batch,
        "sim_mean_merged_batch": sim_mean_batch,
        "merged_batch_relative_error_pct": (
            100.0 * (sim_mean_batch - real_mean_batch) / real_mean_batch
        ),
        "merged_batch_absolute_log_error": abs(
            math.log(sim_mean_batch / real_mean_batch)
        ),
        "real_mean_itl_ms": float(row["itl_ms"]["mean"]),
        "sim_mean_itl_ms": float(result.per_client_inter_token_ms),
        "sim_merge_histograms": merge_histograms,
        "sim_fabric_queue_ms": result.fabric["queue_delay_ms_total"],
        "sim_host_cpu_queue_ms": result.fabric["host_cpu_queue_ms"],
    }


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [abs(float(row["relative_error_pct"])) for row in rows]
    log_errors = [float(row["absolute_log_error"]) for row in rows]
    batch_errors = [
        abs(float(row["merged_batch_relative_error_pct"])) for row in rows
    ]
    batch_log_errors = [
        float(row["merged_batch_absolute_log_error"]) for row in rows
    ]
    rms_log = math.sqrt(statistics.fmean(x * x for x in log_errors))
    batch_rms_log = math.sqrt(
        statistics.fmean(x * x for x in batch_log_errors)
    )
    return {
        "cells": len(rows),
        "mape_pct": statistics.fmean(errors),
        "median_ape_pct": statistics.median(errors),
        "worst_ape_pct": max(errors),
        "rms_log_error": rms_log,
        "merged_batch_mape_pct": statistics.fmean(batch_errors),
        "merged_batch_worst_ape_pct": max(batch_errors),
        "merged_batch_rms_log_error": batch_rms_log,
        "combined_score": rms_log + 0.5 * batch_rms_log,
    }


def evaluate(
    selected: list[dict[str, Any]],
    model: Any,
    tokens: int,
    metric_warmup_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions = [
        simulate(row, model, tokens, metric_warmup_tokens)
        for row in selected
    ]
    return predictions, summary(predictions)


def profile_json(
    candidate: Candidate,
    dataset_id: str,
    selected_segments: list[int],
    calibration_rows: list[dict[str, Any]],
    calibration_summary: dict[str, Any],
    primary_rows: list[dict[str, Any]],
    primary_summary: dict[str, Any],
    external_rows: list[dict[str, Any]],
    external_summary: dict[str, Any],
) -> dict[str, Any]:
    model = make_model(candidate)
    terms = {
        name: {
            "alpha_us": term.alpha_us,
            "beta_us_per_item": term.beta_us_per_item,
            "gamma_us_per_layer": term.gamma_us_per_layer,
            "delta_us_per_item_layer": term.delta_us_per_item_layer,
        }
        for name, term in model.terms.items()
    }
    transfers = {
        direction: {
            "fixed_us": term.fixed_us,
            "per_tensor_us": term.per_tensor_us,
            "bandwidth_gbps": term.bandwidth_gbps,
        }
        for direction, term in model.transfers.items()
    }
    gates = {
        "single_hardware_domain": True,
        "predeclared_primary_holdout": True,
        "calibration_mape_le_15pct": calibration_summary["mape_pct"] <= 15,
        "calibration_worst_ape_le_30pct": (
            calibration_summary["worst_ape_pct"] <= 30
        ),
        "calibration_merge_mape_le_20pct": (
            calibration_summary["merged_batch_mape_pct"] <= 20
        ),
        "primary_holdout_mape_le_20pct": primary_summary["mape_pct"] <= 20,
        "primary_holdout_worst_ape_le_30pct": (
            primary_summary["worst_ape_pct"] <= 30
        ),
        "primary_holdout_merge_mape_le_20pct": (
            primary_summary["merged_batch_mape_pct"] <= 20
        ),
        "external_holdout_mape_le_20pct": external_summary["mape_pct"] <= 20,
        "external_holdout_worst_ape_le_30pct": (
            external_summary["worst_ape_pct"] <= 30
        ),
        "external_holdout_merge_mape_le_20pct": (
            external_summary["merged_batch_mape_pct"] <= 20
        ),
        # These two cannot be repaired by curve fitting.
        "independent_hardware_repeats": False,
        "d2h_h2d_components_independently_identified": False,
    }
    return {
        "schema_version": 1,
        "profile_id": (
            "autodl-clean-scale-"
            + "-".join(f"L{value}" for value in selected_segments)
            + "-candidate-v2"
        ),
        "hardware_id": "autodl-8gpu-host-2026-08",
        "provenance": (
            "fit only on predeclared combined-request-agent calibration cells "
            f"from {dataset_id}; included segments={selected_segments}"
        ),
        "terms": terms,
        "transfers": transfers,
        "valid_batch_range": [1, 8],
        "valid_batch_ranges": {"decoder_prefill": [1, 32]},
        "valid_layer_range": [
            min(32 // value for value in selected_segments),
            max(32 // value for value in selected_segments),
        ],
        "environment": model.environment,
        "calibration": {
            "gates": gates,
            "paper_eligible": all(gates.values()),
            "selected_candidate": asdict(candidate),
            "calibration_summary": calibration_summary,
            "primary_holdout_summary": primary_summary,
            "external_holdout_summary": external_summary,
            "calibration_rows": calibration_rows,
            "primary_holdout_rows": primary_rows,
            "external_holdout_rows": external_rows,
            "limitation": (
                "one independent run per hardware cell; transfer/invocation "
                "decomposition is an ensemble assumption; validation covers "
                f"only measured segment counts {selected_segments}"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--profile-out", type=Path, required=True)
    parser.add_argument("--search-out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument(
        "--validation-tokens", type=int,
        help=(
            "rerun finalists and all holdouts with this many tokens per "
            "client; use 256 with --validation-warmup-tokens 32 to match the "
            "hardware fixed window"
        ),
    )
    parser.add_argument("--validation-warmup-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--segments", nargs="+", type=int,
        help="fit and validate only these segment/GPU counts",
    )
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text())
    rows = dataset["rows"]
    selected_segments = sorted(
        set(args.segments)
        if args.segments is not None
        else {int(row["segments"]) for row in rows}
    )
    rows = [
        row for row in rows
        if int(row["segments"]) in set(selected_segments)
    ]
    present_segments = sorted({int(row["segments"]) for row in rows})
    if present_segments != selected_segments:
        raise SystemExit(
            f"requested segments {selected_segments}, found {present_segments}"
        )
    if len(selected_segments) < 2:
        raise SystemExit("at least two segment counts are required")
    calibration = [row for row in rows if row["split"] == "calibration"]
    stage_a = [row for row in calibration if row["clients"] in (1, 64)]
    stage_b = [row for row in calibration if row["clients"] in (4, 16)]
    primary = [row for row in rows if row["split"] == "primary_holdout"]
    external = [row for row in rows if row["split"] == "external_holdout"]
    anchor_segments = min(selected_segments)
    target = next(
        row for row in calibration
        if row["segments"] == anchor_segments and row["clients"] == 1
    )
    serial_rows = sorted(
        (
            row for row in calibration
            if row["clients"] == 1
        ),
        key=lambda row: row["segments"],
    )
    xs = [float(row["segments"]) for row in serial_rows]
    ys = [float(row["itl_ms"]["mean"]) * 1000 for row in serial_rows]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    target_hop_us = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / sum((x - mean_x) ** 2 for x in xs)
    search: list[dict[str, Any]] = []
    all_candidates = candidates(
        float(target["itl_ms"]["mean"]) * 1000,
        target_hop_us,
        anchor_segments,
    )
    started = time.monotonic()
    print(
        f"fixed-context hop={target_hop_us:.1f}us; stage A: "
        f"{len(all_candidates)} candidates x {len(stage_a)} cells"
    )
    for index, candidate in enumerate(all_candidates, 1):
        predictions, metrics = evaluate(
            stage_a, make_model(candidate), args.tokens
        )
        search.append({
            "candidate": asdict(candidate),
            "candidate_id": candidate.candidate_id,
            "stage_a": metrics,
            "stage_a_rows": predictions,
        })
        print(
            f"  {index:2d}/{len(all_candidates)} {candidate.candidate_id}: "
            f"TPS-MAPE={metrics['mape_pct']:.1f}% "
            f"batch-MAPE={metrics['merged_batch_mape_pct']:.1f}%",
            flush=True,
        )
    search.sort(key=lambda item: item["stage_a"]["combined_score"])
    finalists = search[:args.top_k]
    print(f"stage B: top {len(finalists)} x {len(stage_b)} cells", flush=True)
    for item in finalists:
        candidate = Candidate(**item["candidate"])
        rows_b, metrics_b = evaluate(stage_b, make_model(candidate), args.tokens)
        combined = item["stage_a_rows"] + rows_b
        item["stage_b"] = metrics_b
        item["stage_b_rows"] = rows_b
        item["calibration"] = summary(combined)
    finalists.sort(key=lambda item: item["calibration"]["combined_score"])
    validation_tokens = args.validation_tokens or args.tokens
    validation_warmup = (
        args.validation_warmup_tokens
        if args.validation_tokens is not None
        else None
    )
    if (
        validation_warmup is not None
        and validation_tokens <= validation_warmup
    ):
        raise SystemExit(
            "validation tokens must exceed validation warm-up tokens"
        )
    if args.validation_tokens is not None:
        print(
            f"stage C: top {len(finalists)} x {len(calibration)} cells, "
            f"tokens={validation_tokens}, "
            f"warmup={args.validation_warmup_tokens}",
            flush=True,
        )
        for item in finalists:
            candidate = Candidate(**item["candidate"])
            formal_rows, formal_metrics = evaluate(
                calibration,
                make_model(candidate),
                validation_tokens,
                validation_warmup,
            )
            item["formal_calibration"] = formal_metrics
            item["formal_calibration_rows"] = formal_rows
        finalists.sort(
            key=lambda item: item["formal_calibration"]["combined_score"]
        )
    best_item = finalists[0]
    best = Candidate(**best_item["candidate"])
    print(f"holdout: selected {best.candidate_id}", flush=True)
    primary_rows, primary_metrics = evaluate(
        primary,
        make_model(best),
        validation_tokens,
        validation_warmup,
    )
    external_rows, external_metrics = evaluate(
        external,
        make_model(best),
        validation_tokens,
        validation_warmup,
    )
    calibration_rows = best_item.get(
        "formal_calibration_rows",
        best_item["stage_a_rows"] + best_item["stage_b_rows"],
    )
    calibration_metrics = best_item.get(
        "formal_calibration", best_item["calibration"]
    )
    profile = profile_json(
        best,
        str(dataset["dataset_id"]),
        selected_segments,
        calibration_rows,
        calibration_metrics,
        primary_rows,
        primary_metrics,
        external_rows,
        external_metrics,
    )
    args.profile_out.parent.mkdir(parents=True, exist_ok=True)
    args.search_out.parent.mkdir(parents=True, exist_ok=True)
    args.profile_out.write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    args.search_out.write_text(json.dumps({
        "schema_version": 1,
        "dataset_id": dataset["dataset_id"],
        "selected_segments": selected_segments,
        "search_tokens_per_client": args.tokens,
        "validation_tokens_per_client": validation_tokens,
        "validation_warmup_tokens": (
            validation_warmup
            if validation_warmup is not None
            else max(2, min(4, validation_tokens // 3))
        ),
        "anchor_segments": anchor_segments,
        "target_anchor_itl_us": float(target["itl_ms"]["mean"]) * 1000,
        "target_hop_us_per_segment": target_hop_us,
        "elapsed_wall_seconds": time.monotonic() - started,
        "stage_a_candidates": search,
        "finalists": finalists,
    }, indent=2, ensure_ascii=False))
    print(f"calibration MAPE={calibration_metrics['mape_pct']:.1f}%")
    print(f"primary holdout MAPE={primary_metrics['mape_pct']:.1f}%")
    print(f"external holdout MAPE={external_metrics['mape_pct']:.1f}%")
    print(f"paper_eligible={str(profile['calibration']['paper_eligible']).lower()}")
    print(f"wrote {args.profile_out} and {args.search_out}")


if __name__ == "__main__":
    main()
