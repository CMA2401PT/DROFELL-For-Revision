#!/usr/bin/env python3
"""Replay captured CUDA operations through the Memory Digital Twin.

The original script mixed plotting and replay, fixed the root memory to one
machine, and referenced an undefined ``ret`` for staged operations. This
version emits machine-readable per-operation measurements plus an optional
interactive plot and reads the captured memory policy unless explicitly
overridden.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import time
from typing import Any, TypedDict

from dealers.real_basic.memory import MemoryBrain, MemoryConfigBasic
import split_qllm_models
from scripts.offline.testbed import ModelConfig


STAGED_OPERATIONS = {
    "train_tensor_to_embedding_tokens",
    "forward",
    "compute_logits_loss_and_grad",
    "backward",
    "infer_tensor_to_embedding_tokens",
    "do_prefill",
    "do_generating",
}
KWARG_OPERATIONS = {
    "make_model_head",
    "make_model_tail",
    "make_model_decoders",
    "remove_decoders_train_binding",
    "swapout",
    "remove_decoders_infer_binding",
    "remove_model_head",
    "remove_model_tail",
    "remove_model_decoders",
}


class CaptureRecord(TypedDict):
    sequence: int
    operation: str
    args: Any
    cuda_memory_status: Any
    result: Any
    captured_model_pools: dict[str, int]
    graph_capture_preparations: list[tuple[str, int]]
    graph_runtime_snapshot: dict[Any, int]


def apply_once(
    brain: MemoryBrain,
    name: str,
    args: Any,
    result: Any,
    graph_capture_preparations: list[tuple[str, int]],
    graph_runtime_snapshot: dict[Any, int],
    *,
    virtual: bool,
) -> None:
    operation = getattr(brain, name, None)
    if operation is None:
        raise ValueError(f"unsupported captured operation: {name}")
    if name in STAGED_OPERATIONS:
        virtual_bind, actual_bind = operation(args)
        if virtual:
            virtual_bind()
        else:
            # This is the same lifetime boundary used by BasicBrainOperates:
            # commit the predicted operation before binding its real output.
            brain.merge_all()
            actual_bind(result)
    elif name in KWARG_OPERATIONS:
        operation(**args)
    else:
        operation(args)
    for binding, temporary_block_count in graph_capture_preparations:
        brain.prepare_decode_graph_capture(binding, temporary_block_count)
    brain.reconcile_graph_runtime(graph_runtime_snapshot)


def replay_step(
    brain: MemoryBrain,
    name: str,
    args: Any,
    result: Any,
    graph_capture_preparations: list[tuple[str, int]],
    graph_runtime_snapshot: dict[Any, int],
) -> tuple[int, int, int]:
    """Measure speculative replay followed by the committed replay."""
    started = time.perf_counter_ns()
    apply_once(
        brain, name, args, result,
        graph_capture_preparations, graph_runtime_snapshot,
        virtual=True,
    )
    brain.revert_all()
    after_revert = time.perf_counter_ns()
    apply_once(
        brain, name, args, result,
        graph_capture_preparations, graph_runtime_snapshot,
        virtual=False,
    )
    brain.merge_all()
    ended = time.perf_counter_ns()
    return ended - started, after_revert - started, ended - after_revert


def load_capture(path: Path) -> tuple[dict[str, Any], list[CaptureRecord]]:
    with path.open("rb") as handle:
        captured = pickle.load(handle)
    if isinstance(captured, list):
        if any(record[5].get("runtime", 0) for record in captured):
            raise ValueError(
                "legacy graph-enabled capture has no graph preparation or "
                "runtime snapshot; recapture it with the current scripts"
            )
        records: list[CaptureRecord] = []
        for step, name, args, status, result, pools in captured:
            records.append({
                "sequence": step,
                "operation": name,
                "args": pickle.loads(args),
                "cuda_memory_status": status,
                "result": pickle.loads(result),
                "captured_model_pools": pools,
                "graph_capture_preparations": [],
                "graph_runtime_snapshot": {},
            })
        return {}, records
    if not isinstance(captured, dict) or captured.get("schema_version") != 2:
        raise ValueError("unsupported memory capture format")
    metadata = captured.get("metadata")
    raw_records = captured.get("records")
    if not isinstance(metadata, dict) or not isinstance(raw_records, list):
        raise ValueError("invalid memory capture envelope")
    records = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("invalid memory capture record")
        records.append({
            "sequence": raw["sequence"],
            "operation": raw["operation"],
            "args": pickle.loads(raw["args"]),
            "cuda_memory_status": raw["cuda_memory_status"],
            "result": pickle.loads(raw["result"]),
            "captured_model_pools": raw["captured_model_pools"],
            "graph_capture_preparations": raw.get(
                "graph_capture_preparations", []
            ),
            "graph_runtime_snapshot": raw.get(
                "graph_runtime_snapshot", {}
            ),
        })
    return metadata, records


def replay(
    records: list[CaptureRecord],
    root_memory: int,
    preallocate: bool,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    model_config_kwargs = {}
    for name in ("model_name", "max_seq_len", "decoder_backend"):
        if name in metadata:
            model_config_kwargs[name] = metadata[name]
    model_config = ModelConfig(**model_config_kwargs)
    captured_memory_config = metadata.get("memory_config", {})
    memory_config_kwargs = {
        name: captured_memory_config[name]
        for name in (
            "cache_engine_implement",
            "base_models_name",
            "train_bindings_name",
            "train_grad_optim_name",
            "train_ctx_name",
            "kv_cache_name",
            "preswap",
            "graph_budget",
        )
        if name in captured_memory_config
    }
    brain = MemoryBrain(
        models_path=split_qllm_models.lib_path,
        basic_memory_config=MemoryConfigBasic(
            **memory_config_kwargs,
            root_memory=root_memory,
            preallocate=preallocate,
        ),
        max_seq_len=model_config.max_seq_len,
        decoder_implement=model_config.decoder_backend,
    )
    rows: list[dict[str, Any]] = []
    for record in records:
        sequence = record["sequence"]
        name = record["operation"]
        args = record["args"]
        status = record["cuda_memory_status"]
        result = record["result"]
        captured_pools = record["captured_model_pools"]
        before = int(brain.total)
        duration_ns, revert_phase_ns, commit_phase_ns = replay_step(
            brain,
            name,
            args,
            result,
            record["graph_capture_preparations"],
            record["graph_runtime_snapshot"],
        )
        predicted = int(brain.total)
        # ``preallocate`` changes the twin's optimizer/gradient lifetime.  It
        # must not change the CUDA comparison metric: allocator bin sizes are
        # implementation details, while the twin models requested tensor bytes.
        actual = int(status.accumulate.requested)
        error = predicted - actual
        simulated_pools = dict(brain.pools)
        pool_names = captured_pools.keys() | simulated_pools.keys()
        pools_match = all(
            captured_pools.get(pool, 0) == simulated_pools.get(pool, 0)
            for pool in pool_names
        )
        rows.append({
            "capture_sequence": sequence,
            "operation": name,
            "actual_bytes": actual,
            "actual_allocated_bytes": int(status.accumulate.allocated),
            "actual_approximate_bytes": int(status.accumulate.approximate),
            "predicted_bytes": predicted,
            "prediction_error_bytes": error,
            "absolute_error_bytes": abs(error),
            "relative_error": None if actual == 0 else error / actual,
            "predicted_delta_bytes": predicted - before,
            "actual_delta_bytes": int(status.delta.total_change),
            "simulate_ns": duration_ns,
            "speculate_and_revert_ns": revert_phase_ns,
            "commit_and_merge_ns": commit_phase_ns,
            "captured_model_pools": captured_pools,
            "simulated_model_pools": simulated_pools,
            "model_pools_match": pools_match,
            "graph_capture_preparations": [
                {
                    "binding": binding,
                    "temporary_block_count": temporary_block_count,
                }
                for binding, temporary_block_count in record[
                    "graph_capture_preparations"
                ]
            ],
            "active_graphs": [
                {
                    "key": list(key),
                    "requested_bytes": requested_bytes,
                }
                for key, requested_bytes in record[
                    "graph_runtime_snapshot"
                ].items()
            ],
            "graph_runtime_bytes": sum(
                record["graph_runtime_snapshot"].values()
            ),
        })
    errors = [row["prediction_error_bytes"] for row in rows]
    absolute = [abs(value) for value in errors]
    squared = [value * value for value in errors]
    return {
        "preallocate": preallocate,
        "actual_memory_metric": "cuda_requested_bytes",
        "operation_count": len(rows),
        "revert_count": len(rows),
        "merge_count": len(rows),
        "mean_error_bytes": sum(errors) / len(errors) if errors else None,
        "mean_absolute_error_bytes": sum(absolute) / len(absolute) if absolute else None,
        "root_mean_squared_error_bytes": math.sqrt(sum(squared) / len(squared)) if squared else None,
        "max_absolute_error_bytes": max(absolute, default=None),
        "model_pool_mismatch_count": sum(
            not row["model_pools_match"] for row in rows
        ),
        "total_simulate_ns": sum(row["simulate_ns"] for row in rows),
        "rows": rows,
    }


def write_plot(
    modes: list[dict[str, Any]],
    output: Path,
    *,
    embed_plotlyjs: bool,
    error_subplot_ratio: float,
) -> None:
    """Write an interactive total/pool comparison and error timeline."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    modes_by_preallocate = {mode["preallocate"]: mode for mode in modes}
    base_mode = modes_by_preallocate.get(False, modes[0])
    base_rows = base_mode["rows"]
    x = [
        f'{row["capture_sequence"]}:{row["operation"]}'
        for row in base_rows
    ]

    def to_mib(values: list[int]) -> list[float]:
        return [value / 2**20 for value in values]

    separate_error = error_subplot_ratio > 0
    if separate_error:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[1 - error_subplot_ratio, error_subplot_ratio],
            subplot_titles=(
                "CUDA and digital-twin memory",
                "Prediction error",
            ),
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=("CUDA and digital-twin memory",),
        )

    pool_names = sorted({
        pool
        for row in base_rows
        for pool in row["simulated_model_pools"]
    })
    for pool in pool_names:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=to_mib([
                    row["simulated_model_pools"].get(pool, 0)
                    for row in base_rows
                ]),
                name=f"Twin pool: {pool}",
                mode="lines",
                stackgroup="twin-pools",
                line={"width": 1.5},
                opacity=0.72,
                hovertemplate=(
                    "%{x}<br>%{y:.3f} MiB"
                    "<extra>%{fullData.name}</extra>"
                ),
            ),
            row=1,
            col=1,
        )

    for name, field, dash, color, width in (
        ("CUDA requested", "actual_bytes", "solid", "#111827", 3.2),
        ("CUDA allocated", "actual_allocated_bytes", "dot", "#94a3b8", 2.4),
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=to_mib([row[field] for row in base_rows]),
                name=name,
                mode="lines",
                line={"dash": dash, "width": width, "color": color},
                hovertemplate=(
                    "%{x}<br>%{y:.3f} MiB"
                    "<extra>%{fullData.name}</extra>"
                ),
            ),
            row=1,
            col=1,
        )

    multiple_modes = len(modes) > 1
    for preallocate, policy_label in (
        (False, "no preallocate"),
        (True, "preallocate"),
    ):
        mode = modes_by_preallocate.get(preallocate)
        if mode is None:
            continue
        rows = mode["rows"]
        suffix = f" · {policy_label}" if multiple_modes else ""
        policy_color = "#2563eb" if not preallocate else "#d97706"
        legend_group = f"twin-{policy_label}"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=to_mib([row["predicted_bytes"] for row in rows]),
                name=f"Twin total{suffix}",
                mode="lines",
                line={
                    "dash": "dash",
                    "width": 3.0,
                    "color": policy_color,
                },
                legendgroup=legend_group,
                hovertemplate=(
                    "%{x}<br>%{y:.3f} MiB"
                    "<extra>%{fullData.name}</extra>"
                ),
            ),
            row=1,
            col=1,
        )
        error_row = 2 if separate_error else 1
        fig.add_trace(
            go.Scatter(
                x=x,
                y=to_mib([
                    row["prediction_error_bytes"] for row in rows
                ]),
                name=f"Prediction error{suffix}",
                mode="lines",
                line={"width": 2.6, "color": policy_color},
                legendgroup=legend_group,
                hovertemplate=(
                    "%{x}<br>%{y:.3f} MiB"
                    "<extra>%{fullData.name}</extra>"
                ),
            ),
            row=error_row,
            col=1,
        )

    axis_style = {
        "title_font": {"size": 15},
        "tickfont": {"size": 12},
        "gridcolor": "rgba(148, 163, 184, 0.22)",
        "gridwidth": 1,
        "zeroline": False,
    }
    fig.update_yaxes(
        title_text="Memory (MiB)", row=1, col=1, **axis_style
    )
    if separate_error:
        fig.add_hline(
            y=0,
            line_width=1.5,
            line_color="#64748b",
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text="Error (MiB)", row=2, col=1, **axis_style
        )
        fig.update_xaxes(
            title_text="Capture sequence and operation",
            title_font={"size": 15},
            row=2,
            col=1,
        )
    else:
        fig.update_xaxes(
            title_text="Capture sequence and operation",
            title_font={"size": 15},
            row=1,
            col=1,
        )
    fig.update_xaxes(
        showticklabels=False,
        gridcolor="rgba(148, 163, 184, 0.14)",
        gridwidth=1,
        zeroline=False,
    )
    fig.update_layout(
        template="plotly_white",
        font={"family": "Times New Roman", "size": 13, "color": "#1f2937"},
        autosize=True,
        hovermode="x unified",
        hoverlabel={"font": {"size": 12}},
        legend={
            "font": {"size": 12},
            "x": 0.012,
            "xanchor": "left",
            "y": 0.99,
            "yanchor": "top",
            "bgcolor": "rgba(255, 255, 255, 0.82)",
            "bordercolor": "rgba(148, 163, 184, 0.45)",
            "borderwidth": 1,
            "itemsizing": "constant",
            "tracegroupgap": 8,
            "groupclick": "togglegroup",
        },
        margin={"l": 88, "r": 36, "t": 82, "b": 72},
    )
    fig.update_annotations(font={"size": 17})
    fig.write_html(
        output,
        include_plotlyjs=True if embed_plotlyjs else "cdn",
        full_html=True,
        default_width="100%",
        default_height="calc(100vh - 16px)",
        config={"responsive": True},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument(
        "--root-memory-gib",
        type=float,
        help="override captured root memory; required for legacy captures",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preallocate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "restrict replay to one allocation policy; by default both "
            "preallocate modes are simulated and plotted"
        ),
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write an interactive Plotly comparison to OUTPUT/compare.html",
    )
    parser.add_argument(
        "--embed-plotlyjs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="embed Plotly JS in compare.html instead of using the CDN",
    )
    parser.add_argument(
        "--error-subplot-ratio",
        type=float,
        default=0.28,
        metavar="RATIO",
        help=(
            "vertical share of the prediction-error subplot in [0, 1); "
            "0 keeps errors in the main plot (default: 0.28)"
        ),
    )
    args = parser.parse_args()
    if args.root_memory_gib is not None and args.root_memory_gib <= 0:
        parser.error("--root-memory-gib must be positive")
    if not 0 <= args.error_subplot_ratio < 1:
        parser.error("--error-subplot-ratio must be in [0, 1)")
    metadata, records = load_capture(args.capture)
    if not records:
        raise ValueError("capture contains no operations")
    captured_root_memory = metadata.get("memory_config", {}).get(
        "root_memory", -1
    )
    if args.root_memory_gib is None:
        if captured_root_memory is None or captured_root_memory <= 0:
            parser.error(
                "--root-memory-gib is required because the capture has no "
                "root-memory metadata"
            )
        root_memory = int(captured_root_memory)
    else:
        root_memory = round(args.root_memory_gib * 2**30)
    captured_preallocate = bool(
        metadata.get("memory_config", {}).get("preallocate", False)
    )
    preallocate_modes = (
        [False, True]
        if args.preallocate is None
        else [args.preallocate]
    )
    payload = {
        "schema_version": 2,
        "capture": str(args.capture),
        "root_memory_bytes": root_memory,
        "evidence_type": "captured_cuda_allocator_vs_digital_twin_replay",
        "capture_metadata": metadata,
        "captured_preallocate": captured_preallocate,
        "modes": [
            replay(records, root_memory, preallocate, metadata)
            for preallocate in preallocate_modes
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.plot:
        write_plot(
            payload["modes"],
            args.output / "compare.html",
            embed_plotlyjs=args.embed_plotlyjs,
            error_subplot_ratio=args.error_subplot_ratio,
        )


if __name__ == "__main__":
    main()
