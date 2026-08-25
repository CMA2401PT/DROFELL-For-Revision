#!/usr/bin/env python3
"""CPU-only regression checks for reviewer-grade measurement tooling."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (
        f"  ({detail})" if detail else ""
    ))
    if not condition:
        FAILURES.append(name)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def main() -> int:
    print("\n1. sampled runtime metrics")
    with tempfile.TemporaryDirectory(prefix="drosim-measurement-") as tmp:
        root = Path(tmp)
        metrics = root / "events.jsonl"
        os.environ["DROFELL_METRICS_PATH"] = str(metrics)
        os.environ["DROFELL_METRICS_EVENTS"] = "transport"
        os.environ["DROFELL_METRICS_SAMPLE_EVERY"] = "2"
        from utils.runtime_metrics import (
            _close_descriptor, record_event, record_sampled_event,
        )

        record_event("always")
        for index in range(5):
            record_sampled_event("transport", "sampled", index=index)
        for index in range(3):
            record_sampled_event("transfer", "disabled", index=index)
        _close_descriptor()
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        check(
            "sampling is deterministic and disabled categories stay absent",
            [row["event"] for row in rows]
            == ["always", "sampled", "sampled", "sampled"],
            str([row["event"] for row in rows]),
        )
        check(
            "sample ordinals retain the unsampled population index",
            [row.get("index") for row in rows[1:]] == [0, 2, 4],
        )
        os.environ.pop("DROFELL_METRICS_PATH", None)
        os.environ.pop("DROFELL_METRICS_EVENTS", None)
        os.environ.pop("DROFELL_METRICS_SAMPLE_EVERY", None)

    print("\n2. cumulative decode-graph replay differencing")
    from simulate.calibration.extract_scale_dataset import derive_case_replays

    fake = [
        {
            "cell_id": "L4-c128-combined",
            "segments": 4, "clients": 128,
            "deployment": "combined",
            "decode_graph": {
                "cumulative_replays_snapshot": 100,
                "inferred_work_items_including_warmup": 800,
            },
        },
        {
            "cell_id": "L4-c64-combined",
            "segments": 4, "clients": 64,
            "deployment": "combined",
            "decode_graph": {
                "cumulative_replays_snapshot": 300,
                "inferred_work_items_including_warmup": 800,
            },
        },
    ]
    derive_case_replays(fake)
    check(
        "first descending-concurrency cell uses the initial snapshot",
        fake[0]["decode_graph"]["case_replays"] == 100,
    )
    check(
        "next cell uses the adjacent snapshot delta",
        fake[1]["decode_graph"]["case_replays"] == 200,
    )
    check(
        "derived mean batches use per-case rather than cumulative replay counts",
        fake[0]["decode_graph"]["observed_mean_merged_batch"] == 8
        and fake[1]["decode_graph"]["observed_mean_merged_batch"] == 4,
    )

    print("\n3. hop-breakdown artifact parser")
    with tempfile.TemporaryDirectory(prefix="drosim-hop-artifacts-") as tmp:
        root = Path(tmp)
        experiment_root = root / "exps/infer_scale"
        result_root = experiment_root / "results/synthetic"
        case_dir = result_root / "scale-4/clients-1/combined-request-agent"
        case_dir.mkdir(parents=True)
        base_ns = 1_800_000_000_000_000_000
        outputs = [[0, base_ns + index * 1_000_000] for index in range(64)]
        write_json(case_dir / "result.json", outputs)
        write_json(case_dir / "case.json", {})
        write_json(case_dir / "args.json", {})
        write_json(case_dir / "tokens.json", {})
        write_json(case_dir / "decode_graph.json", {})
        write_json(experiment_root / "environment/synthetic.json", {
            "git_commit": "synthetic-commit",
            "git_status": "",
        })
        write_json(result_root / "network_specs/synthetic.json", {
            "spec_sha256": "synthetic-spec",
        })
        write_json(case_dir / "phases.json", {
            "request_start_ns": base_ns,
            "prefill_end_ns": base_ns,
            "end_ns": base_ns + 64_000_000,
        })
        stamp = datetime.fromtimestamp(
            (base_ns + 32_000_000) / 1e9, timezone.utc
        ).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
        (case_dir / "gpu_monitor.csv").write_text(
            f"{stamp}, 0, 1000, 47000, 50, P0, 1800, 200\n"
        )
        events = [
            {"event": "decoder_batch_selected", "actual_size": 1,
             "operation": "decoder_generating"},
            {"event": "device_to_host_enqueue", "enqueue_cpu_ns": 1000},
            {"event": "host_to_device_enqueue", "enqueue_cpu_ns": 1000},
            {"event": "operate_sync_wait", "duration_cpu_ns": 1000},
            {"event": "transport_serialize", "duration_cpu_ns": 1000,
             "api": "grouped_infer_task", "wire_bytes": 9000},
            {"event": "transport_socket_write", "duration_cpu_ns": 1000,
             "api": "grouped_infer_task"},
            {"event": "transport_deserialize", "duration_cpu_ns": 1000,
             "sender_to_deserialize_ns": 2000,
             "api": "grouped_infer_task"},
        ]
        with (result_root / "runtime_metrics.jsonl").open("w") as handle:
            for offset, event in enumerate(events, 1):
                handle.write(json.dumps({
                    "time_ns": base_ns + (40 + offset) * 1_000_000,
                    **event,
                }) + "\n")
        from simulate.hardware.analyze_hop_breakdown import run_row

        parsed = run_row({
            "run_id": "synthetic", "scale": 4, "clients": 1,
            "deployment": "combined-request-agent", "repeat": 0,
            "result_root": str(result_root), "case_dir": str(case_dir),
        })
        check("complete synthetic artifacts pass parser", parsed["complete"])
        check(
            "all required instrumentation types are recognized",
            not parsed["missing_event_types"],
            str(parsed.get("missing_event_types")),
        )
        check(
            "token timestamps produce a nonempty steady window",
            parsed["tokens"]["measured_intervals"] == 32,
        )

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {FAILURES}")
        return 1
    print("all measurement-tool checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
