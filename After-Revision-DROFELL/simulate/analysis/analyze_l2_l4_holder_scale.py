#!/usr/bin/env python3
"""Audit the L2/L4 active-holder scale result and derive compact claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def endpoint(rows: list[dict[str, Any]], segments: int) -> dict[str, Any]:
    selected = sorted(
        (row for row in rows if int(row["segments"]) == segments),
        key=lambda row: int(row["redundancy"]),
    )
    first, last = selected[0], selected[-1]
    return {
        "segments": segments,
        "clients": int(first["clients"]),
        "start": {
            "redundancy": int(first["redundancy"]),
            "logical_nodes": int(first["logical_nodes"]),
            "steady_tokens_per_second": float(
                first["steady_tokens_per_second"]
            ),
            "mean_itl_ms": float(first["itl_ms"]["mean"]),
            "packets": int(first["packets"]),
            "wire_bytes": int(first["wire_bytes"]),
        },
        "end": {
            "redundancy": int(last["redundancy"]),
            "logical_nodes": int(last["logical_nodes"]),
            "steady_tokens_per_second": float(
                last["steady_tokens_per_second"]
            ),
            "mean_itl_ms": float(last["itl_ms"]["mean"]),
            "packets": int(last["packets"]),
            "wire_bytes": int(last["wire_bytes"]),
        },
        "ratios": {
            "throughput": (
                float(last["steady_tokens_per_second"])
                / float(first["steady_tokens_per_second"])
            ),
            "mean_itl": (
                float(last["itl_ms"]["mean"])
                / float(first["itl_ms"]["mean"])
            ),
            "packets": int(last["packets"]) / int(first["packets"]),
            "wire_bytes": (
                int(last["wire_bytes"]) / int(first["wire_bytes"])
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    rows = result["rows"]
    single = [
        row for row in rows
        if row["sweep"] == "active_holders_single_client"
    ]
    loaded = [
        row for row in rows
        if row["sweep"] == "active_holders_loaded"
    ]

    formula_rows = []
    for row in single:
        segments = int(row["segments"])
        redundancy = int(row["redundancy"])
        tokens = int(row["raw_result"]["tokens_produced"])
        actual = int(
            row["raw_result"]["fabric"]["packets_by_api"].get(
                "grouped_infer_task", 0
            )
        )
        per_token = (
            (segments - 1) * redundancy * redundancy
            + 2 * segments * redundancy
        )
        expected = per_token * tokens
        formula_rows.append({
            "segments": segments,
            "redundancy": redundancy,
            "logical_nodes": int(row["logical_nodes"]),
            "tokens": tokens,
            "actual_grouped_infer_task_packets": actual,
            "expected_grouped_infer_task_packets": expected,
            "expected_packets_per_token": per_token,
            "passed": actual == expected,
        })

    no_cost_extrapolation = all(
        not row["cost_extrapolations"] for row in rows
    )
    clean_teardown = all(not row["notes"] for row in rows)
    checks = {
        "only_L2_L4": sorted({row["segments"] for row in rows}) == [2, 4],
        "grouped_packet_formula_exact": all(
            row["passed"] for row in formula_rows
        ),
        "deterministic_rerun": bool(
            result["reproducibility"]["deterministic_rerun_passed"]
        ),
        "no_cost_extrapolation": no_cost_extrapolation,
        "clean_teardown": clean_teardown,
    }
    report = {
        "schema_version": 1,
        "evidence_id": f"{result['evidence_id']}-analysis",
        "source": str(args.result),
        "structural_checks_passed": all(checks.values()),
        "checks": checks,
        "grouped_packet_formula": {
            "expression": "((L - 1) * r^2 + 2 * L * r) * tokens",
            "rows": formula_rows,
        },
        "single_client_endpoints": [
            endpoint(single, segments) for segments in (2, 4)
        ],
        "loaded_endpoints": [
            endpoint(loaded, segments) for segments in (2, 4)
        ],
        "validity": result["validity"],
        "paper_eligible": False,
        "reason": (
            "structural packet amplification is verified, but absolute latency "
            "is blocked by the source profile and unvalidated shared-domain "
            "capacity"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        f"structural_checks_passed={str(all(checks.values())).lower()}"
    )
    for name, passed in checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print("paper_eligible=false")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
