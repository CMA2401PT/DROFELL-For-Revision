#!/usr/bin/env python3
"""Analyze the short Llama2-7B PEFT table reproduction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PAPER_TOTAL_TPS = {
    "p4-b8-c1": 2012.51,
    "p4-b32-c1": 2669.07,
    "p16-b8-c1": 1991.09,
    "p16-b32-c1": 2668.05,
    "r4-b8-c1": 2004.64,
    "r4-b32-c1": 2654.36,
    "r16-b8-c1": 2015.59,
    "r16-b32-c1": 2669.94,
    "p4-b8-c2": 2877.95,
    "p4-b32-c2": 2958.20,
    "p16-b8-c2": 2869.48,
    "p16-b32-c2": 2931.21,
    "r4-b8-c2": 2954.15,
    "r4-b32-c2": 2955.37,
    "r16-b8-c2": 2948.32,
    "r16-b32-c2": 2957.10,
    "p4-b8-c4": 2930.47,
    "p4-b32-c4": 2950.13,
    "p16-b8-c4": 2939.92,
    "p16-b32-c4": 2935.03,
    "r4-b8-c4": 2974.02,
    "r4-b32-c4": 2981.79,
    "r16-b8-c4": 2970.51,
    "r16-b32-c4": 2974.90,
    "p4-b8-c8": 2935.01,
    "p4-b32-c8": 2946.23,
    "p16-b8-c8": 2942.54,
    "p16-b32-c8": 2942.44,
    "r4-b8-c8": 2981.46,
    "r4-b32-c8": 2979.03,
    "r16-b8-c8": 2974.65,
    "r16-b32-c8": 2967.25,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_case(case_dir: Path) -> dict[str, Any]:
    case = read_json(case_dir / "case.json")
    args = read_json(case_dir / "args.json")
    network = read_json(case_dir / "network.json")
    records = read_json(case_dir / "results.json")
    global_record = records[-1]
    tokens = sum(global_record["tokens"])
    legacy_start = global_record["start_times"][0]
    legacy_end = global_record["times"][-1]
    envelope_start = min(global_record["start_times"])
    envelope_end = max(global_record["times"])
    measured = tokens / (legacy_end - legacy_start)
    envelope = tokens / (envelope_end - envelope_start)
    paper = PAPER_TOTAL_TPS[case["paper_key"]]
    expected_minibatches = (
        case["clients"] * case["samples_per_client"]
        // case["minibatch_size"]
    )
    actual_minibatches = len(global_record["times"])
    if actual_minibatches != expected_minibatches:
        raise ValueError(
            f"{case_dir}: expected {expected_minibatches} minibatches, "
            f"got {actual_minibatches}"
        )
    if args["agent"] is not None:
        raise ValueError(f"{case_dir}: expected combined Request-Agent")
    return {
        **case,
        "network_name": network["name"],
        "network_spec_sha256": network["spec_sha256"],
        "global_minibatches": actual_minibatches,
        "global_tokens": tokens,
        "measurement_seconds": legacy_end - legacy_start,
        "total_tokens_per_second": measured,
        "envelope_total_tokens_per_second": envelope,
        "legacy_minus_envelope_tokens_per_second": measured - envelope,
        "paper_total_tokens_per_second": paper,
        "measured_over_paper": measured / paper,
    }


def summarize(run_root: Path) -> dict[str, Any]:
    rows = [
        summarize_case(case_file.parent)
        for case_file in sorted(run_root.glob("*/*/case.json"))
    ]
    if not rows:
        raise ValueError(f"no case results found under {run_root}")
    paired = []
    for case_id in ("ptuning-p4-b8-c4", "lora-r4-b8-c4"):
        by_backend = {
            row["backend"]: row for row in rows if row["case_id"] == case_id
        }
        if set(by_backend) == {"exllama_fuse", "marlin"}:
            exllama = by_backend["exllama_fuse"]["total_tokens_per_second"]
            marlin = by_backend["marlin"]["total_tokens_per_second"]
            paired.append({
                "case_id": case_id,
                "exllama_fuse_tokens_per_second": exllama,
                "marlin_tokens_per_second": marlin,
                "exllama_fuse_over_marlin": exllama / marlin,
            })
    summary = {
        "run_root": str(run_root),
        "cases": rows,
        "backend_comparison": paired,
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (run_root / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = (
            "backend", "case_id", "clients", "adapter", "trainable",
            "effective_batch", "samples_per_client", "global_tokens",
            "measurement_seconds", "total_tokens_per_second",
            "paper_total_tokens_per_second", "measured_over_paper",
        )
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
