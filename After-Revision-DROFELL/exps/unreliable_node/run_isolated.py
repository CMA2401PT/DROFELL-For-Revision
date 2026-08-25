#!/usr/bin/env python3
"""Run fixed, independent high-failure replicates after a topology smoke pass."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Sequence


EXP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXP_ROOT.parents[1]
DEFAULT_PYTHON = Path("/home/dai/projects/LLM/dflash_related/.venv/bin/python")
TOPOLOGIES = ("1_1_1", "1_2_1", "1_4_1", "2_2_1", "2_4_1",
              "2_1_2", "2_2_2", "2_4_2")
PAPER_PROBABILITIES = (0.0, 1e-4, 1e-3, 1e-2, 5e-2)


class ExperimentError(RuntimeError):
    pass


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def command_text(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--batch-id", default=f"isolated-{utc_run_id()}")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--probability", type=float, default=0.05)
    parser.add_argument(
        "--topologies", nargs="+", choices=TOPOLOGIES, default=TOPOLOGIES,
    )
    args = parser.parse_args()
    if args.replicates <= 0:
        parser.error("--replicates must be positive")
    if args.probability not in PAPER_PROBABILITIES:
        parser.error("--probability must come from the paper grid")
    if not args.batch_id or "/" in args.batch_id or args.batch_id in {".", ".."}:
        parser.error("--batch-id must be one path component")

    controller_root = EXP_ROOT / "results" / args.batch_id
    log_root = EXP_ROOT / "logs" / args.batch_id
    environment_path = EXP_ROOT / "environment" / f"{args.batch_id}.json"
    for path in (controller_root, log_root, environment_path):
        if path.exists():
            raise ExperimentError(f"refusing to overwrite: {path}")
    controller_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for topology in dict.fromkeys(args.topologies):
        for replicate in range(args.replicates):
            child_id = f"{args.batch_id}-{topology}-rep{replicate:02d}"
            log_path = log_root / f"{topology}-rep{replicate:02d}.log"
            command = (
                str(args.python.absolute()), str(EXP_ROOT / "run.py"),
                "--run-id", child_id,
                "--topologies", topology,
                "--probabilities", format(args.probability, ".8g"),
                "--repeats", "1",
                "--skip-smoke",
            )
            started = datetime.now(timezone.utc).isoformat()
            record: dict[str, Any] = {
                "topology": topology,
                "replicate": replicate,
                "child_run_id": child_id,
                "command": command_text(command),
                "log": str(log_path.relative_to(EXP_ROOT)),
                "started_utc": started,
                "finished_utc": None,
                "returncode": None,
                "validated": False,
            }
            records.append(record)
            write_json(environment_path, {
                "schema_version": 1,
                "batch_id": args.batch_id,
                "protocol": (
                    "Each replicate starts a fresh network and benchmark; "
                    "the topology had already passed the built-in smoke in a "
                    "separate run. Failed fixed replicates are retained and "
                    "are not replaced."
                ),
                "python": str(args.python.absolute()),
                "probability": args.probability,
                "requested_replicates_per_topology": args.replicates,
                "topologies": list(dict.fromkeys(args.topologies)),
                "records": records,
            })
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command, cwd=REPO_ROOT, text=True,
                    stdout=log, stderr=subprocess.STDOUT, check=False,
                )
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            record["returncode"] = completed.returncode
            summary_path = EXP_ROOT / "results" / child_id / "summary.json"
            if completed.returncode == 0 and summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                candidates = [
                    row for row in summary.get("runs", [])
                    if row.get("deployment") == "combined-request-agent"
                    and row.get("topology") == topology
                    and float(row.get("probability")) == args.probability
                ]
                if len(candidates) != 1:
                    record["validation_error"] = (
                        f"expected one result row, found {len(candidates)}"
                    )
                else:
                    row = dict(candidates[0])
                    invalid = []
                    if int(row.get("timestamps", -1)) != 1027:
                        invalid.append("timestamps!=1027")
                    if int(row.get("graph_fallback_records", -1)) != 0:
                        invalid.append("graph_fallback_records!=0")
                    if int(row.get("graph_non_sampling_fallback_records", -1)) != 0:
                        invalid.append("graph_non_sampling_fallback_records!=0")
                    if invalid:
                        record["validation_error"] = ", ".join(invalid)
                        record["validated"] = False
                    else:
                        row["isolated_replicate"] = replicate
                        row["child_run_id"] = child_id
                        rows.append(row)
                        record["validated"] = True
            else:
                record["validated"] = False
                record["summary_exists"] = summary_path.exists()
            write_json(environment_path, {
                "schema_version": 1,
                "batch_id": args.batch_id,
                "protocol": (
                    "Each replicate starts a fresh network and benchmark; "
                    "the topology had already passed the built-in smoke in a "
                    "separate run. Failed fixed replicates are retained and "
                    "are not replaced."
                ),
                "python": str(args.python.absolute()),
                "probability": args.probability,
                "requested_replicates_per_topology": args.replicates,
                "topologies": list(dict.fromkeys(args.topologies)),
                "records": records,
            })

    write_csv(controller_root / "successful_runs.csv", rows)
    write_json(controller_root / "summary.json", {
        "schema_version": 1,
        "batch_id": args.batch_id,
        "probability": args.probability,
        "requested_replicates_per_topology": args.replicates,
        "successful_rows": rows,
        "attempts": records,
        "success_count": len(rows),
        "failure_count": len(records) - len(rows),
    })
    print(json.dumps({
        "batch_id": args.batch_id,
        "success_count": len(rows),
        "failure_count": len(records) - len(rows),
        "summary": str(controller_root / "summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
