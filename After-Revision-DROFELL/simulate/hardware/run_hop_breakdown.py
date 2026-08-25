from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path("/home/dai/projects/LLM/dflash_related/.venv/bin/python")
EXPERIMENT_ROOT = REPO_ROOT / "exps/infer_scale"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run-prefix", default="hop-breakdown-v1")
    parser.add_argument("--scales", nargs="+", type=int, default=(4, 8))
    parser.add_argument("--clients", nargs="+", type=int, default=(1, 64, 128))
    parser.add_argument(
        "--deployments", nargs="+",
        choices=("combined-request-agent", "persistent-agent"),
        default=("combined-request-agent",),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-tokens-per-client", type=int, default=256)
    parser.add_argument("--sample-every", type=int, default=8)
    parser.add_argument("--monitor-interval-ms", type=int, default=100)
    parser.add_argument(
        "--instrumentation-control-repeats", type=int, default=3,
        help="extra L4/c64 repeats with new instrumentation disabled",
    )
    parser.add_argument("--timeout", type=float, default=2 * 60 * 60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.output_tokens_per_client < 2:
        parser.error("repeats must be positive and output length >= 2")
    if any(scale not in (1, 2, 4, 8) for scale in args.scales):
        parser.error("scales must be among 1,2,4,8")

    journal_path = (
        REPO_ROOT / "simulate/results/reviewer-grade"
        / f"{args.run_prefix}-hardware-journal.json"
    )
    journal: dict[str, Any] = {
        "schema_version": 1,
        "run_prefix": args.run_prefix,
        "matrix": {
            "scales": args.scales,
            "clients": args.clients,
            "deployments": args.deployments,
            "repeats": args.repeats,
            "output_tokens_per_client": args.output_tokens_per_client,
        },
        "instrumentation": {
            "metrics_events": "transport,transfer,sync,batch",
            "sample_every": args.sample_every,
            "batch_sample_every": 1,
            "gpu_monitor_interval_ms": args.monitor_interval_ms,
            "control_repeats": args.instrumentation_control_repeats,
        },
        "runs": [],
    }
    env = os.environ.copy()
    env["DROFELL_METRICS_SAMPLE_EVERY_BATCH"] = "1"

    for scale in args.scales:
        for clients in args.clients:
            for deployment in args.deployments:
                for repeat in range(args.repeats):
                    run_id = (
                        f"{args.run_prefix}-L{scale}-c{clients}-"
                        f"{deployment}-r{repeat}"
                    )
                    result_root = EXPERIMENT_ROOT / "results" / run_id
                    case_dir = (
                        result_root / f"scale-{scale}" / f"clients-{clients}"
                        / deployment
                    )
                    command = [
                        str(args.python), "-m", "exps.infer_scale.run", "run",
                        "--python", str(args.python),
                        "--run-id", run_id,
                        "--clients", str(clients),
                        "--scales", str(scale),
                        "--deployment", deployment,
                        "--total-tokens",
                        str(clients * args.output_tokens_per_client),
                        "--tokens-per-client-cap",
                        str(args.output_tokens_per_client),
                        "--timeout", str(args.timeout),
                        "--resume",
                        "--metrics-events", "transport,transfer,sync,batch",
                        "--metrics-sample-every", str(args.sample_every),
                        "--monitor-interval-ms", str(args.monitor_interval_ms),
                    ]
                    record = {
                        "run_id": run_id,
                        "scale": scale,
                        "clients": clients,
                        "deployment": deployment,
                        "repeat": repeat,
                        "command": command,
                        "result_root": str(result_root),
                        "case_dir": str(case_dir),
                    }
                    if args.dry_run:
                        record["status"] = "dry_run"
                        print(" ".join(command))
                    elif (case_dir / "case.json").exists() and not args.force:
                        record["status"] = "existing"
                        print(f"skip existing {run_id}")
                    else:
                        started = time.monotonic()
                        print(f"run {run_id}", flush=True)
                        completed = subprocess.run(
                            command,
                            cwd=REPO_ROOT,
                            env=env,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                        record["wall_seconds"] = time.monotonic() - started
                        record["returncode"] = completed.returncode
                        record["log_tail"] = completed.stdout.splitlines()[-80:]
                        record["status"] = (
                            "complete" if completed.returncode == 0 else "failed"
                        )
                        if completed.returncode != 0:
                            journal["runs"].append(record)
                            write_json(journal_path, journal)
                            print(completed.stdout[-8000:], file=sys.stderr)
                            raise SystemExit(completed.returncode)
                    journal["runs"].append(record)
                    write_json(journal_path, journal)

    # Quantify observer effect on one representative saturated cell.
    control_scale = 4
    control_clients = 64
    control_deployment = "combined-request-agent"
    for repeat in range(args.instrumentation_control_repeats):
        run_id = f"{args.run_prefix}-control-L4-c64-r{repeat}"
        result_root = EXPERIMENT_ROOT / "results" / run_id
        case_dir = (
            result_root / "scale-4" / "clients-64"
            / control_deployment
        )
        command = [
            str(args.python), "-m", "exps.infer_scale.run", "run",
            "--python", str(args.python), "--run-id", run_id,
            "--clients", str(control_clients),
            "--scales", str(control_scale),
            "--deployment", control_deployment,
            "--total-tokens",
            str(control_clients * args.output_tokens_per_client),
            "--tokens-per-client-cap",
            str(args.output_tokens_per_client),
            "--timeout", str(args.timeout),
            "--resume",
            "--monitor-interval-ms", str(args.monitor_interval_ms),
        ]
        record = {
            "run_id": run_id,
            "scale": control_scale,
            "clients": control_clients,
            "deployment": control_deployment,
            "repeat": repeat,
            "instrumentation_control": True,
            "command": command,
            "result_root": str(result_root),
            "case_dir": str(case_dir),
        }
        if args.dry_run:
            record["status"] = "dry_run"
            print(" ".join(command))
        elif (case_dir / "case.json").exists() and not args.force:
            record["status"] = "existing"
            print(f"skip existing {run_id}")
        else:
            control_env = env.copy()
            control_env.pop("DROFELL_METRICS_SAMPLE_EVERY_BATCH", None)
            started = time.monotonic()
            print(f"run {run_id}", flush=True)
            completed = subprocess.run(
                command, cwd=REPO_ROOT, env=control_env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                check=False,
            )
            record["wall_seconds"] = time.monotonic() - started
            record["returncode"] = completed.returncode
            record["log_tail"] = completed.stdout.splitlines()[-80:]
            record["status"] = (
                "complete" if completed.returncode == 0 else "failed"
            )
            if completed.returncode != 0:
                journal["runs"].append(record)
                write_json(journal_path, journal)
                print(completed.stdout[-8000:], file=sys.stderr)
                raise SystemExit(completed.returncode)
        journal["runs"].append(record)
        write_json(journal_path, journal)

    print(f"wrote {journal_path}")
    print(f"runs: {len(journal['runs'])}")


if __name__ == "__main__":
    main()
