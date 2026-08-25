#!/usr/bin/env python3
"""Run one isolated unreliable-node matrix cell.

This entry point is used after a cell-level liveness failure so a failed
topology/probability cannot prevent independent cells from being measured.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import run as matrix


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=matrix.DEFAULT_PYTHON)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--topology", required=True,
        choices=tuple(item.name for item in matrix.TOPOLOGIES),
    )
    parser.add_argument(
        "--deployment", required=True,
        choices=("combined-request-agent", "persistent-agent"),
    )
    parser.add_argument("--probability", type=float, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--port", type=int, default=7255)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--stall-timeout", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=matrix.FORMAL_MAX_TOKENS)
    parser.add_argument("--deterministic-drop-dealer")
    parser.add_argument("--deterministic-drop-step", type=int)
    parser.add_argument("--deterministic-drop-plan")
    args = parser.parse_args()
    if args.deployment == "persistent-agent" and args.probability != 0.0:
        parser.error("persistent-agent is valid only for probability 0")
    if args.probability not in matrix.FAILURE_PROBABILITIES:
        parser.error(
            f"probability must be one of {matrix.FAILURE_PROBABILITIES}"
        )
    if (
        args.repeats <= 0
        or args.timeout <= 0
        or args.stall_timeout <= 0
        or args.max_tokens <= 1
    ):
        parser.error("repeats and timeouts must be positive")
    if (args.deterministic_drop_dealer is None) != \
            (args.deterministic_drop_step is None):
        parser.error(
            "deterministic drop dealer and step must be provided together"
        )
    if args.deterministic_drop_step is not None \
            and args.deterministic_drop_step < 1:
        parser.error("deterministic drop step must be positive")
    if args.deterministic_drop_plan is not None and \
            args.deterministic_drop_dealer is not None:
        parser.error("use either deterministic drop plan or dealer/step")
    return args


def main() -> None:
    args = parse_args()
    selected = next(
        item for item in matrix.TOPOLOGIES if item.name == args.topology
    )
    dealer_names = {dealer for dealer, _ in selected.dealers}
    deterministic_plan = {}
    if args.deterministic_drop_plan is not None:
        deterministic_plan = json.loads(args.deterministic_drop_plan)
        if not isinstance(deterministic_plan, dict):
            raise TypeError("deterministic drop plan must be a JSON object")
        unknown_dealers = set(deterministic_plan) - dealer_names
        if unknown_dealers:
            raise ValueError(
                f"deterministic drop Dealers are not in {args.topology}: "
                f"{sorted(unknown_dealers)}"
            )
        for dealer, steps in deterministic_plan.items():
            if (
                not isinstance(steps, list)
                or any(type(step) is not int or step < 1 for step in steps)
                or len(set(steps)) != len(steps)
            ):
                raise ValueError(
                    f"invalid deterministic steps for {dealer}: {steps!r}"
                )
    if (
        args.deterministic_drop_dealer is not None
        and args.deterministic_drop_dealer not in dealer_names
    ):
        raise ValueError(
            f"deterministic drop Dealer is not in {args.topology}: "
            f"{args.deterministic_drop_dealer}"
        )
    selected = replace(selected, probabilities=(args.probability,))
    matrix.TOPOLOGIES = (selected,)
    matrix.REPEATS = args.repeats
    matrix.FORMAL_MAX_TOKENS = args.max_tokens
    matrix.FORMAL_TIMEOUT = args.timeout
    matrix.FORMAL_STALL_TIMEOUT = args.stall_timeout
    if args.deterministic_drop_dealer is not None:
        os.environ["DROFELL_DETERMINISTIC_DROP_DEALER"] = \
            args.deterministic_drop_dealer
        os.environ["DROFELL_DETERMINISTIC_DROP_STEP"] = \
            str(args.deterministic_drop_step)
    if deterministic_plan:
        os.environ["DROFELL_DETERMINISTIC_DROP_PLAN"] = json.dumps(
            deterministic_plan,
            separators=(",", ":"),
        )

    python = args.python.expanduser()
    if not python.is_absolute():
        python = matrix.REPO_ROOT / python
    experiment = matrix.Experiment(
        python=python,
        run_id=args.run_id,
        port=args.port,
        repeats=args.repeats,
        topologies=(selected,),
        include_persistent_p0=False,
        capture_metrics=False,
        skip_smoke=True,
    )
    experiment.metrics_path = experiment.result_dir / "runtime_metrics.jsonl"
    experiment.initialize()
    experiment.environment["cell"] = {
        "topology": args.topology,
        "deployment": args.deployment,
        "probability": args.probability,
        "repeats": args.repeats,
        "timeout_seconds": args.timeout,
        "stall_timeout_seconds": args.stall_timeout,
        "max_tokens": args.max_tokens,
        "deterministic_drop_dealer": args.deterministic_drop_dealer,
        "deterministic_drop_step": args.deterministic_drop_step,
        "deterministic_drop_plan": deterministic_plan,
    }
    experiment.flush_environment()
    try:
        networks = experiment.generate_networks()
        experiment.run_formal_network(
            topology=selected,
            network=networks[selected.name],
            deployment=args.deployment,
        )
        experiment.environment["source_fingerprint_end"] = \
            matrix.source_fingerprint()
        experiment.flush_environment()
        if (
            experiment.environment["source_fingerprint_start"]
            != experiment.environment["source_fingerprint_end"]
        ):
            raise matrix.ExperimentError(
                "runtime source changed during the experiment; refusing to "
                "generate a formal summary"
            )
        summary = matrix.summarize_run(
            experiment.result_dir,
            experiment.network_runs,
            repeats=args.repeats,
            topologies=(selected,),
            include_persistent_p0=False,
            capture_metrics=False,
        )
        matrix.write_json(experiment.result_dir / "summary.json", summary)
        matrix.write_summary_csv(
            experiment.result_dir / "summary.csv", summary["runs"]
        )
        experiment.environment["finished_utc"] = (
            datetime.now(timezone.utc).isoformat()
        )
        experiment.environment["status"] = "complete"
        experiment.flush_environment()
    except BaseException as error:
        failure = {
            "status": "failed",
            "topology": args.topology,
            "deployment": args.deployment,
            "probability": args.probability,
            "repeats": args.repeats,
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(experiment.result_dir / "failure.json", failure)
        experiment.environment["status"] = "failed"
        experiment.environment["failure"] = failure
        experiment.flush_environment()
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except matrix.ExperimentError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
