#!/usr/bin/env python3
"""Run the zero-delay, one-recovery, Re-Prefill multi-path experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import run as base_run


CLEAN_NETWORK_NAMES = {
    "ordinary": "recover-analysis-clean-ordinary",
    "backbone": "recover-analysis-clean-backbone",
}

CLEAN_DEALERS = {
    "ordinary": (
        base_run.DealerPlacement("T0-clean-ordinary-a", "0,1,2,3", 0, "cuda:0"),
        base_run.DealerPlacement("T1-clean-ordinary-b", "0,1,2,3", 1, "cuda:1"),
        base_run.DealerPlacement("T2-clean-ordinary-c", "0,1,2,3", 2, "cuda:0"),
        base_run.DealerPlacement("T3-clean-ordinary-d", "0,1,2,3", 3, "cuda:1"),
    ),
    "backbone": (
        base_run.DealerPlacement("T0-clean-backbone-a", "0,1,2,3", 0, "cuda:0"),
        base_run.DealerPlacement("T1-clean-backbone-b", "0,1,2,3", 1, "cuda:1"),
        base_run.DealerPlacement("T2-clean-backbone-c", "0,1,2,3", 2, "cuda:0"),
        base_run.DealerPlacement("T3-clean-backbone-d", "0,1,2,3", 3, "cuda:1"),
    ),
}

CLEAN_CASES = tuple(
    base_run.Case(topology=topology, failure=failure, retrieve=False)
    for topology in ("ordinary", "backbone")
    for failure in (False, True)
)


class CleanExperiment(base_run.Experiment):
    def __init__(self, **kwargs) -> None:
        super().__init__(standby_latency_ms=0.0, **kwargs)
        self.network_names = CLEAN_NETWORK_NAMES
        self.dealers = CLEAN_DEALERS
        self.cases = CLEAN_CASES
        self.network_paths = {
            topology: self.result_root / "networks" / f"{topology}-clean-8x4.json"
            for topology in CLEAN_NETWORK_NAMES
        }

    def failure_plan(self, topology: str) -> dict[str, list[int]]:
        return {}

    def standby_dealers(self, topology: str) -> list[str]:
        return []

    def inner_dealers(self, topology: str) -> list[str]:
        if topology != "backbone":
            return []
        return [dealer.name for dealer in self.dealers[topology]]

    def num_paths(self, topology: str) -> int:
        return 2 if topology == "ordinary" else 1

    def extra_environment(
        self,
        case: base_run.Case,
        repeat: int,
    ) -> dict[str, str]:
        prefix = f"clean-{case.topology}"
        replacement_map = {
            f"T0-{prefix}-a": f"T2-{prefix}-c",
            f"T1-{prefix}-b": f"T3-{prefix}-d",
        }
        environment = {
            "RECOVER_MULTI_REPLACEMENT_MAP": json.dumps(
                replacement_map, separators=(",", ":")
            ),
        }
        if case.topology == "backbone":
            environment["RECOVER_MULTI_INITIAL_BACKBONE_DEALERS"] = json.dumps([
                "T0-clean-backbone-a",
                "T1-clean-backbone-b",
            ])
        if case.failure:
            environment.update({
                "RECOVER_MULTI_ACTIVE_WINNER_DROP_STEP": str(self.drop_steps[0]),
                "RECOVER_MULTI_ACTIVE_WINNER_DROP_RELAY": (
                    "T0-backbone" if case.topology == "backbone" else ""
                ),
            })
        return environment

    @property
    def parameters(self) -> dict:
        return {
            "experiment_profile": "clean_reprefill_single_recovery_zero_delay",
            "model": base_run.MODEL,
            "prompt": base_run.PROMPT,
            "max_tokens": self.max_tokens,
            "warmup_tokens": 32,
            "parallel_sequences": 1,
            "outer_num_paths": {"ordinary": 2, "backbone": 1},
            "backbone_inner_paths": 2,
            "drop_steps": list(self.drop_steps),
            "repeats": self.repeats,
            "selected_cases": list(self.selected_cases),
            "recovery_method": "Re-Prefill",
            "recovery_replay": "chunk",
            "failure_model": "agent_observed_active_winner_teardown",
            "artificial_network_latency_ms": 0.0,
            "temperature": 0.0,
            "ignore_eos": True,
            "decoder_backend": "marlin",
            "cuda_graph": True,
            "topologies": {
                "ordinary": {
                    "path_location": "Agent outer graph",
                    "recovery_scope": "agent",
                },
                "backbone": {
                    "path_location": "Backbone internal graph",
                    "recovery_scope": "backbone",
                },
            },
        }

    def network_command(self, topology: str) -> list[str]:
        command = super().network_command(topology)
        if topology == "backbone":
            command.extend((
                "--backbone-process", "4",
                "--backbone-search-redundancy", "3",
                "--backbone-path-redundancy", "1",
            ))
        return command

    def analyze(self) -> None:
        command = [
            str(self.python),
            str(base_run.EXP_ROOT / "analyze_clean.py"),
            str(self.result_root),
        ]
        log_path = self.log_root / "analysis.log"
        with log_path.open("wb") as file:
            completed = base_run.subprocess.run(
                command,
                cwd=base_run.REPO_ROOT,
                stdout=file,
                stderr=base_run.subprocess.STDOUT,
                check=False,
            )
        self.record_command("analyze-clean", command, log_path, completed.returncode)
        if completed.returncode != 0:
            raise base_run.ExperimentError(
                f"analysis failed:\n{base_run.read_text(log_path)[-16000:]}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=base_run.DEFAULT_PYTHON)
    parser.add_argument(
        "--run-id",
        default=f"multipath-clean-{base_run.utc_run_id()}",
    )
    parser.add_argument("--port", type=int, default=7331)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--drop-step", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.name for case in CLEAN_CASES),
        dest="cases",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.drop_step < 2 or args.max_tokens <= args.drop_step + 8:
        parser.error("drop step must leave a previous winner and 8 later outputs")
    if args.cases is None:
        args.cases = [case.name for case in CLEAN_CASES]
    if len(set(args.cases)) != len(args.cases):
        parser.error("--case values must be unique")
    return args


def main() -> None:
    args = parse_args()
    experiment = CleanExperiment(
        python=args.python,
        run_id=args.run_id,
        port=args.port,
        repeats=args.repeats,
        drop_steps=(args.drop_step,),
        max_tokens=args.max_tokens,
        selected_cases=tuple(args.cases),
    )
    experiment.execute()
    print(f"completed: {experiment.result_root}")


if __name__ == "__main__":
    main()
