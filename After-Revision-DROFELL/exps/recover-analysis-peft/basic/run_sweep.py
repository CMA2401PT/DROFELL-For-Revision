#!/usr/bin/env python3
"""Run the short, dense symmetric-latency PEFT recovery sweep."""

from __future__ import annotations

from run import Experiment, parse_args, utc_run_id


def main() -> None:
    args = parse_args(
        default_run_id=f"peft-real-symmetric-sweep-{utc_run_id()}",
        default_port=7267,
        default_repeats=3,
        default_drop_versions=(1,),
        default_latencies_ms=(0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0),
        default_samples=32,
        experiment_profile="latency-sweep",
    )
    experiment = Experiment(args)
    experiment.execute()
    print(f"completed: {experiment.result_root}")


if __name__ == "__main__":
    main()
