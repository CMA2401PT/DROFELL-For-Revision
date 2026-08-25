#!/usr/bin/env python3
"""Run the independent 240-second real-data PEFT loss/recovery experiment."""

from __future__ import annotations

import sys

from run import Experiment, parse_args, utc_run_id


SUBMISSION_WINDOW_S = 240.0
SUBMISSION_INTERVAL_MS = 250.0
MINIBATCHES = int(SUBMISSION_WINDOW_S * 1000 / SUBMISSION_INTERVAL_MS)


def main() -> None:
    defaults = [
        "--run-id", f"peft-real-long240-timed-r1-{utc_run_id()}",
        "--repeats", "1",
        "--drop-times-s", "60", "120", "180",
        "--recovery-latencies-ms", "0",
        "--samples", str(MINIBATCHES),
        "--minibatch-size", "1",
        "--grad-accum-steps", "8",
        "--submission-interval-ms", str(SUBMISSION_INTERVAL_MS),
        "--ckpt-period", "10",
        "--case", "normal",
        "--case", "recover",
    ]
    args = parse_args(
        [*defaults, *sys.argv[1:]],
        default_drop_versions=(),
        default_drop_times_s=(60.0, 120.0, 180.0),
        default_latencies_ms=(0.0,),
        default_samples=MINIBATCHES,
        default_repeats=1,
        default_port=7268,
        experiment_profile="long-loss",
    )
    experiment = Experiment(args)
    experiment.execute()
    print(f"completed: {experiment.result_root}")


if __name__ == "__main__":
    main()
