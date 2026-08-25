#!/usr/bin/env python3
"""Run a 20-update, two-recovery PEFT state-equivalence reproducer."""

from __future__ import annotations

import sys

from run import Experiment, parse_args, utc_run_id


def main() -> None:
    # 160 minibatches with accumulation=8 produce optimizer versions 0..19.
    # Wall-time failures force A -> B -> A while tasks are in flight. Exact
    # checkpoint fingerprints expose divergence before FP16 loss does.
    defaults = [
        "--run-id", f"peft-repeat-recovery-mini-{utc_run_id()}",
        "--experiment-profile", "repeat-recovery-mini",
        "--port", "7272",
        "--repeats", "1",
        "--drop-versions",
        "--drop-times-s", "10", "20",
        "--recovery-latencies-ms", "0",
        "--samples", "160",
        "--minibatch-size", "1",
        "--grad-accum-steps", "8",
        "--submission-interval-ms", "250",
        "--ckpt-period", "2",
        "--learning-rate", "1e-4",
        "--optimizer-eps", "1e-4",
        "--dataset-query-tokens", "64",
        "--dataset-response-tokens", "64",
        "--pytorch-math-attention",
        "--state-fingerprints",
        "--case", "normal",
        "--case", "recover",
    ]
    args = parse_args(
        [*defaults, *sys.argv[1:]],
        default_port=7272,
        default_repeats=1,
        default_drop_versions=(),
        default_drop_times_s=(10.0, 20.0),
        default_latencies_ms=(0.0,),
        default_samples=160,
        experiment_profile="repeat-recovery-mini",
    )
    experiment = Experiment(args)
    experiment.execute()
    print(f"completed: {experiment.result_root}")


if __name__ == "__main__":
    main()
