#!/usr/bin/env python3
"""Aggregate a DROFELL inference ``result.json`` token timeline."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence


NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class InferenceMetrics:
    clients: int
    tokens_by_client: dict[int, int]
    total_tokens: int
    first_timestamp_ns: int
    last_timestamp_ns: int
    duration_seconds: float
    tokens_per_second: float


def aggregate_records(records: Sequence[Sequence[Any]]) -> InferenceMetrics:
    """Compute aggregate timed-decode throughput across all clients.

    The formula intentionally matches ``compute_avg_speed`` in
    ``scripts/online/speed/analysis_utils.py``: every entry in ``result.json``
    is one generated token, and elapsed time spans the first through last
    recorded completion timestamp.
    """
    if len(records) < 2:
        raise ValueError("result.json must contain at least two token records")

    parsed: list[tuple[int, int]] = []
    for index, record in enumerate(records):
        if not isinstance(record, (list, tuple)) or len(record) != 2:
            raise ValueError(
                f"record {index} must be [client_id, timestamp_ns]"
            )
        client_id, timestamp_ns = record
        if (
            not isinstance(client_id, int)
            or isinstance(client_id, bool)
            or client_id < 0
        ):
            raise ValueError(f"record {index} has an invalid client id")
        if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
            raise ValueError(f"record {index} has an invalid timestamp")
        parsed.append((client_id, timestamp_ns))

    timestamps = [timestamp_ns for _, timestamp_ns in parsed]
    if any(
        current < previous
        for previous, current in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("timestamps must be globally nondecreasing")

    elapsed_ns = timestamps[-1] - timestamps[0]
    if elapsed_ns <= 0:
        raise ValueError("first and last timestamps must define a positive duration")

    token_counts = Counter(client_id for client_id, _ in parsed)
    total_tokens = len(parsed)
    duration_seconds = elapsed_ns / NANOSECONDS_PER_SECOND
    return InferenceMetrics(
        clients=len(token_counts),
        tokens_by_client=dict(sorted(token_counts.items())),
        total_tokens=total_tokens,
        first_timestamp_ns=timestamps[0],
        last_timestamp_ns=timestamps[-1],
        duration_seconds=duration_seconds,
        tokens_per_second=total_tokens / duration_seconds,
    )


def analyze_file(path: Path) -> InferenceMetrics:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"result file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(data, list):
        raise ValueError("result.json root must be a list")
    return aggregate_records(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="path to an inference result.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = analyze_file(args.result)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(metrics), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
