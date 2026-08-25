from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

DEFAULT_TRACE = (
    "一稿返修/整理后就绪的实验/推理整流/c0-m1-d1/repeat-1/traces.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", default=DEFAULT_TRACE, type=Path)
    parser.add_argument("--top", type=int, default=16)
    args = parser.parse_args()

    traces = json.loads(args.trace.read_text())
    segments: dict[str, list[float]] = collections.defaultdict(list)
    token_count = 0
    for client in traces:
        for token in client:
            token_count += 1
            for (name_a, t_a), (name_b, t_b) in zip(token, token[1:]):
                segments[f"{name_a}->{name_b}"].append((t_b - t_a) / 1e3)

    total = sum(sum(values) for values in segments.values())
    print(f"clients={len(traces)} tokens={token_count} segments={len(segments)}")
    print(f"{'segment':34s} {'n':>6s} {'p50 us':>9s} {'p90 us':>9s} {'share%':>7s}")
    ranked = sorted(segments.items(), key=lambda item: -sum(item[1]))
    for name, values in ranked[: args.top]:
        ordered = sorted(values)
        p90 = ordered[min(int(0.9 * len(ordered)), len(ordered) - 1)]
        share = 100 * sum(values) / total if total else 0.0
        print(
            f"{name:34s} {len(values):6d} {statistics.median(ordered):9.1f} "
            f"{p90:9.1f} {share:7.1f}"
        )


if __name__ == "__main__":
    main()
