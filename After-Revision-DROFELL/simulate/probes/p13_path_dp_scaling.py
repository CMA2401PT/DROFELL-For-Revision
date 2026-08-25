#!/usr/bin/env python3
"""P13: measure the real scaling cost of the multi-path DP (`gen_lowest_energy_paths`).

WHY THIS EXISTS
---------------
The DROSIM design (docs/00 §2.2, claim 2) originally asserted that path search is
"O(L * P^2), independent of N". Measuring it refuted that claim.

`gen_lowest_energy_paths` does not return one minimum-switch path: it returns
EVERY path tied at the minimum switch count. When each of L segments has P
distinct providers, every assignment ties at L-1 switches, so the result set is
exactly P^L -- exponential in the number of model segments.

This is a genuine scalability boundary in the orchestration layer, on the live
path (`AgentControlPlane.inspect_fastest_path`, `Backbone*.inspect_fastest_path`,
and every recovery `lookup_replacement`). It is exactly the kind of limit
reviewer 2.7 asks about, so it is reported rather than smoothed over.

Real deployed topologies do NOT hit it, because providers are shared across
consecutive layers, which collapses the tie set (see the second table).

Memory is capped so a blow-up cannot destabilise the host.
"""

from __future__ import annotations

import argparse
import resource
import time

MEMORY_CAP_BYTES = 3 * 1024 ** 3


def measure(segments: int, providers_per_segment: int) -> tuple[int | None, float]:
    from node.path_utils import gen_lowest_energy_paths

    providers = [
        [f"n{layer}_{index}" for index in range(providers_per_segment)]
        for layer in range(segments)
    ]
    started = time.perf_counter_ns()
    try:
        paths = gen_lowest_energy_paths(providers)
    except MemoryError:
        return None, float("nan")
    return len(paths), (time.perf_counter_ns() - started) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-segments", type=int, default=16)
    args = parser.parse_args()

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_CAP_BYTES, MEMORY_CAP_BYTES))
    import extend_asyncio as ea

    ea.config_impl("py", False)
    from node.path_utils import gen_lowest_energy_paths, paths_to_keypoints

    print("### worst case: every segment has P DISTINCT providers")
    print("(all assignments tie at L-1 switches, so the tie set is P^L)")
    print(f"{'L':>4s} {'P=1':>12s} {'P=2':>14s} {'P=3':>16s}")
    for segments in range(4, args.max_segments + 1, 2):
        cells = []
        for providers in (1, 2, 3):
            count, millis = measure(segments, providers)
            cells.append("OOM" if count is None else f"{count}({millis:.0f}ms)")
        print(f"{segments:4d} {cells[0]:>12s} {cells[1]:>14s} {cells[2]:>16s}")

    print()
    print("### verification that the count is exactly P^L")
    for segments, providers in ((8, 2), (8, 3), (12, 2)):
        count, _ = measure(segments, providers)
        expected = providers ** segments
        print(f"  L={segments:2d} P={providers}: got {count}, P^L={expected}, "
              f"match={count == expected}")

    print()
    print("### deployed topologies (exps/unreliable_node) do NOT hit this")
    print("providers are shared across consecutive layers, collapsing the tie set")
    deployed = {
        "1_1_1": [["dealer-g0"]] * 3,
        "2_4_1": [
            ["dealer-g0"], ["dealer-g0"], ["dealer-g1"],
            ["dealer-g2"], ["dealer-g3"], ["dealer-g3"],
        ],
        "2_4_2": [
            [f"dealer-g{group}-r{replica}" for replica in range(2)]
            for group in range(4)
        ],
    }
    for name, providers in deployed.items():
        paths = gen_lowest_energy_paths(providers)
        keypoints = paths_to_keypoints(paths, no_overlap=True)
        print(f"  {name:8s} L={len(providers):2d} -> paths={len(paths):5d} "
              f"keypoints={len(keypoints)}")

    print()
    print("=> Reportable boundary: the tie set is exponential in SEGMENT COUNT")
    print("   (not in total node count N). With redundancy 2 it stays tractable")
    print("   to ~16 segments; with redundancy 3 it exhausts memory past ~12.")
    print("   docs/00 §2.2 claim 2 was corrected accordingly.")


if __name__ == "__main__":
    main()
