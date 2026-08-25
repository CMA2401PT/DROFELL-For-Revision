"""Simplified multi-path selection for DROSIM.

WHY DROSIM DOES NOT USE THE PROTOTYPE'S SELECTOR
------------------------------------------------
Path *selection strategy* is explicitly out of scope for the paper. The
contribution is the multi-path / redundancy / recovery **capability**: that a
logical task can advance along several paths, that redundant arrivals are
absorbed by canon, and that a failed node can be rebuilt from shadow state.
How the paths are chosen is a scheduling algorithm layered on top, which the
paper deliberately leaves open (see simulate/docs/11).

The prototype therefore ships a coarse placeholder: `gen_lowest_energy_paths`
runs a DP over switch count and returns EVERY path tied at the minimum, which is
(r+1)^L in the worst case (measured in probes/p13: 65 536 tied paths at L=16,
r=2; beyond a 4 GiB budget at L=32). Its consumer immediately collapses that set
to a keypoint list whose size is linear in L, so the exponential part contributes
nothing to the result.

Since the simulator's job is to measure how the ORCHESTRATION scales -- not how
that placeholder scales -- DROSIM substitutes the selector below: greedy,
node-disjoint, and linear in segment count. Any selector that yields the
requested number of disjoint paths is adequate for the properties under study.

GUARANTEES
----------
* Deterministic: same input -> same output, no RNG.
* O(L * r) time, no tie-set enumeration.
* Prefers to stay on the previous segment's provider, preserving the "few
  switches" preference that motivated the prototype's DP.
* Paths are node-disjoint where the topology allows, and the shortfall is
  reported rather than silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

# A provider assignment: one provider name per model segment.
Path = list[str]
# A keypoint: (dealer, first segment index, segment count).
Keypoint = tuple[str, int, int]


@dataclass
class Selection:
    """The chosen paths plus what the topology could not satisfy."""

    paths: list[Path]
    requested: int
    keypoints: list[Keypoint]
    disjoint: bool
    shortfall_reason: Optional[str] = None

    @property
    def achieved(self) -> int:
        return len(self.paths)


def path_to_keypoints(path: Path) -> list[Keypoint]:
    """Collapse consecutive same-provider segments into (dealer, start, count)."""
    if not path:
        return []
    keypoints: list[Keypoint] = []
    current = path[0]
    start = 0
    for index in range(1, len(path) + 1):
        if index == len(path) or path[index] != current:
            keypoints.append((current, start, index - start))
            if index < len(path):
                current = path[index]
                start = index
    return keypoints


def paths_to_keypoints(paths: Iterable[Path]) -> list[Keypoint]:
    """Union of each path's keypoints, deduplicated and ordered."""
    collected: set[Keypoint] = set()
    for path in paths:
        collected.update(path_to_keypoints(path))
    return sorted(collected)


def select_paths(
    providers: list[list[str]],
    redundancy: int = 1,
    allow_reuse: bool = True,
) -> Selection:
    """Choose `redundancy` node-disjoint paths through `providers`.

    `providers[i]` lists the dealers holding model segment `i`.

    Greedy construction: for each path, walk the segments in order and take the
    first provider that (a) has not been used by an earlier path, preferring (b)
    the provider used for the previous segment so that switches stay rare. When
    disjointness cannot be met, fall back to reusing providers if `allow_reuse`
    is set, and record why.
    """
    if not providers or any(len(group) == 0 for group in providers):
        return Selection([], redundancy, [], True, "a segment has no provider")

    used: set[str] = set()
    paths: list[Path] = []
    disjoint = True
    reason: Optional[str] = None

    for _ in range(redundancy):
        path: Path = []
        previous: Optional[str] = None
        for group in providers:
            fresh = [name for name in group if name not in used]
            pool = fresh
            if not pool:
                # every holder of this segment is already committed
                if not allow_reuse:
                    reason = "cannot build a disjoint path: providers exhausted"
                    break
                disjoint = False
                reason = "providers exhausted; reused a holder to keep the path whole"
                pool = list(group)
            # prefer continuing on the previous segment's provider
            choice = previous if previous in pool else pool[0]
            path.append(choice)
            previous = choice
        if len(path) != len(providers):
            break
        used.update(path)
        if path in paths:
            reason = reason or "topology cannot yield another distinct path"
            break
        paths.append(path)

    if len(paths) < redundancy and reason is None:
        reason = "topology cannot yield the requested number of paths"

    return Selection(
        paths=paths,
        requested=redundancy,
        keypoints=paths_to_keypoints(paths),
        disjoint=disjoint,
        shortfall_reason=reason,
    )


def logical_nodes(providers: list[list[str]]) -> int:
    """Distinct dealers appearing in a topology."""
    return len({name for group in providers for name in group})
