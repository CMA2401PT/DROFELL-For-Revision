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
