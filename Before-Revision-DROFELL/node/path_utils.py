import collections
import re
from typing import List, Optional, Tuple, Set, Union
from typing import Dict, List, Set, Tuple

from dataclasses import dataclass


def gen_lowest_energy_paths(providers_list: List[List[str]]):
    if any(len(ps) == 0 for ps in providers_list):
        return
    n = len(providers_list)

    # dynamic programming：dp[i][p] = (min_switches, all_paths)
    dp: List[Dict[str, Tuple[int, List[List[str]]]]] = [{} for _ in range(n)]

    # init step 0
    for p in providers_list[0]:
        dp[0][p] = (0, [[p]])

    # build dp
    for i in range(1, n):
        for p in providers_list[i]:
            best_switch = 2**30
            best_paths = []

            for prev_p, (prev_switch, prev_paths) in dp[i - 1].items():
                # check cost
                switch_cost = 0 if p == prev_p else 1
                total_cost = prev_switch + switch_cost

                if total_cost < best_switch:
                    best_switch = total_cost
                    best_paths = [path + [p] for path in prev_paths]
                elif total_cost == best_switch:
                    best_paths.extend(path + [p] for path in prev_paths)

            dp[i][p] = (best_switch, best_paths)

    # find path with minimal energy (switch)
    if len([v[0] for v in dp[-1].values()]) == 0:
        raise ValueError("no path found")
    min_switch = min(v[0] for v in dp[-1].values())
    result: List[List[str]] = []
    for v in dp[-1].values():
        if v[0] == min_switch:
            result.extend(v[1])

    return result


def paths_to_edge(path: List[str]):
    results: Set[Tuple[str, int, int]] = set()
    start = 0
    current = path[0]
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != current:
            results.add((current, start, i-start))
            if i < len(path):
                current = path[i]
                start = i
    return results


def remove_overlap(intervals: Set[Tuple[str, int, int]]) -> Set[Tuple[str, int, int]]:
    out: Set[Tuple[str, int, int]] = set()
    by_node: Dict[str, List[Tuple[int, int]]] = {}
    for node, s, length in intervals:
        by_node.setdefault(node, []).append((s, s + length))

    for node, segs in by_node.items():
        if not segs:
            continue
        boundaries: Set[int] = set()
        for s, e in segs:
            boundaries.add(s)
            boundaries.add(e)
        sorted_b = sorted(boundaries)
        for i in range(len(sorted_b) - 1):
            bs, be = sorted_b[i], sorted_b[i+1]
            covered = False
            for s, e in segs:
                if s <= bs and be <= e:
                    covered = True
                    break
            if covered:
                out.add((node, bs, be - bs))
    return out


def paths_to_keypoints(paths: List[List[str]], no_overlap: bool = False) -> List[Tuple[str, int, int]]:
    keypoints: Set[Tuple[str, int, int]] = set()
    for path in paths:
        keypoints |= paths_to_edge(path)
    if no_overlap:
        keypoints = remove_overlap(keypoints)
    return sorted(keypoints)


def reduce_providers(
    providers_list: List[List[str]],
    reduces: List[str],
    allow_reuse: bool,
):
    out: List[List[str]] = []
    for (providers, reduce) in zip(providers_list, reduces):
        reduce_providers = [p for p in providers if p != reduce]
        if len(reduce_providers) == 0:
            if not allow_reuse:
                return None
            reduce_providers = [reduce]
        out.append(reduce_providers)
    return out


@dataclass
class InspectedPath:
    providers: List[List[str]]
    best: List[str]

    def reduce_providers(self, allow_reuse: bool):
        reduced_paths = reduce_providers(self.providers, self.best, allow_reuse)
        if reduced_paths is None:
            return None
        for providers, avoid in zip(reduced_paths, self.best):
            if len(providers) > 1 or providers[0] != avoid:
                return reduced_paths
        return None


def keypoints_to_paths(keypoints: List[Tuple[str, int, int]]) -> List[List[str]]:
    if not keypoints:
        return []
    max_len = max(s + l for _, s, l in keypoints)
    starts: Dict[int, List[Tuple[str, int, int]]] = {}
    for node, s, l in keypoints:
        starts.setdefault(s, []).append((node, s, l))
    memo: Dict[int, List[List[Tuple[str, int]]]] = {}

    def dfs(pos: int) -> List[List[Tuple[str, int]]]:
        if pos == max_len:
            return [[]]
        if pos in memo:
            return memo[pos]

        res: List[List[Tuple[str, int]]] = []
        if pos not in starts:
            memo[pos] = []
            return []

        for node, s, l in starts[pos]:
            for tail in dfs(pos + l):
                res.append([(node, l)] + tail)
        memo[pos] = res
        return res

    segmented_paths = dfs(0)
    full_paths: List[List[str]] = []
    for seg_seq in segmented_paths:
        path: List[str] = []
        for node, l in seg_seq:
            path.extend([node] * l)
        if len(path) == max_len:
            full_paths.append(path)

    return full_paths


@dataclass
class InspectedMultiPaths:
    fastest_paths: List[List[str]]
    keypoints: List[Tuple[str, int, int]]

    def all_possible_paths(self):
        return keypoints_to_paths(self.keypoints)


_tile_search = re.compile(r'T(\d+)')


def sort_by_tile_priority(items: List[str]):
    def priority(s: str):
        m = _tile_search.search(s)
        return int(m.group(1)) if m else float('inf')

    return sorted(items, key=priority)


def group_by_tile_priority(items: List[str]):
    groups: Dict[Union[float, int], List[str]] = collections.defaultdict(list)

    for s in items:
        m = re.search(r'tile-(\d+)', s)
        key = int(m.group(1)) if m else float('inf')
        groups[key].append(s)
    return [groups[k] for k in sorted(groups.keys())]


def fmt_inspect_path(base_models: List[str], providers: Optional[List[List[str]]], select: Optional[List[str]] = None):
    if providers is None:
        providers = [[] for _ in range(len(base_models))]
    assert len(base_models) == len(providers)
    if select is None:
        select = [""]*len(base_models)
    path_strs: List[str] = []
    for m, ps, s in zip(base_models, providers, select):
        ps = sort_by_tile_priority(ps)
        providers_str: List[str] = [
            f"√ {p}"if p == s else p
            for p in sorted(ps)
        ]
        path_strs.append(f"{m:<40}: " + ", ".join(providers_str))
    return "\n  ".join(path_strs)
