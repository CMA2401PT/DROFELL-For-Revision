from typing import Tuple


def block_eviction_priority(
    *,
    generation: int,
    next_generation: int,
    readonly_level: int,
    active: bool,
    engine_desc: str,
    preferred_engine_desc: str,
    block_idx: int,
) -> Tuple[bool, int, str, int]:
    """Return an ascending key for the partitioned block eviction score.

    ``next_generation`` is G and every block generation is delta(g) < G.
    The positive partition encoding is delta(g) + G * readonly_level; the
    negative value below lets Python's ascending sort select its maximum.
    Active blocks belong to pending operations and are placed after every
    inactive candidate.
    """
    if next_generation <= 0:
        raise ValueError("next_generation must be positive")
    if generation < 0 or generation >= next_generation:
        raise ValueError(
            f"block generation must satisfy 0 <= delta(g) < G: "
            f"delta(g)={generation}, G={next_generation}"
        )
    if readonly_level not in (0, 1, 2):
        raise ValueError(f"unsupported readonly level: {readonly_level}")
    partitioned_priority = generation + next_generation * readonly_level
    engine_order = "" if engine_desc == preferred_engine_desc else engine_desc
    return (active, -partitioned_priority, engine_order, block_idx)
