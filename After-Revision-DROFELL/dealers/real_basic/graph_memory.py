"""Fatal configuration errors for CUDA graph memory."""

from __future__ import annotations

from typing import Any, Mapping, NoReturn

from utils.format import format_size


class GraphMemoryConfigurationError(SystemExit):
    """Stop the process when CUDA graph memory cannot honor its contract."""


def require_graph_budget(graph_budget: int) -> None:
    if graph_budget > 0:
        return
    raise GraphMemoryConfigurationError(
        "CUDA graph capture requires a positive graph_budget; adjust the "
        "memory configuration before enabling CUDA graphs"
    )


def enforce_graph_budget(
    graph_budget: int,
    active_graphs: Mapping[Any, int],
    *,
    captured_key: Any,
) -> int:
    """Return active graph bytes, or terminate on a hard-budget violation."""
    active_bytes = sum(active_graphs.values())
    if active_bytes <= graph_budget:
        return active_bytes
    graph_sizes = ", ".join(
        f"{key!r}={size}" for key, size in active_graphs.items()
    )
    raise GraphMemoryConfigurationError(
        "CUDA graph memory budget exceeded after capture: "
        f"captured={captured_key!r}, active={format_size(active_bytes)}, "
        f"graph_budget={format_size(graph_budget)}, "
        f"required_minimum={active_bytes} bytes, active_graphs=[{graph_sizes}]. "
        "Increase graph_budget (and reserved/root memory as needed); captured "
        "graphs will not be evicted to recover from a budget violation"
    )


def raise_graph_capture_oom(
    *,
    graph_kind: str,
    binding: str,
    graph_budget: int,
    cause: BaseException,
) -> NoReturn:
    error = GraphMemoryConfigurationError(
        "CUDA graph capture ran out of device memory: "
        f"kind={graph_kind}, binding={binding!r}, "
        f"graph_budget={format_size(graph_budget)}. Adjust graph_budget and "
        "reserved/root memory, or reduce the configured graph shapes"
    )
    raise error from cause
