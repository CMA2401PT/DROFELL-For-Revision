from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import extend_asyncio as asyncio

from dealers.common.brain import ListWithHook
from .clock import VirtualClock
from .cost import CostModel

# The 8 infer-path queues, measured in probes/p10b.
INFER_QUEUES = (
    "operates_create_decoders_infer_binding",
    "operates_do_prefill",
    "operates_do_generating",
    "operates_get_blocks",
    "operates_put_blocks",
    "operates_put_user_model",
    "operates_infer_tensor_to_embedding_tokens",
    "operates_hidden_state_to_sampling_output",
)
# queue name -> cost-model operation key
QUEUE_COST_KEY = {
    "operates_do_generating": "decoder_generating",
    "operates_do_prefill": "decoder_prefill",
    "operates_infer_tensor_to_embedding_tokens": "infer_head",
    "operates_hidden_state_to_sampling_output": "infer_tail",
}
# Queues whose work is bookkeeping rather than GPU compute. They still cost a
# little time but do not occupy the device.
NON_DEVICE_QUEUES = {
    "operates_create_decoders_infer_binding",
    "operates_put_user_model",
    "operates_get_blocks",
    "operates_put_blocks",
}
BOOKKEEPING_NS = 50_000  # 50 us, order of the measured dealer-dispatch overhead


@dataclass
class DeviceState:
    """One simulated accelerator. Operations on it serialise."""

    device_id: str
    memory_bytes: int
    free_at_ns: int = 0
    busy_ns: int = 0
    operations: int = 0


@dataclass
class BrainStats:
    operations: int = 0
    by_queue: dict[str, int] = field(default_factory=dict)
    merged_batches: dict[str, list[int]] = field(default_factory=dict)
    device_busy_ns: int = 0

    def record(self, queue: str, batch: int) -> None:
        self.operations += 1
        self.by_queue[queue] = self.by_queue.get(queue, 0) + 1
        self.merged_batches.setdefault(queue, []).append(batch)


class SimBrain:
    """Operate-queue scheduler over the real memory twin."""

    def __init__(
        self,
        node_name: str,
        clock: VirtualClock,
        cost: CostModel,
        device: DeviceState,
        max_concurrency: int = 8,
        twin: Optional[Any] = None,
    ) -> None:
        self.node_name = node_name
        self.clock = clock
        self.cost = cost
        self.device = device
        self.max_concurrency = max_concurrency
        self.twin = twin  # real MemoryBrain, or None when memory is not modelled
        self.stats = BrainStats()
        self._closed = False

        for name in INFER_QUEUES:
            setattr(self, name, ListWithHook())

        # `Brain` exposes these for the dealer's barrier accounting; in
        # simulation the barrier is a no-op counter because the scheduler already
        # serialises per device.
        self._barriers: set[str] = set()
        self._wake: Optional[asyncio.Future[None]] = None

        # Wake on arrival, never on a timeout. Under a virtual clock a polling
        # loop does not idle -- it advances the clock and spins forever, so a
        # `wait_for(..., timeout=...)` fallback would hang the simulation. The
        # real Brain has the same structure (`Brain.wait_new_operate` woken from
        # the queue hook), so this mirrors production rather than inventing a
        # scheme.
        for name in INFER_QUEUES:
            getattr(self, name).hook = self._on_arrival

        self._loop_task = asyncio.create_task(self.loop())

    def _on_arrival(self, operate: Any) -> Any:
        self._awake()
        return operate

    def _awake(self) -> None:
        if self._wake is not None and not self._wake.done():
            fut, self._wake = self._wake, None
            fut.set_result(None)

    # -- barrier interface used by dealer_node / compute_node --------------
    def add_barrier(self, context_id: str) -> None:
        self._barriers.add(context_id)

    def remove_barrier(self, context_id: str) -> None:
        self._barriers.discard(context_id)

    # -- memory interface -------------------------------------------------
    @property
    def memory(self) -> tuple[int, int]:
        """(total, base-model) bytes, as the dealer announces to the directory."""
        if self.twin is not None:
            return self.device.memory_bytes, int(self.twin.total)
        return self.device.memory_bytes, 0

    def query_extra_info(self, key: str) -> Any:
        return None

    # -- scheduling -------------------------------------------------------
    def _pending_queues(self) -> list[str]:
        return [
            name for name in INFER_QUEUES if len(getattr(self, name)) > 0
        ]

    def _take_merged(self, queue_name: str) -> list[Any]:
        """Pop up to `max_concurrency` operations that can merge.

        Mirrors `dealers/memory_schedule/brain.py::select_multi_with_group`:
        operations merge only while a grouping property matches. Here the
        property is the binding, which is what the production hook compares.
        """
        queue: ListWithHook[Any] = getattr(self, queue_name)
        if len(queue) == 0:
            return []
        first = queue.pop(0)
        selected = [first]
        group_key = getattr(first, "binding", None)
        while len(selected) < self.max_concurrency and len(queue) > 0:
            candidate = queue.list[0]
            if getattr(candidate, "binding", None) != group_key:
                break
            selected.append(queue.pop(0))
        return selected

    async def loop(self) -> None:
        while not self._closed:
            pending = self._pending_queues()
            if not pending:
                # park until a shim appends something (see _on_arrival)
                self._wake = asyncio.Future()
                await self._wake
                continue
            for queue_name in pending:
                batch = self._take_merged(queue_name)
                if batch:
                    await self._execute(queue_name, batch)

    async def _execute(self, queue_name: str, batch: list[Any]) -> None:
        cost_key = QUEUE_COST_KEY.get(queue_name)
        if cost_key is None:
            duration_ns = BOOKKEEPING_NS
        else:
            duration_ns = self.cost.duration_ns(cost_key, len(batch))

        self.stats.record(queue_name, len(batch))

        if queue_name in NON_DEVICE_QUEUES:
            await asyncio.sleep(duration_ns / 1e9)
        else:
            # the device is a serial resource: wait for it, then hold it
            start = max(self.device.free_at_ns, self.clock.now_ns)
            wait_ns = start - self.clock.now_ns
            if wait_ns > 0:
                await asyncio.sleep(wait_ns / 1e9)
            self.device.free_at_ns = self.clock.now_ns + duration_ns
            self.device.busy_ns += duration_ns
            self.device.operations += 1
            self.stats.device_busy_ns += duration_ns
            await asyncio.sleep(duration_ns / 1e9)

        for operate in batch:
            self._complete(operate, queue_name)

    def _complete(self, operate: Any, queue_name: str) -> None:
        """Hand a result back to the awaiting shim.

        Results are deliberately shaped by the caller-supplied `result_factory`
        so that SimBrain does not need to know the semantics of every operate.
        """
        factory = getattr(operate, "_drosim_result", None)
        result = factory() if callable(factory) else None
        if not operate.fut.done():
            operate.set_result(result)

    def close(self) -> None:
        self._closed = True
        self._awake()
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()

    # -- explicit failure for unmodelled paths ----------------------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith("operates_"):
            raise NotImplementedError(
                f"SimBrain does not model the queue {name!r}. The simulated "
                f"scenario reached a code path outside the infer set measured in "
                f"probes/p10b, or the production Brain interface changed."
            )
        raise AttributeError(
            f"SimBrain has no attribute {name!r}: this path is not modelled. "
            f"Returning a placeholder would silently corrupt results, so this "
            f"raises instead (docs/02 §4.4)."
        )

    def report(self) -> dict[str, Any]:
        merged = {
            queue: (sum(sizes) / len(sizes)) if sizes else 0.0
            for queue, sizes in self.stats.merged_batches.items()
        }
        return {
            "node": self.node_name,
            "operations": self.stats.operations,
            "operations_by_queue": dict(self.stats.by_queue),
            "mean_merged_batch": merged,
            "device_busy_ns": self.stats.device_busy_ns,
            "device_operations": self.device.operations,
            "twin_total_bytes": int(self.twin.total) if self.twin else None,
        }
