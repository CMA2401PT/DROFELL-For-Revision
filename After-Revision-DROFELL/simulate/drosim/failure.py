"""Failure injection for DROSIM, reusing the production guards verbatim.

WHY THE GUARDS MATTER MORE THAN THE MECHANISM
---------------------------------------------
`scripts/online/benchmark/network_nodes.py::_embed_failure_injection` samples the
drop probability per task event at a Dealer, but only for events satisfying THREE
conditions:

    str(task.task.seq_id).startswith("bench-")   # benchmark traffic only
    and not duplicate                            # per (seq_id, hop), once
    and not task.trace_invalid

The middle one is load-bearing for anything that compares redundancy levels. Under
redundancy r, the same logical step arrives at a Dealer r times over distinct
paths. Sampling every arrival would make the effective failure rate scale with r,
so a redundant deployment would look worse purely as an artefact of the injector.
The guard makes p mean "per non-duplicate task event", which is exactly the
definition the paper and reply letter use.

Reproducing that here, rather than writing a fresh injector, is what lets a
simulated failure-rate sweep be read next to the measured one.

WHAT IS DELIBERATELY NOT MODELLED
---------------------------------
Link partition and packet loss exist in `SimFabric` but stay OFF: reviewer points
3.7 and 2.11 are out of scope, and switching them on would silently widen what
the paper claims to have evaluated (docs/04 §3).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import extend_asyncio as asyncio
from node.dealer_node.dealer_node import DealerNode
from node.dealer_node.protocols import MultiPathInferTask
from utils.runtime_metrics import record_event


@dataclass
class DeterministicPlan:
    """A budget of deliberate kills, SHARED across every Dealer.

    Why shared: under redundancy r, each of the r holders of a segment is a
    separate Dealer with its own injector. A per-injector budget therefore fires
    once per holder, so raising redundancy silently raises the NUMBER OF FAILURES
    -- measured 2 / 4 / 8 kills at redundancy 2 / 4 / 8, which turned a
    "recovery cost vs network size" sweep into "recovery cost vs failure count"
    and produced a spurious 54% rise.

    Sharing the budget makes "one deliberate failure" mean one, so network size is
    the only variable. `remaining` caps the total.
    """

    steps: frozenset[int] = frozenset()
    remaining: int = 1
    fired: set[int] = field(default_factory=set)

    def claim(self, step: int) -> bool:
        """Consume one kill for `step`, if the plan still allows it."""
        if step not in self.steps or self.remaining <= 0 or step in self.fired:
            return False
        self.fired.add(step)
        self.remaining -= 1
        return True


@dataclass
class InjectionStats:
    events_considered: int = 0
    duplicates_skipped: int = 0
    non_benchmark_skipped: int = 0
    trace_invalid_skipped: int = 0
    drops_triggered: int = 0
    deterministic_drops: int = 0

    def report(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class FailureInjector:
    """Wraps one Dealer's `on_infer_task`, mirroring the production wrapper."""

    def __init__(
        self,
        dealer: DealerNode,
        probability: float = 0.0,
        plan: Optional[DeterministicPlan] = None,
        rng: Optional[random.Random] = None,
        stats: Optional[InjectionStats] = None,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("drop probability must be in [0, 1]")
        self.dealer = dealer
        self.probability = probability
        # Shared across injectors on purpose -- see DeterministicPlan.
        self.plan = plan if plan is not None else DeterministicPlan(frozenset(), 0)
        # Seeded per node so a scenario is reproducible: an unseeded sweep would
        # make every repeat a different experiment.
        self.rng = rng if rng is not None else random.Random(0)
        self.stats = stats if stats is not None else InjectionStats()
        self._progress: Dict[Tuple[str, int], int] = {}
        self._original = dealer.on_infer_task
        dealer.on_infer_task = self._on_infer_task  # type: ignore[assignment]

    def _on_infer_task(
        self, src: str, task: MultiPathInferTask, from_group: bool = False
    ) -> None:
        model, _, _ = task.task.navi.target(0)
        node_names = self.dealer.user_model_to_nodes.get(model, [])
        if not node_names:
            self.dealer.base.logger.error(
                f"cannot find {model} for task {task.task.unique_id}; discard"
            )
            return

        # Same three guards, in the same order, as the production injector.
        node_id = (task.task.seq_id, task.task.navi.current)
        step = task.task.step
        duplicate = step <= self._progress.get(node_id, -1)
        if not duplicate:
            self._progress[node_id] = step

        is_benchmark = str(task.task.seq_id).startswith("bench-")
        self.stats.events_considered += 1
        if not is_benchmark:
            self.stats.non_benchmark_skipped += 1
        if duplicate:
            self.stats.duplicates_skipped += 1
        if task.trace_invalid:
            self.stats.trace_invalid_skipped += 1

        eligible = is_benchmark and not duplicate and not task.trace_invalid

        for node_name in node_names:
            deterministic = eligible and self.plan.claim(step)
            should_drop = eligible and self.rng.random() < self.probability
            if deterministic:
                self.stats.deterministic_drops += 1
            if should_drop or deterministic:
                self.stats.drops_triggered += 1
                asyncio.create_task(
                    self._drop_compute_node(
                        node_name, task, model, deterministic
                    )
                )
            else:
                self._original(src, task, from_group)
                # `_original` re-dispatches to every node itself, so break out
                # rather than looping: calling it once per node would duplicate
                # the work the production handler already fans out.
                return

    async def _drop_compute_node(
        self,
        target: str,
        task: MultiPathInferTask,
        model: str,
        deterministic: bool,
    ) -> None:
        await self.dealer.wait_node_init(target)
        if target not in self.dealer.name_to_nodes:
            return
        record_event(
            "failure_injected",
            dealer_node=self.dealer.base.node_name,
            compute_node=target,
            sequence_id=task.task.seq_id,
            step=task.task.step,
            model=model,
        )
        self.dealer.base.logger.warn(
            f"trigger compute-node drop: {target}"
            + (" (deterministic)" if deterministic else "")
        )
        self.dealer.suicide_node(target)

    def restore(self) -> None:
        self.dealer.on_infer_task = self._original  # type: ignore[assignment]
