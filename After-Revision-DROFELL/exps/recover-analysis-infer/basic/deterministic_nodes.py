"""Experiment-local deterministic failure injection for basic recovery analysis.

This module deliberately lives under ``exps/recover-analysis-infer/basic``.  It
reuses the production Dealer and recovery path, but replaces the benchmark's
Bernoulli trigger with an explicit sequence of decode-step failures.
"""

from __future__ import annotations

import os
import json
import time
from typing import Dict, Tuple

import extend_asyncio as asyncio

from node.dealer_node.dealer_node import DealerNode
from node.dealer_node.protocols import MultiPathInferTask
from orbit.defines import Context
from scripts.online.common_nodes import dealer_coro
from utils.runtime_metrics import record_event


DROP_PLAN_ENV = "RECOVER_ANALYSIS_DROP_PLAN"


def _configured_drop_plan() -> Dict[str, Tuple[int, ...]]:
    raw = os.getenv(DROP_PLAN_ENV, "{}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"{DROP_PLAN_ENV} must be a JSON object")
    plan: Dict[str, Tuple[int, ...]] = {}
    for dealer, configured_steps in value.items():
        if not isinstance(dealer, str) or not dealer:
            raise ValueError(f"{DROP_PLAN_ENV} has an invalid Dealer name")
        if not isinstance(configured_steps, list) or not configured_steps:
            raise ValueError(
                f"{DROP_PLAN_ENV}[{dealer!r}] must be a non-empty list"
            )
        steps = tuple(configured_steps)
        if (
            any(type(step) is not int or step < 1 for step in steps)
            or tuple(sorted(steps)) != steps
            or len(set(steps)) != len(steps)
        ):
            raise ValueError(
                f"{DROP_PLAN_ENV}[{dealer!r}] must contain unique, "
                "increasing positive integers"
            )
        plan[dealer] = steps
    return plan


def _embed_deterministic_failure(dealer: DealerNode) -> None:
    """Delete this Dealer's active compute node at each configured step."""

    progress: Dict[Tuple[str, int], int] = {}
    drop_plan = _configured_drop_plan()
    drop_steps = set(drop_plan.get(dealer.base.node_name, tuple()))
    ordered_steps = sorted({step for steps in drop_plan.values() for step in steps})
    injected_steps: set[int] = set()
    dealer.base.tags.add("recover_analysis_deterministic_failure")

    def on_infer_task(
        src: str,
        task: MultiPathInferTask,
        from_group: bool = False,
    ) -> None:
        model, _, _ = task.task.navi.target(0)
        node_names = dealer.user_model_to_nodes.get(model, [])
        if not node_names:
            dealer.base.logger.error(
                f"cannot find {model} for task {task.task.unique_id}; discard"
            )
            return

        metrics = None
        if task.detail_trace is not None:
            metrics = [("dealer-recv", time.time_ns())]

        node_id = (task.task.seq_id, task.task.navi.current)
        step = task.task.step
        duplicate = step <= progress.get(node_id, -1)
        if not duplicate:
            progress[node_id] = step

        for node_name in node_names:
            should_drop = (
                step in drop_steps
                and step not in injected_steps
                and str(task.task.seq_id).startswith("bench-")
                and not duplicate
                and not task.trace_invalid
            )
            if should_drop:
                # Mark before scheduling so duplicate packets cannot schedule
                # the same failure twice.
                injected_steps.add(step)

                async def drop_compute_node(target: str) -> None:
                    await dealer.wait_node_init(target)
                    if target not in dealer.name_to_nodes:
                        return
                    record_event(
                        "failure_injected",
                        dealer_node=dealer.base.node_name,
                        compute_node=target,
                        sequence_id=task.task.seq_id,
                        step=task.task.step,
                        planned_failure_index=ordered_steps.index(step),
                        model=model,
                        injection_mode="deterministic_compute_node_teardown",
                    )
                    dealer.base.logger.warn(
                        "recover-analysis-basic deterministic drop: "
                        f"node={target}, step={task.task.step}"
                    )
                    # This removes the live per-request compute node and its
                    # runtime state, then reports the failure to the Agent.  It
                    # intentionally does not kill the whole Dealer process.
                    dealer.suicide_node(target)

                asyncio.create_task(drop_compute_node(node_name))
            else:
                copied_metrics = None if metrics is None else metrics.copy()
                asyncio.create_task(
                    dealer.handle_infer_task(
                        node_name,
                        task.shallow_copy(),
                        copied_metrics,
                    )
                )

    dealer.on_infer_task = on_infer_task


def deterministic_failure_dealer_node(ctx: Context) -> None:
    """Dealer entry point compatible with benchmark ``launch_network``."""

    async def run(ctx: Context, *args, **kwargs) -> None:
        base_node, dealer, initialize = dealer_coro(ctx, *args, **kwargs)
        _embed_deterministic_failure(dealer)
        asyncio.create_task(initialize())
        while True:
            base_node.on_packet(await ctx.recv())

    asyncio.get_event_loop().create_task(
        run(ctx, *ctx.args.args, **ctx.args.kwargs)
    )
