"""Deterministic per-compute-node failure injection for multi-path recovery."""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Tuple

import extend_asyncio as asyncio

from node.dealer_node.dealer_node import DealerNode
from node.dealer_node.protocols import MultiPathInferTask
from orbit.defines import Context
from scripts.online.common_nodes import dealer_coro
from utils.runtime_metrics import record_event


DROP_PLAN_ENV = "RECOVER_MULTI_DROP_PLAN"
DROP_ACTIVE_WINNER_API = "recover_multi.drop_active_winner"


def configured_drop_plan() -> Dict[str, Tuple[int, ...]]:
    raw = os.getenv(DROP_PLAN_ENV, "{}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"{DROP_PLAN_ENV} must be a JSON object")
    plan: Dict[str, Tuple[int, ...]] = {}
    for dealer, configured_steps in value.items():
        if not isinstance(dealer, str) or not dealer:
            raise ValueError(f"{DROP_PLAN_ENV} has an invalid Dealer name")
        if not isinstance(configured_steps, list):
            raise ValueError(f"{DROP_PLAN_ENV}[{dealer!r}] must be a list")
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


def embed_deterministic_failure(dealer: DealerNode) -> None:
    """Drop only the selected physical branch at each configured decode step."""

    progress: Dict[Tuple[str, int], int] = {}
    drop_plan = configured_drop_plan()
    drop_steps = set(drop_plan.get(dealer.base.node_name, tuple()))
    all_steps = sorted({step for steps in drop_plan.values() for step in steps})
    injected_steps: set[int] = set()
    topology = os.getenv("RECOVER_MULTI_TOPOLOGY", "")
    dealer.base.tags.add("recover_multi_deterministic_failure")

    def drop_active_winner(src: str, payload: dict) -> None:
        if payload.get("dealer_node") != dealer.base.node_name:
            raise ValueError(
                f"active-winner command for {payload.get('dealer_node')!r} "
                f"reached {dealer.base.node_name!r}"
            )
        sequence_id = str(payload["sequence_id"])
        candidates = {
            node_name for node_name in dealer.name_to_nodes
            if node_name.startswith(sequence_id)
        }
        for user_model, node_names in dealer.user_model_to_nodes.items():
            if user_model.startswith(sequence_id):
                candidates.update(node_names)
        candidates = sorted(candidates)
        if len(candidates) != 1:
            raise RuntimeError(
                "active-winner injection requires exactly one live inference "
                f"node on {dealer.base.node_name}, found {candidates}"
            )
        target = candidates[0]
        step = int(payload["failure_step"])
        record_event(
            "failure_injected",
            dealer_node=dealer.base.node_name,
            compute_node=target,
            sequence_id=sequence_id,
            step=step,
            previous_winner_step=int(payload["previous_winner_step"]),
            model=None,
            topology=topology,
            injection_mode="agent_observed_active_winner_teardown",
            command_source=src,
        )
        dealer.base.logger.warn(
            "recover-analysis-infer/multi active-winner drop: "
            f"node={target}, next_step={step}"
        )
        dealer.suicide_node(target)

    dealer.base.apis[DROP_ACTIVE_WINNER_API] = drop_active_winner

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
                        planned_failure_index=all_steps.index(step),
                        model=model,
                        topology=topology,
                        injection_mode="deterministic_active_branch_teardown",
                    )
                    dealer.base.logger.warn(
                        "recover-analysis-infer/multi deterministic drop: "
                        f"node={target}, step={task.task.step}"
                    )
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
    async def run(ctx: Context, *args, **kwargs) -> None:
        base_node, dealer, initialize = dealer_coro(ctx, *args, **kwargs)
        embed_deterministic_failure(dealer)
        asyncio.create_task(initialize())
        while True:
            base_node.on_packet(await ctx.recv())

    asyncio.create_task(run(ctx, *ctx.args.args, **ctx.args.kwargs))
