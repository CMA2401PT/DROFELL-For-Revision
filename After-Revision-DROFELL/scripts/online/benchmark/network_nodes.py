"""Additional online nodes used by benchmark deployments.

The normal serving path uses :func:`scripts.online.common_nodes.dealer_node`.
This module contains the persistent Agent entry point and the optional
failure-injection Dealer wrapper.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, Tuple

import extend_asyncio as asyncio

from node.agent import AgentNode
from node.base import BaseNode
from node.dealer_node.dealer_node import DealerNode
from node.dealer_node.protocols import MultiPathInferTask
from orbit.defines import Context
from orbit.link.extender import ContextNetExtender
from scripts.online.common_nodes import dealer_coro
from utils.runtime_metrics import record_event


def agent_node(ctx: Context) -> None:
    async def run(ctx: Context, path_search_redundancy: int) -> None:
        ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        AgentNode(
            base_node,
            path_search_redundancy=path_search_redundancy,
        )
        ctx.report_ready()
        while True:
            base_node.on_packet(await ctx.recv())

    asyncio.create_task(run(ctx, *ctx.args.args, **ctx.args.kwargs))


def _embed_failure_injection(dealer: DealerNode) -> None:
    progress: Dict[Tuple[str, int], int] = {}
    drop_probability = 0.0
    deterministic_dealer = os.getenv("DROFELL_DETERMINISTIC_DROP_DEALER", "")
    deterministic_step_raw = os.getenv("DROFELL_DETERMINISTIC_DROP_STEP", "")
    deterministic_plan_raw = os.getenv("DROFELL_DETERMINISTIC_DROP_PLAN", "{}")
    deterministic_plan = json.loads(deterministic_plan_raw)
    if not isinstance(deterministic_plan, dict):
        raise TypeError("DROFELL_DETERMINISTIC_DROP_PLAN must be a JSON object")
    deterministic_steps = set(deterministic_plan.get(
        dealer.base.node_name, []
    ))
    if deterministic_dealer == dealer.base.node_name \
            and deterministic_step_raw:
        deterministic_steps.add(int(deterministic_step_raw))
    deterministic_dropped_steps: set[int] = set()
    dealer.base.tags.add("infer_drop_reset")

    def reset_probability(
        source: str,
        ticket_and_probability: Tuple[int, float],
    ) -> None:
        nonlocal drop_probability, progress
        ticket, probability = ticket_and_probability
        if not 0.0 <= probability <= 1.0:
            raise ValueError("drop probability must be in [0, 1]")
        drop_probability = probability
        progress = {}
        dealer.base.logger.warn(
            f"failure-injection probability reset to {probability}"
        )
        dealer.base.send_resp(source, ticket, probability)

    dealer.base.apis["infer_drop_reset"] = reset_probability

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
            deterministic_failure = (
                step in deterministic_steps
                and step not in deterministic_dropped_steps
                and str(task.task.seq_id).startswith("bench-")
                and not duplicate
                and not task.trace_invalid
            )
            should_drop = (
                str(task.task.seq_id).startswith("bench-")
                and not duplicate
                and not task.trace_invalid
                and random.random() < drop_probability
            )
            if deterministic_failure:
                deterministic_dropped_steps.add(step)
            if should_drop or deterministic_failure:
                async def drop_compute_node(
                    target: str,
                    is_deterministic: bool,
                ) -> None:
                    await dealer.wait_node_init(target)
                    if target not in dealer.name_to_nodes:
                        return
                    record_event(
                        "failure_injected",
                        dealer_node=dealer.base.node_name,
                        compute_node=target,
                        sequence_id=task.task.seq_id,
                        step=task.task.step,
                        model=model,
                    )
                    dealer.base.logger.warn(
                        f"trigger compute-node drop: {target}"
                        + (" (deterministic)" if is_deterministic else "")
                    )
                    dealer.suicide_node(target)

                asyncio.create_task(
                    drop_compute_node(node_name, deterministic_failure)
                )
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


def failure_dealer_node(ctx: Context) -> None:
    async def run(ctx: Context, *args, **kwargs) -> None:
        base_node, dealer, initialize = dealer_coro(ctx, *args, **kwargs)
        _embed_failure_injection(dealer)
        asyncio.create_task(initialize())
        while True:
            base_node.on_packet(await ctx.recv())

    asyncio.get_event_loop().create_task(
        run(ctx, *ctx.args.args, **ctx.args.kwargs)
    )
