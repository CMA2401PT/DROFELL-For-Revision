#!/usr/bin/env python3
"""Training benchmark with experiment-local active-winner failure commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import extend_asyncio as asyncio

from node.agent.train import AgentTrainController
from node.base import BaseNode
from node.dealer_node.protocols import API_TrainTask, MultiPathTrainTask
from node.request import RequestNode
from orbit.link.extender import ContextNetExtender
from scripts.online.benchmark import benchmark_train as base
from utils.runtime_metrics import record_event

from deterministic_nodes import DROP_ACTIVE_WINNER_API


def install_active_winner_failures(
    request: RequestNode,
    start_time_file: Path,
    failure_times_s: tuple[float, ...],
    relay_node: str,
) -> None:
    """Fail the physical source of the latest accepted training result."""

    if not failure_times_s:
        return
    local_agent = request._local_agent
    if local_agent is None:
        raise RuntimeError("PEFT multi experiment requires a local Agent")
    controller: AgentTrainController = local_agent.train_controller
    original = controller.on_train_task
    next_failure = 0
    start_time_ns = int(start_time_file.read_text(encoding="utf-8").strip())

    def on_train_task(node: str, task: MultiPathTrainTask) -> None:
        nonlocal next_failure
        control = controller.base.train_controls.get(task.control_id or "")
        accepted = bool(
            task.task.navi.finished
            and control is not None
            and task.unique_id in control.result_waitor
        )
        original(node, task)
        if (
            not accepted
            or str(task.control_id).startswith("train-inspect-")
            or next_failure >= len(failure_times_s)
        ):
            return
        actual_elapsed_s = (time.time_ns() - start_time_ns) / 1e9
        planned_elapsed_s = failure_times_s[next_failure]
        if actual_elapsed_s < planned_elapsed_s:
            return
        payload = {
            "dealer_node": node,
            "control_id": str(task.control_id),
            "previous_winner_minibatch_id": task.task.minibatch_id,
            "previous_winner_request_version": task.task.request_version,
            "planned_failure_index": next_failure,
            "planned_elapsed_s": planned_elapsed_s,
            "actual_elapsed_s": actual_elapsed_s,
        }
        record_event(
            "active_winner_failure_command_send",
            **payload,
            relay_node=relay_node,
        )
        controller.base.send((relay_node, DROP_ACTIVE_WINNER_API, payload))
        next_failure += 1

    controller.on_train_task = on_train_task
    controller.base.base.apis[API_TrainTask] = on_train_task


def benchmark_node(ctx):
    async def coro(ctx, config, output_dir, agent_node):
        ctx.report_ready()
        ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        request = RequestNode(base_node, agent_node=agent_node)
        failure_times = tuple(float(value) for value in json.loads(
            os.getenv("PEFT_MULTI_ACTIVE_WINNER_FAILURE_TIMES_S", "[]")
        ))
        if failure_times:
            install_active_winner_failures(
                request=request,
                start_time_file=Path(os.environ[
                    "PEFT_MULTI_TRAIN_START_TIME_FILE"
                ]),
                failure_times_s=failure_times,
                relay_node=os.environ["PEFT_MULTI_FAILURE_RELAY"],
            )

        async def benchmark_and_stop() -> None:
            await base.run_benchmark(request, config, output_dir)
            await asyncio.sleep(0.3)
            raise SystemExit(0)

        asyncio.create_task(benchmark_and_stop())
        while True:
            base_node.on_packet(await ctx.recv())

    asyncio.get_event_loop().create_task(
        coro(ctx, *ctx.args.args, **ctx.args.kwargs)
    )


if __name__ == "__main__":
    base.benchmark_node = benchmark_node
    base.main()
