"""Deterministic, experiment-local training compute-node failures."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Dict, Tuple

import extend_asyncio as asyncio

from node.dealer_node.dealer_node import DealerNode
from node.dealer_node.protocols import API_TrainTask, MultiPathTrainTask
from orbit.defines import Context
from scripts.online.common_nodes import dealer_coro
from utils.runtime_metrics import record_event


DROP_PLAN_ENV = "PEFT_RECOVER_DROP_PLAN"
DROP_TIMES_ENV = "PEFT_RECOVER_DROP_TIMES_S"
START_TIME_FILE_ENV = "PEFT_RECOVER_TRAIN_START_TIME_FILE"
CLAIM_DIR_ENV = "PEFT_RECOVER_CLAIM_DIR"


def _drop_plan() -> Dict[str, Tuple[int, ...]]:
    raw = json.loads(os.getenv(DROP_PLAN_ENV, "{}"))
    if not isinstance(raw, dict):
        raise TypeError(f"{DROP_PLAN_ENV} must be a JSON object")
    result: Dict[str, Tuple[int, ...]] = {}
    for dealer, versions in raw.items():
        if not isinstance(dealer, str) or not dealer:
            raise ValueError("drop-plan Dealer names must be non-empty strings")
        if not isinstance(versions, list):
            raise ValueError(f"{DROP_PLAN_ENV}[{dealer!r}] must be a list")
        values = tuple(versions)
        if (
            any(type(value) is not int or value < 0 for value in values)
            or values != tuple(sorted(values))
            or len(values) != len(set(values))
        ):
            raise ValueError(
                f"{DROP_PLAN_ENV}[{dealer!r}] must contain unique, "
                "increasing non-negative integers"
            )
        result[dealer] = values
    return result


def _embed_failure(dealer: DealerNode) -> None:
    """Install request-version or wall-time-triggered TrainNode failure."""

    plan = _drop_plan()
    versions = set(plan.get(dealer.base.node_name, ()))
    ordered_versions = sorted({v for items in plan.values() for v in items})
    raw_drop_times = json.loads(os.getenv(DROP_TIMES_ENV, "[]"))
    if not isinstance(raw_drop_times, list):
        raise TypeError(f"{DROP_TIMES_ENV} must be a JSON list")
    drop_times = tuple(float(value) for value in raw_drop_times)
    if (
        any(value <= 0 for value in drop_times)
        or drop_times != tuple(sorted(drop_times))
        or len(drop_times) != len(set(drop_times))
    ):
        raise ValueError(f"{DROP_TIMES_ENV} must be positive and increasing")
    if ordered_versions and drop_times:
        raise ValueError("version and wall-time failure plans are mutually exclusive")
    injected: set[int] = set()
    claim_dir_text = os.getenv(CLAIM_DIR_ENV)
    if not claim_dir_text:
        raise RuntimeError(f"{CLAIM_DIR_ENV} is required")
    claim_dir = Path(claim_dir_text)
    claim_dir.mkdir(parents=True, exist_ok=True)
    dealer.base.tags.add("peft_recover_deterministic_failure")
    record_event(
        "train_failure_plan_loaded",
        dealer_node=dealer.base.node_name,
        planned_request_versions=sorted(versions),
        planned_elapsed_times_s=list(drop_times),
        full_plan=plan,
    )

    latest_normal_task: dict[str, object] = {
        "minibatch_id": "",
        "request_version": -1,
        "model": "",
    }

    async def inject_at_elapsed_times() -> None:
        if not drop_times:
            return
        start_file_text = os.getenv(START_TIME_FILE_ENV)
        if not start_file_text:
            raise RuntimeError(f"{START_TIME_FILE_ENV} is required")
        start_file = Path(start_file_text)
        while not start_file.is_file():
            await asyncio.sleep(0.01)
        start_time_ns = int(start_file.read_text(encoding="utf-8").strip())
        for failure_index, planned_s in enumerate(drop_times):
            remaining_s = planned_s - (time.time_ns() - start_time_ns) / 1e9
            if remaining_s > 0:
                await asyncio.sleep(remaining_s)
            # Only the Dealer currently hosting a live TrainNode may claim the
            # global timed failure.  A replacement Dealer can therefore claim
            # later deadlines without resetting the experiment clock.
            while not dealer.name_to_nodes:
                await asyncio.sleep(0.01)
            claim = claim_dir / f"elapsed-{planned_s:g}s.claim"
            try:
                descriptor = os.open(
                    claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
                )
            except FileExistsError:
                continue
            target = sorted(dealer.name_to_nodes)[0]
            actual_s = (time.time_ns() - start_time_ns) / 1e9
            os.write(descriptor, dealer.base.node_name.encode("utf-8"))
            os.close(descriptor)
            record_event(
                "failure_injected",
                dealer_node=dealer.base.node_name,
                compute_node=target,
                minibatch_id=latest_normal_task["minibatch_id"],
                request_version=latest_normal_task["request_version"],
                planned_failure_index=failure_index,
                planned_elapsed_s=planned_s,
                actual_elapsed_s=actual_s,
                model=latest_normal_task["model"],
                injection_mode="timed_train_compute_node_teardown",
            )
            dealer.base.logger.warn(
                "recover-analysis-peft-basic timed drop: "
                f"node={target}, planned={planned_s:g}s, actual={actual_s:.3f}s"
            )
            dealer.suicide_node(target)

    asyncio.create_task(inject_at_elapsed_times())

    def on_train_task(src: str, task: MultiPathTrainTask) -> None:
        record_event(
            "train_packet_receive",
            recovery_id=task.recovery_id,
            logical_handoff_id=task.logical_handoff_id,
            control_id=task.control_id,
            dealer_node=dealer.base.node_name,
            source_node=src,
            packet_id=task.packet_id,
            origin=task.origin,
            task_index=task.recovery_task_index,
            task_count=task.recovery_task_count,
            minibatch_id=task.task.minibatch_id,
            request_version=task.task.request_version,
        )
        directed_model = task.task.navi.current_dir_user_model
        is_backward = directed_model.endswith(",backward")
        model = directed_model.removesuffix(",forward").removesuffix(",backward")
        node_names = dealer.user_model_to_nodes.get(model, [])
        if not node_names:
            dealer.base.logger.error(
                f"cannot find {model} for task {task.task.unique_id_str}; discard"
            )
            return

        metrics = None
        if task.detail_trace is not None:
            metrics = [("dealer-recv", time.time_ns())]

        version = task.task.request_version
        if task.origin == "normal" and not task.trace_invalid:
            latest_normal_task.update({
                "minibatch_id": task.task.minibatch_id,
                "request_version": version,
                "model": model,
            })
        eligible = (
            version in versions
            and version not in injected
            and task.task.navi.current == 0
            and not task.trace_invalid
            and task.task.minibatch_id[0].isdigit()
        )
        should_drop = False
        if eligible:
            # Dealer processes do not share Python memory. O_EXCL provides a
            # cross-process, per-case claim so a second in-flight minibatch of
            # the same version cannot fail the replacement node again.
            claim = claim_dir / f"request-version-{version}.claim"
            try:
                descriptor = os.open(
                    claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
                )
            except FileExistsError:
                pass
            else:
                os.write(descriptor, dealer.base.node_name.encode("utf-8"))
                os.close(descriptor)
                should_drop = True
        if version in versions and not task.trace_invalid:
            record_event(
                "train_failure_candidate",
                dealer_node=dealer.base.node_name,
                request_version=version,
                minibatch_id=task.task.minibatch_id,
                navigation_index=task.task.navi.current,
                directed_model=directed_model,
                compute_nodes=node_names,
                already_injected=(version in injected or (eligible and not should_drop)),
                should_drop=should_drop,
            )
        if should_drop:
            # Mark synchronously: two gradient-accumulation minibatches with
            # the same request version may arrive back-to-back.
            injected.add(version)
            target = node_names[0]

            async def drop_compute_node() -> None:
                await dealer.wait_node_init(target)
                if target not in dealer.name_to_nodes:
                    return
                record_event(
                    "failure_injected",
                    dealer_node=dealer.base.node_name,
                    compute_node=target,
                    minibatch_id=task.task.minibatch_id,
                    request_version=version,
                    planned_failure_index=ordered_versions.index(version),
                    model=model,
                    injection_mode="deterministic_train_compute_node_teardown",
                )
                dealer.base.logger.warn(
                    "recover-analysis-peft-basic deterministic drop: "
                    f"node={target}, request_version={version}, "
                    f"minibatch={task.task.minibatch_id}"
                )
                # This deletes the live TrainNode and reports API_NodeSuicide
                # to the Agent; the Dealer process itself remains online.
                dealer.suicide_node(target)

            asyncio.create_task(drop_compute_node())
            return

        for node_name in node_names:
            copied_metrics = None if metrics is None else metrics.copy()
            asyncio.create_task(
                dealer.handle_train_task(
                    node_name,
                    task.shallow_copy(),
                    is_backward,
                    copied_metrics,
                )
            )

    dealer.on_train_task = on_train_task
    # DealerNode registers a bound callback during __init__; replacing only
    # the attribute would leave BaseNode dispatching to the old callback.
    dealer.base.apis[API_TrainTask] = on_train_task


def deterministic_failure_dealer_node(ctx: Context) -> None:
    async def run(ctx: Context, *args, **kwargs) -> None:
        base_node, dealer, initialize = dealer_coro(ctx, *args, **kwargs)
        _embed_failure(dealer)
        asyncio.create_task(initialize())
        while True:
            base_node.on_packet(await ctx.recv())

    asyncio.get_event_loop().create_task(
        run(ctx, *ctx.args.args, **ctx.args.kwargs)
    )
