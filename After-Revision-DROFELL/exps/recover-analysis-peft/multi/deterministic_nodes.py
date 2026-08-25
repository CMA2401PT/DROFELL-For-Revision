"""Active-winner TrainNode failure injection for PEFT multi-path runs."""

from __future__ import annotations

from typing import Any

import extend_asyncio as asyncio

from node.dealer_node.dealer_node import DealerNode
from orbit.defines import Context
from scripts.online.common_nodes import dealer_coro
from utils.runtime_metrics import record_event


DROP_ACTIVE_WINNER_API = "peft_multi.drop_active_winner"


def embed_active_winner_failure(dealer: DealerNode) -> None:
    """Destroy the live training node named by an Agent-observed winner."""

    injected_indices: set[int] = set()
    dealer.base.tags.add("peft_multi_active_winner_failure")

    def drop_active_winner(src: str, payload: dict[str, Any]) -> None:
        if payload.get("dealer_node") != dealer.base.node_name:
            raise ValueError(
                f"winner command for {payload.get('dealer_node')!r} reached "
                f"{dealer.base.node_name!r}"
            )
        failure_index = int(payload["planned_failure_index"])
        if failure_index in injected_indices:
            raise RuntimeError(f"duplicate failure index {failure_index}")
        control_id = str(payload["control_id"])
        candidates = sorted(
            name for name in dealer.name_to_nodes
            if not name.startswith("train-inspect-")
        )
        if len(candidates) != 1:
            raise RuntimeError(
                "active-winner injection requires exactly one live TrainNode "
                f"for {control_id!r} on {dealer.base.node_name}, found "
                f"{candidates}"
            )
        target = candidates[0]
        injected_indices.add(failure_index)
        record_event(
            "failure_injected",
            dealer_node=dealer.base.node_name,
            compute_node=target,
            control_id=control_id,
            minibatch_id=str(payload["previous_winner_minibatch_id"]),
            request_version=int(payload["previous_winner_request_version"]),
            planned_failure_index=failure_index,
            planned_elapsed_s=float(payload["planned_elapsed_s"]),
            actual_elapsed_s=float(payload["actual_elapsed_s"]),
            topology="backbone_inner_multipath",
            injection_mode="agent_observed_active_winner_teardown",
            command_source=src,
        )
        dealer.base.logger.warn(
            "recover-analysis-peft/multi active-winner drop: "
            f"node={target}, failure_index={failure_index}"
        )
        dealer.suicide_node(target)

    dealer.base.apis[DROP_ACTIVE_WINNER_API] = drop_active_winner


def deterministic_failure_dealer_node(ctx: Context) -> None:
    async def run(ctx: Context, *args, **kwargs) -> None:
        base_node, dealer, initialize = dealer_coro(ctx, *args, **kwargs)
        embed_active_winner_failure(dealer)
        asyncio.create_task(initialize())
        while True:
            base_node.on_packet(await ctx.recv())

    asyncio.create_task(run(ctx, *ctx.args.args, **ctx.args.kwargs))
