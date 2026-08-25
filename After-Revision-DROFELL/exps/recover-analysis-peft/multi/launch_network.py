#!/usr/bin/env python3
"""Launch the Backbone outer-multipath PEFT recovery network."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import random
from typing import cast

import extend_asyncio as asyncio
import numpy as np
import torch
from orbit import Orbit, SimpleLogger
from orbit.defines import Context, Location
from orbit.launch_pad import LaunchPad
from orbit.link.cpn import NetModel, OrbitCPNCommander
from orbit.link.define import LinkDefine, LinkPerfDescribe
from orbit.link.extender import ContextNetExtender

from node.backbone import BackboneNode
from node.base import BaseNode
from node.dealer_node.dealer_node import BackboneSelectConfig
from scripts.online.benchmark.launch_network import (
    AsyncBackend,
    Placement,
    _dealer_arguments,
    placement,
)
from scripts.online.benchmark.network_spec import NetworkSpec
from scripts.online.common_nodes import aux_node

from deterministic_nodes import (
    DROP_ACTIVE_WINNER_API,
    deterministic_failure_dealer_node,
)


_NO_GC: list[object] = []


class ExperimentNetModel(NetModel):
    """Hide inner Dealers and optionally delay the designated detour path."""

    def __init__(self) -> None:
        self.agent = os.getenv("PEFT_MULTI_AGENT_NAME", "")
        self.inner_dealers = set(json.loads(os.getenv(
            "PEFT_MULTI_INNER_DEALERS", "[]"
        )))
        self.backbone = os.getenv(
            "PEFT_MULTI_BACKBONE_NAME", "T0-peft-backbone"
        )
        self.delayed_dealers = set(json.loads(os.getenv(
            "PEFT_MULTI_DELAYED_DEALERS", "[]"
        )))
        self.path_latency_ms = float(os.getenv(
            "PEFT_MULTI_PATH_LATENCY_MS", "0"
        ))

    def can_initiate_connect(self, src: str, dst: str) -> bool:
        if src == self.agent and dst in self.inner_dealers:
            return False
        return super().can_initiate_connect(src, dst)

    def connect_perf(self, src: str, dst: str) -> LinkDefine:
        if (
            self.path_latency_ms > 0
            and (
                (src in self.delayed_dealers and dst in {self.agent, self.backbone})
                or (dst in self.delayed_dealers and src in {self.agent, self.backbone})
            )
        ):
            return (
                "tcp",
                LinkPerfDescribe(latency_ms=self.path_latency_ms),
            )
        return super().connect_perf(src, dst)


def experiment_backbone_node(ctx: Context) -> None:
    async def run(
        ctx: Context,
        max_search_redundancy: int = 0,
        path_gen_redundancy: int = 0,
    ) -> None:
        ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        BackboneNode(
            base_node,
            max_search_redundancy=max_search_redundancy,
            path_gen_redundancy=path_gen_redundancy,
        )

        def relay_active_winner_drop(_source: str, payload: dict) -> None:
            dealer_node = str(payload["dealer_node"])
            record = dict(payload)
            record["relay_node"] = base_node.node_name
            base_node.send((dealer_node, DROP_ACTIVE_WINNER_API, record))

        base_node.apis[DROP_ACTIVE_WINNER_API] = relay_active_winner_drop
        ctx.report_ready()
        while True:
            base_node.on_packet(await ctx.recv())

    asyncio.create_task(run(ctx, *ctx.args.args, **ctx.args.kwargs))


def initialize_process(async_backend: str, busy_wait: bool) -> asyncio.Loop:
    def on_exception(_: asyncio.Loop, error: Exception) -> None:
        raise error

    asyncio.config_impl(cast(AsyncBackend, async_backend), busy_wait=busy_wait)
    seed = int(os.getenv("PEFT_MULTI_SEED", "20260814"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    loop = asyncio.get_event_loop()
    loop.on_except(on_exception)
    return loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--advertise-host", default="127.0.0.1")
    parser.add_argument(
        "--async-backend", choices=("py", "chimera", "golang"),
        default="golang",
    )
    parser.add_argument(
        "--busy-wait", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--place", action="append", nargs=4,
        metavar=("DEALER", "PROCESS", "THREAD", "DEVICE"), required=True,
    )
    parser.add_argument("--backbone-name", default="T0-peft-backbone")
    parser.add_argument("--backbone-process", type=int, default=4)
    parser.add_argument("--backbone-search-redundancy", type=int, default=3)
    parser.add_argument("--backbone-path-redundancy", type=int, default=1)
    args = parser.parse_args()
    args.place = [placement(item) for item in args.place]
    return args


def main() -> None:
    args = parse_args()
    spec = NetworkSpec.load(args.network)
    placements: list[Placement] = args.place
    if {item.dealer_name for item in placements} != set(spec.dealer_names):
        raise ValueError("placements must cover every Dealer exactly once")
    launch_pad = LaunchPad(partial(
        initialize_process, args.async_backend, args.busy_wait
    ))

    async def start() -> None:
        functions = {"aux": aux_node, "backbone": experiment_backbone_node}
        for dealer in spec.dealers:
            functions[f"dealer:{dealer.name}"] = deterministic_failure_dealer_node
        orbit = Orbit(
            launch_pad=launch_pad,
            functions_dict=functions,
            logger=SimpleLogger(),
            network_hostname=args.advertise_host,
            log_level="info",
        )
        commander = OrbitCPNCommander(orbit, ExperimentNetModel())
        _NO_GC.append(commander)
        await orbit.init(port=args.port, satellites=tuple())
        await orbit.launch(
            Location(name="aux"), fn_name="aux",
            runtime_extra_info=spec.extra_info(),
        )
        await orbit.launch(
            Location(name=args.backbone_name, process=args.backbone_process),
            fn_name="backbone",
            max_search_redundancy=args.backbone_search_redundancy,
            path_gen_redundancy=args.backbone_path_redundancy,
        )

        dealers = {dealer.name: dealer for dealer in spec.dealers}
        inner_dealers = set(json.loads(os.getenv(
            "PEFT_MULTI_INNER_DEALERS", "[]"
        )))
        tasks = []
        for item in placements:
            dealer = dealers[item.dealer_name]
            kwargs = _dealer_arguments(spec, dealer)
            kwargs["device_str"] = item.device
            kwargs["backbone"] = BackboneSelectConfig.from_specific(
                {args.backbone_name} if item.dealer_name in inner_dealers else set()
            )
            tasks.append(asyncio.create_task(orbit.launch(
                Location(
                    name=item.dealer_name,
                    process=item.process,
                    thread=item.thread,
                ),
                fn_name=f"dealer:{item.dealer_name}",
                **kwargs,
            )))
        await asyncio.gather(*tasks)
        print(
            f"network '{spec.name}' ready at "
            f"{args.advertise_host}:{args.port}"
        )
        await asyncio.Future()

    launch_pad.loop.create_task(start())
    launch_pad.loop.run_forever()


if __name__ == "__main__":
    main()
