#!/usr/bin/env python3
"""Launch a normal benchmark network with the experiment-local failure node."""

from __future__ import annotations

import json
import os

from orbit.link.cpn import NetModel
from orbit.link.define import LinkDefine, LinkPerfDescribe
from scripts.online.benchmark import launch_network as base_launcher
from scripts.online.common_nodes import dealer_node

from deterministic_nodes import deterministic_failure_dealer_node


class ExperimentNetModel(NetModel):
    """Apply experiment-scoped one-way latency to selected data-plane links."""

    def __init__(self) -> None:
        self.dealers = set(json.loads(os.getenv(
            "RECOVER_ANALYSIS_DEALERS", "[]"
        )))
        self.latency_ms = float(os.getenv(
            "RECOVER_ANALYSIS_INTER_DEALER_LATENCY_MS", "0"
        ))
        self.agent_name = os.getenv("RECOVER_ANALYSIS_AGENT_NAME", "")
        self.agent_to_dealer_latency_ms = float(os.getenv(
            "RECOVER_ANALYSIS_AGENT_TO_DEALER_LATENCY_MS", "0"
        ))
        self.dealer_to_agent_latency_ms = float(os.getenv(
            "RECOVER_ANALYSIS_DEALER_TO_AGENT_LATENCY_MS", "0"
        ))
        if (
            self.latency_ms < 0
            or self.agent_to_dealer_latency_ms < 0
            or self.dealer_to_agent_latency_ms < 0
        ):
            raise ValueError("experiment network latencies must be non-negative")

    def connect_perf(self, src: str, dst: str) -> LinkDefine:
        if (
            self.latency_ms > 0
            and src in self.dealers
            and dst in self.dealers
        ):
            return ("tcp", LinkPerfDescribe(latency_ms=self.latency_ms))
        if (
            self.agent_to_dealer_latency_ms > 0
            and src == self.agent_name
            and dst in self.dealers
        ):
            return (
                "tcp",
                LinkPerfDescribe(
                    latency_ms=self.agent_to_dealer_latency_ms
                ),
            )
        if (
            self.dealer_to_agent_latency_ms > 0
            and src in self.dealers
            and dst == self.agent_name
        ):
            return (
                "tcp",
                LinkPerfDescribe(
                    latency_ms=self.dealer_to_agent_latency_ms
                ),
            )
        return super().connect_perf(src, dst)


def _experiment_dealer_function(dealer):
    if dealer.failure_injection:
        return deterministic_failure_dealer_node
    return dealer_node


def main() -> None:
    # ``launch_local_network`` resolves this module global when it constructs
    # the function table, so the patch remains local to this launcher process.
    base_launcher._dealer_function = _experiment_dealer_function
    base_launcher.NetModel = ExperimentNetModel
    base_launcher.main()


if __name__ == "__main__":
    main()
