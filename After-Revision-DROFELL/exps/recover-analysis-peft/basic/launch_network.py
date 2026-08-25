#!/usr/bin/env python3
"""Launch the normal benchmark network with deterministic train failures."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from scripts.online.benchmark import launch_network as base_launcher
from scripts.online.common_nodes import dealer_node
from orbit.link.cpn import NetModel
from orbit.link.define import LinkPerfDescribe

from deterministic_nodes import deterministic_failure_dealer_node


_initialize_process = base_launcher.initialize_process


class RecoveryNetModel(NetModel):
    """Symmetric Agent-to-Dealer delay used by the causal experiment."""

    def connect_perf(self, src: str, dst: str):
        latency_ms = float(os.getenv("PEFT_RECOVER_LINK_LATENCY_MS", "0"))
        bandwidth_mbps = float(
            os.getenv("PEFT_RECOVER_LINK_BANDWIDTH_MBPS", "1024")
        )
        agent_prefix = os.getenv(
            "PEFT_RECOVER_AGENT_PREFIX", "peft-recover-"
        )
        src_is_agent = src.startswith(agent_prefix)
        dst_is_agent = dst.startswith(agent_prefix)
        src_is_dealer = src.startswith("T0-train-")
        dst_is_dealer = dst.startswith("T0-train-")
        if ((src_is_agent and dst_is_dealer)
                or (src_is_dealer and dst_is_agent)):
            return (
                "tcp",
                LinkPerfDescribe(
                    latency_ms=latency_ms,
                    bandwidth_mbps=bandwidth_mbps,
                ),
            )
        return ("tcp", LinkPerfDescribe())


def _seeded_initialize_process(async_backend: str, busy_wait: bool):
    loop = _initialize_process(async_backend, busy_wait)
    seed = int(os.getenv("PEFT_RECOVER_SEED", "20260813"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return loop


def _dealer_function(dealer):
    if dealer.failure_injection:
        return deterministic_failure_dealer_node
    return dealer_node


def main() -> None:
    # launch_local_network resolves this module global while constructing its
    # function table.  The patch is limited to this launcher process.
    base_launcher._dealer_function = _dealer_function
    base_launcher.initialize_process = _seeded_initialize_process
    base_launcher.NetModel = RecoveryNetModel
    base_launcher.main()


if __name__ == "__main__":
    main()
