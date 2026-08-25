"""DROSIM: DROFELL orchestration-layer simulation harness.

DROSIM runs DROFELL's own control-plane code (node/, dealers/) against modelled
time, transport and compute. See simulate/docs/02-设计-DROSIM.md.

The modules are deliberately small and each replaces exactly one boundary with
the physical world:

    clock.py    virtual time      (replaces monotonic/wall clock)
    fabric.py   packet transport  (replaces TCP + tc/qdisc)
    cost.py     operator timing   (replaces real CUDA execution)
    brain.py    backend scheduler (operate queues over the real memory twin)
    context.py  the Context ABC   (what node/ actually depends on)
    routing.py  path selection    (deliberately simple; strategy is out of scope)

Everything else -- STM admission, canon, shadow state, recovery orchestration,
the resource-location map, the memory digital twin -- is imported unchanged
from the production tree.
"""

from .brain import DeviceState, SimBrain
from .backend import TransferDomain
from .clock import VirtualClock, VirtualClockLoop, install, run_simulation
from .context import SimContext
from .cost import CostModel, ServiceTerm, TransferTerm
from .fabric import LOOPBACK, WAN, HostCpuCost, Link, SimFabric
from .routing import select_paths

__all__ = [
    "CostModel",
    "DeviceState",
    "LOOPBACK",
    "HostCpuCost",
    "Link",
    "ServiceTerm",
    "TransferTerm",
    "TransferDomain",
    "SimBrain",
    "SimContext",
    "SimFabric",
    "VirtualClock",
    "VirtualClockLoop",
    "WAN",
    "install",
    "run_simulation",
    "select_paths",
]
