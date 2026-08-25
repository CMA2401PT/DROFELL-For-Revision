"""Assemble and run one simulated DROFELL deployment.

WHAT RUNS HERE IS THE PRODUCTION CODE
-------------------------------------
`AuxNode`, `AgentNode`, `DealerNode`, `InferNode`, `InferControl`, `Provider`,
every `Shim`, the STM in `dealers/common/reorder.py` and the shadow state in
`node/shadow_node_state/` are imported and used UNMODIFIED. Only three things are
replaced, at the three boundaries where the code touches the physical world:

    orbit Context   -> SimContext  (transport on a virtual timeline)
    Brain           -> SimBackend  (operator time from the cost model)
    wall clock      -> VirtualClock

So a simulated run exercises the same admission decisions, the same canon
semantics, the same recovery orchestration and the same merge policy as hardware.
That is the entire credibility argument for using it to answer 2.7 (docs/01 §2).

WHAT IS NOT CLAIMED
-------------------
Token values are faked (`SimBackend`), so this says nothing about output quality.
Absolute throughput is only as good as the cost model, whose calibration range is
stated in `cost.py` and re-stated in every manifest this module writes.
"""

from __future__ import annotations

import json
import math
import statistics
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import extend_asyncio as asyncio

from dealers.common.base_types import ModelDescribe, gen_user_model_names
from node.aux_node.aux_node import AuxNode
from node.aux_node.protocols import ExtraInfo, ExtraModelInfo
from node.agent.agent import AgentNode
from node.base import BaseNode
from node.dealer_node.dealer_node import DealerNode, ModelTargets
from node.request.request import RequestNode
from operators.core.sampler.common import SamplingParams
from operators.core.tokenizer.common import SequenceCore, StopConfig
from utils.runtime_metrics import record_event

from .backend import DeviceModel, SimBackend, TransferDomain
from .clock import VirtualClock, install, run_simulation, uninstall
from .context import SimContext
from .cost import CostModel
from .fabric import LOOPBACK, HostCpuCost, Link, SimFabric
from .failure import DeterministicPlan, FailureInjector, InjectionStats
from .routing import select_paths
from .topology import Placement, Topology, build


@dataclass
class SimConfig:
    """Everything a scenario needs, with measured defaults."""

    model_name: str = "teared-Llama-2-7B-GPTQ"
    num_decoders: int = 32
    hidden_size: int = 4096
    segments: int = 4
    redundancy: int = 1
    placement: Placement = Placement.DEDICATED
    device_count: Optional[int] = None
    hosts: int = 1
    clients: int = 1
    deployment: str = "combined-request-agent"
    tokens: int = 16
    prompt_len: int = 4
    max_concurrency: int = 8
    max_seq_len: int = 4096
    link: Optional[Link] = None
    host_cpu_cost: Optional[HostCpuCost] = None
    log_level: str = "error"
    models_path: Optional[str] = None
    # Quantitative runs must point to a single-machine calibration profile.
    # None keeps the legacy mixed-host model only for semantic regression and
    # forces validity.paper_eligible=false in the result manifest.
    cost_profile: Optional[Path] = None
    # Calibration code may inject an in-memory single-host model to avoid
    # repeatedly serialising trial profiles. Normal experiment entry points
    # should use `cost_profile` so provenance is file-backed and auditable.
    cost_model: Optional[CostModel] = field(default=None, repr=False)
    retrieve: bool = False
    # See CLIENT_TURNAROUND_US. Set to 0.0 to measure the orchestration path
    # alone, which is what the scale sweeps do.
    client_turnaround_us: Optional[float] = None
    # Timing summaries exclude TTFT and begin after this many completed output
    # tokens per client. One is the minimum meaningful value: the first output
    # is TTFT, and the first decode ITL is the gap from output 0 to output 1.
    # Hardware calibration runs should set this to the benchmark's warm-up
    # count instead of relying on short-run setup amortisation.
    metric_warmup_tokens: int = 1
    # Failure injection. `failure_probability` is per non-duplicate task event at
    # a Dealer, matching the production injector's definition exactly (see
    # drosim/failure.py). Seeded, so a repeat is a repeat.
    failure_probability: float = 0.0
    failure_seed: int = 0
    deterministic_drop_steps: tuple[int, ...] = ()
    # Total deliberate kills allowed across the WHOLE deployment, not per Dealer.
    # Default 1 so a "one node failure" scenario means exactly that.
    deterministic_drop_budget: int = 1

    def resolved_link(self) -> Link:
        # Same-host loopback by default, matching the autodl runs the segment-count
        # axis was calibrated on. A shaped Link reproduces the 3090 tc tiers.
        return self.link if self.link is not None else LOOPBACK


@dataclass
class SimResult:
    schema_version: int
    topology: Dict[str, Any]
    config: Dict[str, Any]
    tokens_produced: int
    virtual_seconds: float
    wall_seconds: float
    aggregate_tokens_per_second: float
    per_client_inter_token_ms: float
    fabric: Dict[str, Any]
    backends: List[Dict[str, Any]]
    cost_model: Dict[str, Any]
    validity: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)
    injection: Dict[str, Any] = field(default_factory=dict)
    events: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


@dataclass
class ClientTiming:
    """Per-step virtual timestamps for one sequential client stream."""

    client: int
    issue_ns: List[int] = field(default_factory=list)
    completion_ns: List[int] = field(default_factory=list)

    def report(self, warmup_tokens: int) -> Dict[str, Any]:
        if len(self.issue_ns) != len(self.completion_ns):
            raise AssertionError(
                f"client {self.client}: {len(self.issue_ns)} issues but "
                f"{len(self.completion_ns)} completions"
            )
        if not self.completion_ns:
            return {
                "client": self.client,
                "tokens": 0,
                "ttft_ms": None,
                "request_latency_ms": None,
                "measurement_anchor_ns": None,
                "measurement_intervals": 0,
                "itl_ms": [],
            }

        # At least the first output belongs to TTFT, not to steady decode.
        warmup = max(1, int(warmup_tokens))
        anchor_index = min(warmup - 1, len(self.completion_ns) - 1)
        selected = self.completion_ns[anchor_index:]
        itl_ms = [
            (right - left) / 1e6
            for left, right in zip(selected, selected[1:])
        ]
        return {
            "client": self.client,
            "tokens": len(self.completion_ns),
            "first_issue_ns": self.issue_ns[0],
            "first_completion_ns": self.completion_ns[0],
            "last_completion_ns": self.completion_ns[-1],
            "ttft_ms": (self.completion_ns[0] - self.issue_ns[0]) / 1e6,
            "request_latency_ms": (
                self.completion_ns[-1] - self.issue_ns[0]
            ) / 1e6,
            "measurement_anchor_ns": selected[0],
            "measurement_intervals": len(itl_ms),
            # Raw intervals are retained so every aggregate in a paper table is
            # auditable and can be recomputed without rerunning the simulator.
            "itl_ms": itl_ms,
        }


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    """Linearly interpolated percentile, with an explicit empty result."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _distribution(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


# ---------------------------------------------------------------------------
# The one calibrated constant that is NOT derived from a service-time fit
# ---------------------------------------------------------------------------
# Comparing the simulator against the measured autodl curve (probes/p18) leaves a
# residual that is nearly flat in segment count:
#
#     L        1      2      4      8
#     gap   9.97   9.54   8.45   6.34   ms
#
# The SLOPE already agrees -- the simulator's own fit gives h = 1.74 ms/hop
# against a measured 1.55, i.e. 12% on the term that actually carries the scale
# answer, with nothing fitted to make it so. What is missing is an L-INDEPENDENT
# constant, and its identity is structural rather than mysterious: on hardware a
# benchmark client is a separate process that receives a token and only then
# issues the next step, so every token pays one client round trip. In the
# simulator `_one_client` awaits the control object directly and that round trip
# costs nothing.
#
# It is declared here as an explicit, named, defaults-to-ZERO term rather than
# folded into a service time, because:
#   * it is calibrated against an end-to-end residual, not measured in isolation,
#     which is a weaker kind of evidence than alpha/beta/gamma and must not be
#     disguised as the same thing;
#   * every scale result in this line is reported with it OFF, so no scaling claim
#     depends on it. It exists so the absolute-latency comparison against hardware
#     can be made honestly, with the residual named instead of hidden.
CLIENT_TURNAROUND_US = 8580.0


def _models_path() -> str:
    import split_qllm_models

    return split_qllm_models.lib_path


def _extra_info(topology: Topology) -> ExtraInfo:
    """The Aux directory entry, built from the topology rather than the model lib.

    `scripts/online/models.py` builds this by scanning the installed model
    library and picking a split. Here the split is the independent variable, so
    it is stated explicitly -- otherwise the Aux would advertise a different
    segmentation than the Dealers actually hold.
    """
    return ExtraInfo(
        models={
            topology.model_name: ExtraModelInfo(
                hidden_size=topology.hidden_size,
                num_decoders=topology.num_decoders,
                decoders_groups=list(topology.decoder_groups),
                head=topology.head,
                tail=topology.tail,
            )
        },
        network_name=f"drosim-L{topology.segments}-r{topology.redundancy}",
        network_spec_hash="simulated",
    )


class Deployment:
    """One simulated network: aux + agent + N dealers + a request node."""

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.clock = VirtualClock()
        self.loop = install(self.clock)
        if config.cost_profile is not None and config.cost_model is not None:
            raise ValueError("set cost_profile or cost_model, not both")
        if config.cost_model is not None:
            self.cost = copy.deepcopy(config.cost_model)
        elif config.cost_profile is not None:
            self.cost = CostModel.from_profile(config.cost_profile)
        else:
            self.cost = CostModel.measured()
        environment = self.cost.environment
        self.client_turnaround_us = (
            float(config.client_turnaround_us)
            if config.client_turnaround_us is not None
            else float(environment.get("client_turnaround_us", 0.0))
        )
        if config.host_cpu_cost is not None:
            self.host_cpu_cost = config.host_cpu_cost
        else:
            raw_host_cpu = environment.get("host_cpu_cost") or {}
            self.host_cpu_cost = HostCpuCost(**raw_host_cpu)
        self.topology = build(
            model_name=config.model_name,
            num_decoders=config.num_decoders,
            hidden_size=config.hidden_size,
            segments=config.segments,
            redundancy=config.redundancy,
            placement=config.placement,
            device_count=config.device_count,
            hosts=config.hosts,
        )
        self.fabric = SimFabric(
            self.clock,
            lambda when_ns, callback: self.loop.call_at(when_ns / 1e9, callback),
            default_link=config.resolved_link(),
            # An explicit scenario link is a global shaping override, including
            # same-host logical peers. With no override, same-host peers use the
            # loopback model and the default applies only across hosts.
            force_default_link=config.link is not None,
            host_cpu_cost=self.host_cpu_cost,
        )
        self.contexts: Dict[str, SimContext] = {}
        self.base_nodes: Dict[str, BaseNode] = {}
        self.backends: Dict[str, SimBackend] = {}
        self.dealers: Dict[str, DealerNode] = {}
        self.devices: Dict[str, DeviceModel] = {}
        self.transfer_domains: Dict[str, TransferDomain] = {}
        self._pumps: List[asyncio.Task[None]] = []
        self.aux: Optional[AuxNode] = None
        self.agent: Optional[AgentNode] = None
        self.request_node: Optional[RequestNode] = None
        self.request_base: Optional[BaseNode] = None
        self.notes: List[str] = []
        self.injection_stats = InjectionStats()
        self.injectors: List[FailureInjector] = []
        self.client_timings: List[ClientTiming] = []
        self.deployment_start_ns = self.clock.now_ns
        self.setup_end_ns = self.clock.now_ns
        self.workload_start_ns = self.clock.now_ns
        self.workload_end_ns = self.clock.now_ns
        self.drain_end_ns = self.clock.now_ns

    # -- construction ------------------------------------------------------
    def _context(
        self, name: str, host: str, cpu_domain: Optional[str] = None
    ) -> SimContext:
        ctx = SimContext(
            name, self.fabric, self.clock, host=host,
            cpu_domain=cpu_domain,
            log_level=self.config.log_level,
        )
        self.contexts[name] = ctx
        return ctx

    def _pump(self, ctx: SimContext, node: BaseNode) -> None:
        """Deliver packets into a node, forever.

        This is what `scripts/online/common_nodes.py` does in its `while True:
        node.on_packet(await ctx.recv())` loop, so the packet handling path is
        identical.
        """

        async def pump() -> None:
            while True:
                node.on_packet(await ctx.recv())

        self._pumps.append(asyncio.create_task(pump()))

    async def start(self) -> None:
        topology = self.topology
        models_path = self.config.models_path or _models_path()

        # ---- aux -----------------------------------------------------------
        aux_ctx = self._context(
            topology.aux_name, "host0", "host0:aux-process"
        )
        aux_base = BaseNode(aux_ctx)
        self.aux = AuxNode(aux_base, _extra_info(topology))
        self.base_nodes[topology.aux_name] = aux_base
        self._pump(aux_ctx, aux_base)

        # ---- dealers -------------------------------------------------------
        dealer_cpu_domains: Dict[str, str] = {}
        for plan in topology.dealers:
            cpu_domain = f"{plan.host}:dealer-process:{plan.name}"
            dealer_cpu_domains[plan.name] = cpu_domain
            ctx = self._context(plan.name, plan.host, cpu_domain)
            base = BaseNode(ctx)
            device = self.devices.get(plan.device_id)
            if device is None:
                from .topology import RTX3090_BYTES

                device = DeviceModel(plan.device_id, RTX3090_BYTES)
                self.devices[plan.device_id] = device
            transfer_domain = self.transfer_domains.setdefault(
                plan.host, TransferDomain(domain_id=plan.host)
            )
            backend = SimBackend(
                node_name=plan.name,
                clock=self.clock,
                cost=self.cost,
                device=device,
                transfer_domain=transfer_domain,
                models_path=models_path,
                max_seq_len=self.config.max_seq_len,
                max_concurrency=self.config.max_concurrency,
            )
            dealer = DealerNode(base)
            self.base_nodes[plan.name] = base
            self.backends[plan.name] = backend
            self.dealers[plan.name] = dealer
            self._pump(ctx, base)

        # ---- request + Agent placement ------------------------------------
        request_ctx = self._context(
            topology.request_name, "host0", "host0:request-process"
        )
        self.request_base = BaseNode(request_ctx)
        self.base_nodes[topology.request_name] = self.request_base
        self._pump(request_ctx, self.request_base)
        if self.config.deployment == "combined-request-agent":
            # Production RequestNode constructs a local AgentNode on this SAME
            # BaseNode. There is no Request<->Agent packet boundary.
            self.request_node = RequestNode(
                self.request_base,
                agent_node=None,
                path_search_redundancy=self.config.redundancy,
            )
            self.agent = self.request_node._local_agent
            assert self.agent is not None
        elif self.config.deployment == "persistent-agent":
            # The hardware launcher colocates the persistent Agent in the same
            # process/thread as dealer-g0 (the first Dealer placement).
            colocated = topology.dealers[0]
            agent_ctx = self._context(
                topology.agent_name,
                colocated.host,
                dealer_cpu_domains[colocated.name],
            )
            agent_base = BaseNode(agent_ctx)
            self.agent = AgentNode(
                agent_base, path_search_redundancy=self.config.redundancy
            )
            self.base_nodes[topology.agent_name] = agent_base
            self._pump(agent_ctx, agent_base)
            self.request_node = RequestNode(
                self.request_base,
                agent_node=topology.agent_name,
                path_search_redundancy=self.config.redundancy,
            )
        else:
            raise ValueError(
                "deployment must be 'combined-request-agent' or "
                "'persistent-agent'"
            )

        # Every node must be able to reach every other: `try_connect` is what
        # populates `BaseNode.connections`, and the Agent's provider filtering
        # calls it before selecting a node.
        for name, base in self.base_nodes.items():
            for other in self.base_nodes:
                if other != name:
                    await base.try_connect(other)

        # ---- dealer init (the real ainit) ----------------------------------
        await asyncio.gather(
            *[
                self.dealers[plan.name].ainit(
                    self.backends[plan.name],
                    ModelTargets.from_specific(
                        decoders=[
                            s for s in plan.segments
                            if ModelDescribe.from_str(s).is_decoders
                        ],
                        bundle={
                            next(
                                s for s in plan.segments
                                if ModelDescribe.from_str(s).is_decoders
                            ): [
                                s for s in plan.segments
                                if not ModelDescribe.from_str(s).is_decoders
                            ]
                        } if any(
                            not ModelDescribe.from_str(s).is_decoders
                            for s in plan.segments
                        ) else {},
                    ),
                )
                for plan in topology.dealers
            ]
        )

        # ---- failure injection --------------------------------------------
        # Installed AFTER ainit so the announce/profile handshake is never
        # dropped: on hardware the injector is also armed on an already-running
        # network, and dropping a node before it has announced would test a
        # bootstrap path the measured runs never exercise.
        if self.config.failure_probability > 0 or self.config.deterministic_drop_steps:
            import random

            # ONE shared plan for the whole deployment, so "one deliberate
            # failure" stays one however many holders a segment has.
            shared_plan = DeterministicPlan(
                steps=frozenset(self.config.deterministic_drop_steps),
                remaining=self.config.deterministic_drop_budget,
            )
            self.deterministic_plan = shared_plan
            for index, plan in enumerate(topology.dealers):
                self.injectors.append(
                    FailureInjector(
                        self.dealers[plan.name],
                        probability=self.config.failure_probability,
                        plan=shared_plan,
                        rng=random.Random(self.config.failure_seed * 1000 + index),
                        stats=self.injection_stats,
                    )
                )

    # -- workload ----------------------------------------------------------
    def _keypoints(self) -> List[Tuple[str, int, int]]:
        """Providers per model segment, from DROSIM's own selector.

        `routing.select_paths` is used rather than the prototype's
        `gen_lowest_energy_paths` because the latter enumerates every path tied
        at the minimum switch count -- (r+1)^L of them, OOM past ~12 segments
        (probes/p13). Path-selection strategy is out of scope for the paper
        (docs/11), so the simulator is free to use a linear one.
        """
        providers = [
            self.topology.providers_for(model)
            for model in self.topology.base_models
        ]
        selection = select_paths(
            providers, redundancy=self.config.redundancy, allow_reuse=True
        )
        if selection.shortfall_reason:
            self.notes.append(f"routing shortfall: {selection.shortfall_reason}")
        return selection.keypoints

    async def _one_client(self, index: int) -> int:
        """Drive one client stream through the real Agent-side InferControl."""
        assert self.request_node is not None
        seq_id = f"bench-{index}"
        base_models = self.topology.base_models
        seq_core = SequenceCore(
            stop_config=StopConfig(self.config.tokens, ignore_eos=True)
        )
        seq_core.prompt_token_ids.extend(range(10, 10 + self.config.prompt_len))
        control = self.request_node.create_infer_control(
            seq_id=seq_id,
            base_models=base_models,
            seq_core=seq_core,
            sampling_params=SamplingParams(temperature=0.0, logprobs=None),
            user_model_data=None,
            keypoint_providers=self._keypoints(),
            retrieve=self.config.retrieve,
        )
        produced = 0
        timing = ClientTiming(client=index)
        self.client_timings.append(timing)
        token: Any = list(range(10, 10 + self.config.prompt_len))
        is_prefill = True
        turnaround_s = self.client_turnaround_us / 1e6
        while produced < self.config.tokens:
            timing.issue_ns.append(self.clock.now_ns)
            stopped, token, _logprobs, _trace = await control.infer_step(
                token, is_prefill=is_prefill
            )
            timing.completion_ns.append(self.clock.now_ns)
            is_prefill = False
            produced += 1
            if stopped:
                break
            if turnaround_s > 0:
                # The client is a separate process on hardware; it cannot issue
                # the next step in zero time.
                await asyncio.sleep(turnaround_s)
        self.request_node.remove_infer_control(seq_id)
        return produced

    async def run_workload(self) -> int:
        self.workload_start_ns = self.clock.now_ns
        results = await asyncio.gather(
            *[self._one_client(i) for i in range(self.config.clients)]
        )
        self.workload_end_ns = self.clock.now_ns
        await self._drain()
        self.drain_end_ns = self.clock.now_ns
        return sum(results)

    def timing_report(self) -> Dict[str, Any]:
        clients = [
            timing.report(self.config.metric_warmup_tokens)
            for timing in sorted(self.client_timings, key=lambda item: item.client)
        ]
        ttft = [
            float(client["ttft_ms"])
            for client in clients
            if client["ttft_ms"] is not None
        ]
        request_latency = [
            float(client["request_latency_ms"])
            for client in clients
            if client["request_latency_ms"] is not None
        ]
        itls = [
            float(value)
            for client in clients
            for value in client["itl_ms"]
        ]

        first_issues = [
            int(client["first_issue_ns"])
            for client in clients
            if client.get("first_issue_ns") is not None
        ]
        last_completions = [
            int(client["last_completion_ns"])
            for client in clients
            if client.get("last_completion_ns") is not None
        ]
        anchors = [
            int(client["measurement_anchor_ns"])
            for client in clients
            if client.get("measurement_anchor_ns") is not None
        ]
        measured_intervals = sum(
            int(client["measurement_intervals"]) for client in clients
        )

        active_ns = (
            max(last_completions) - min(first_issues)
            if first_issues and last_completions else 0
        )
        steady_ns = (
            max(last_completions) - min(anchors)
            if anchors and last_completions else 0
        )
        total_tokens = sum(int(client["tokens"]) for client in clients)
        return {
            "schema_version": 2,
            "definitions": {
                "ttft": "first completion - first issue, per client",
                "itl": (
                    "adjacent output-completion gaps after metric_warmup_tokens; "
                    "deployment setup, TTFT and teardown are excluded"
                ),
                "end_to_end_throughput": (
                    "all completed outputs / (latest completion - earliest issue)"
                ),
                "steady_state_throughput": (
                    "post-warmup output intervals / global post-warmup window"
                ),
            },
            "metric_warmup_tokens": max(1, self.config.metric_warmup_tokens),
            "phases_ms": {
                "setup": (self.setup_end_ns - self.deployment_start_ns) / 1e6,
                "workload_active": (
                    self.workload_end_ns - self.workload_start_ns
                ) / 1e6,
                "teardown_drain": (
                    self.drain_end_ns - self.workload_end_ns
                ) / 1e6,
                "total": (self.drain_end_ns - self.deployment_start_ns) / 1e6,
            },
            "ttft_ms": _distribution(ttft),
            "itl_ms": _distribution(itls),
            "request_latency_ms": _distribution(request_latency),
            "end_to_end_tokens_per_second": (
                total_tokens * 1e9 / active_ns if active_ns > 0 else None
            ),
            "steady_state_tokens_per_second": (
                measured_intervals * 1e9 / steady_ns if steady_ns > 0 else None
            ),
            "measured_intervals": measured_intervals,
            "clients": clients,
        }

    async def _drain(self) -> None:
        """Let the teardown packets actually be delivered.

        `remove_infer_control` -> `InferNodeHandle.drop` SENDS a `RemoveNode`
        packet; it does not remove the remote compute node synchronously. On the
        virtual timeline those packets are still in flight when the workload
        coroutine returns, so tearing down immediately leaves every Dealer holding
        its compute node -- and `DealerNode._debug_check_resource_release` (an
        atexit leak detector, not a simulator artefact) fires for each one.
        Hardware does not hit this because the process lives on past the request.

        Draining is also the honest thing to do: the release path is production
        code, so letting it run is one more piece of the orchestration actually
        being exercised.
        """
        for _ in range(64):
            await asyncio.sleep(0.01)
            if all(
                not dealer.name_to_nodes for dealer in self.dealers.values()
            ):
                return
        remaining = {
            name: list(dealer.name_to_nodes)
            for name, dealer in self.dealers.items()
            if dealer.name_to_nodes
        }
        if remaining:
            self.notes.append(f"compute nodes not released at drain: {remaining}")

    def close(self) -> None:
        # Production processes construct these nodes once and intentionally keep
        # their atexit leak checks for process lifetime. A calibration process
        # constructs hundreds of short-lived deployments; leaving every bound
        # callback registered retains the whole object graph and emits hundreds
        # of stale "all released" lines at final exit. The simulator already
        # performs `_drain` and records unreleased nodes in `notes`, so unregister
        # only after that explicit check has run.
        import atexit

        for dealer in self.dealers.values():
            atexit.unregister(dealer._debug_check_resource_release)
        if self.agent is not None:
            control_plane = self.agent.control_plane
            atexit.unregister(
                control_plane.infer_controls._debug_check_resource_release
            )
            atexit.unregister(
                control_plane.train_controls._debug_check_resource_release
            )
        for backend in self.backends.values():
            backend.close()
        for task in self._pumps:
            if not task.done():
                task.cancel()


def run(config: SimConfig) -> SimResult:
    """Run one scenario end to end and return a self-describing result."""
    import time

    deployment = Deployment(config)
    wall_started = time.monotonic()

    async def scenario() -> int:
        deployment.deployment_start_ns = deployment.clock.now_ns
        await deployment.start()
        deployment.setup_end_ns = deployment.clock.now_ns
        return await deployment.run_workload()

    try:
        tokens = run_simulation(deployment.loop, scenario())
    finally:
        deployment.close()
        uninstall()

    wall_elapsed = time.monotonic() - wall_started
    virtual_seconds = deployment.clock.now_s
    timing = deployment.timing_report()
    steady_tps = timing["steady_state_tokens_per_second"]
    mean_itl = timing["itl_ms"]["mean"]
    return SimResult(
        schema_version=2,
        topology=deployment.topology.report(),
        config={
            "clients": config.clients,
            "deployment": config.deployment,
            "tokens_per_client": config.tokens,
            "prompt_len": config.prompt_len,
            "max_concurrency": config.max_concurrency,
            "link": {
                "latency_ms": config.resolved_link().latency_ms,
                "bandwidth_mbit": config.resolved_link().bandwidth_mbit,
                "scope": (
                    "all_non_self_links"
                    if config.link is not None
                    else "same_host_loopback_and_default_inter_host"
                ),
            },
            "client_turnaround_us_requested": config.client_turnaround_us,
            "client_turnaround_us_effective": deployment.client_turnaround_us,
            # Compatibility alias; schema-v2 readers should use `_effective`.
            "client_turnaround_us": deployment.client_turnaround_us,
            "metric_warmup_tokens": config.metric_warmup_tokens,
            "cost_profile": (
                str(config.cost_profile) if config.cost_profile is not None
                else None
            ),
            "failure_probability": config.failure_probability,
            "failure_seed": config.failure_seed,
            "deterministic_drop_steps": list(config.deterministic_drop_steps),
            "deterministic_drop_budget": config.deterministic_drop_budget,
        },
        tokens_produced=tokens,
        virtual_seconds=virtual_seconds,
        wall_seconds=wall_elapsed,
        aggregate_tokens_per_second=(
            float(steady_tps) if steady_tps is not None else float("nan")
        ),
        per_client_inter_token_ms=(
            float(mean_itl) if mean_itl is not None else float("nan")
        ),
        fabric=deployment.fabric.report(),
        backends=[b.report() for b in deployment.backends.values()],
        cost_model=deployment.cost.report(),
        validity={
            # A cost profile passing its own fit/holdout gates is necessary but
            # not sufficient: scenario-level G1/G3/G4 evidence is attached by
            # the unified validation runner, never inferred by `run()` alone.
            "paper_eligible": False,
            "cost_profile_eligible": deployment.cost.paper_eligible,
            "calibration_status": (
                "profile_loaded_scenario_gates_pending"
                if config.cost_profile is not None
                else "legacy_mixed_profile_for_regression_only"
            ),
            "reason": (
                "scenario-level G1/G3/G4 gates must be attached by the "
                "reviewer-grade validation runner"
            ),
        },
        timing=timing,
        injection=deployment.injection_stats.report(),
        notes=deployment.notes,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--redundancy", type=int, default=1)
    parser.add_argument("--clients", type=int, default=1)
    parser.add_argument(
        "--deployment",
        choices=("combined-request-agent", "persistent-agent"),
        default="combined-request-agent",
    )
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument(
        "--placement", choices=[p.value for p in Placement],
        default=Placement.DEDICATED.value,
    )
    parser.add_argument("--device-count", type=int, default=None)
    parser.add_argument("--log-level", default="error")
    parser.add_argument("--cost-profile", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = run(
        SimConfig(
            segments=args.segments,
            redundancy=args.redundancy,
            clients=args.clients,
            deployment=args.deployment,
            tokens=args.tokens,
            placement=Placement(args.placement),
            device_count=args.device_count,
            log_level=args.log_level,
            cost_profile=args.cost_profile,
        )
    )
    print(result.to_json())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.to_json())
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
