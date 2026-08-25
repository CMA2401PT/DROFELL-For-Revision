"""Cost model for DROSIM: how long does one backend operation take?

CALIBRATION SOURCES (all measured, nothing invented)
----------------------------------------------------
probes/p12  一稿返修/整理后就绪的实验/推理整流
            8 optimisation cases x 3 repeats, ONE topology (4 segments,
            8 layers each, 16 clients). Gives the service-time FORM and the
            merge-size term.
probes/p16  一稿返修/整理后就绪的实验/推理扩展性(带规模实验)
            same model split over 1/2/4/8 segments ({32,16,8,4} layers each)
            x 1..128 clients. Gives the SEGMENT-COUNT axis, i.e. how cost
            splits into per-layer work and per-hop overhead.
probes/p17  一稿返修/整理后就绪的实验/吞吐量实验/3090-带tc
            three shaped tiers on one host. Validates that transport is
            separable from compute (two independent axes agree on the number of
            crossings per token; the compute residual holds to 2.8% across a
            20x latency range).

MODEL
-----
    T_service(stage, n, layers)
        = alpha + beta*n + gamma*layers + delta*n*layers              [us]

`alpha` is the per-invocation cost that does not scale with the shard (kernel
launch, cache-table preparation, sampling setup). `gamma` is the shared
per-decoder-layer cost, `beta` the per-merged-item invocation cost, and `delta`
the per-item per-layer work. The interaction is necessary to express a basic
physical distinction: batching eight requests through a 32-layer shard cannot
cost the same incremental work as batching them through a 4-layer shard. Legacy
p12 saw only 8-layer shards and therefore could not identify this term.

WHY THE SPLIT MATTERS FOR THE REVIEWER'S QUESTION
-------------------------------------------------
Reviewer 2.7 asks what happens as node count grows. Splitting the model over more
nodes holds TOTAL layers constant while multiplying the number of invocations, so
a model with only `alpha + beta*n` would predict that splitting is free, and one
with only `gamma*layers` would predict it is free too. Measured, it is neither:
each extra segment pays `alpha` again plus one network crossing. That is the
mechanism behind the non-monotonicity the authors report on hardware, and it is
what the simulator must reproduce to be trusted past 8 nodes.

WHY A SINGLE SERVICE TERM AND NOT A CPU/GPU SPLIT
-------------------------------------------------
The instrumentation records `exec_X -> done_X` (submission) and
`done_X -> async_X` (the CPU-CUDA async gap). Semantically distinct, but NOT
separately identifiable: the boundary moves with the dual-queue flag while their
sum stays stable (p12 finding 1; docs/10 §1.3). So the model uses the sum.

CONSEQUENCE FOR DUAL QUEUES
---------------------------
Dual queues do NOT shorten service time -- measured 1685 -> 2158 us, slightly
longer. What they change is whether the blocking is yieldable. The simulator
reproduces that by keeping service time fixed and letting the *unmodified*
dual-queue loop policy do its work.

WHAT IS DELIBERATELY NOT IN HERE
--------------------------------
* mailbox queueing (`wait_X -> exec_X`): the simulator's own queues produce it;
  modelling it here as well would double count.
* transport (`leave -> dealer-recv`): that is SimFabric's job, validated in p17.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Decomposition of the measured decoder service time
# ---------------------------------------------------------------------------
# p12 measured, on 8-layer shards:  T(8 layers) ~= 1918.8 us, beta ~= 0.8 us/item
# p16 measured, on the autodl host: h = 1.55 ms per extra segment at fixed total
#                                   layers, W = 14.71 ms of L-independent work.
#
# Solving the two together for a 32-decoder model:
#   the L-independent part W covers 32 layers of decode plus head, tail and the
#   client turnaround; the per-segment part h covers one extra invocation plus
#   one crossing. p17 measured the crossing on the shaped host, and on the
#   unshaped autodl loopback it is small, so h is dominated by the extra
#   invocation -- which is `alpha`.
#
# Hence: alpha ~= 1.55 ms per decoder invocation (p16, per-hop term)
#        gamma  = (T(8 layers) - alpha) / 8 layers    (p12, residual per layer)
DECODER_ALPHA_US = 1552.0      # p16: per-invocation, host-independent part
DECODER_GAMMA_US = 45.8        # (1918.8 - 1552.0) / 8, per decoder layer
DECODER_BETA_US = 0.8          # p12: per merged item

DEFAULT_TERMS: dict[str, tuple[float, float, float, float]] = {
    # stage                 alpha, beta/item, gamma/layer, delta/(item*layer)
    "decoder_generating": (
        DECODER_ALPHA_US, DECODER_BETA_US, DECODER_GAMMA_US, 0.0
    ),
    "decoder_prefill": (
        DECODER_ALPHA_US, DECODER_BETA_US, DECODER_GAMMA_US, 0.0
    ),
    "infer_head": (142.5, 8.1, 0.0, 0.0),
    "infer_tail": (1978.8, 0.0, 0.0, 0.0),
    # Bookkeeping stages: no measured trace segment isolates them, and the
    # dealer-dispatch overhead they sit inside is ~50 us in the traces. They are
    # given that order of magnitude rather than zero, because zero would make
    # block retrieval look free.
    "create_infer_binding": (50.0, 0.0, 0.0, 0.0),
    "put_user_model": (50.0, 0.0, 0.0, 0.0),
    "put_blocks": (50.0, 0.0, 0.0, 0.0),
    "get_blocks": (50.0, 0.0, 0.0, 0.0),
}
# Measured pipeline-utilisation constant (p12): throughput ~= ETA * batch/service
# on the calibration topology. Reported for cross-checking; the simulator derives
# throughput from its actual schedule rather than applying ETA.
MEASURED_ETA = 0.264
# Range over which the merge-size term was fitted. Past it the model
# extrapolates, which is warned about once per stage.
VALID_BATCH_RANGE = (1, 8)
# Range of layers-per-shard actually observed across p12 and p16.
VALID_LAYER_RANGE = (1, 32)


@dataclass(frozen=True)
class ServiceTerm:
    """alpha + beta*n + gamma*l + delta*n*l, in microseconds."""

    alpha_us: float
    beta_us_per_item: float
    gamma_us_per_layer: float = 0.0
    delta_us_per_item_layer: float = 0.0

    def duration_us(self, batch: int, layers: int = 0) -> float:
        return (
            self.alpha_us
            + self.beta_us_per_item * max(1, batch)
            + self.gamma_us_per_layer * max(0, layers)
            + self.delta_us_per_item_layer
            * max(1, batch) * max(0, layers)
        )

    def duration_ns(self, batch: int, layers: int = 0) -> int:
        return int(round(self.duration_us(batch, layers) * 1000))


@dataclass(frozen=True)
class TransferTerm:
    """One direction of a host/device copy path.

    `fixed_us` is paid once per merged transfer group, `per_tensor_us` once per
    member, and bandwidth is shared by the combined bytes. Keeping the two
    fixed terms separate lets holdout data determine whether batched D2H/H2D
    launch overhead amortizes or grows linearly with merge size.
    """

    fixed_us: float = 0.0
    per_tensor_us: float = 0.0
    bandwidth_gbps: float = 0.0

    def duration_us(self, size_bytes: int, transfers: int = 1) -> float:
        serial_us = (
            max(0, size_bytes) * 8 / (self.bandwidth_gbps * 1e3)
            if self.bandwidth_gbps > 0 else 0.0
        )
        return (
            (self.fixed_us if transfers > 0 else 0.0)
            + self.per_tensor_us * max(0, transfers)
            + serial_us
        )

    def duration_ns(self, size_bytes: int, transfers: int = 1) -> int:
        return int(round(self.duration_us(size_bytes, transfers) * 1000))


@dataclass
class CostModel:
    """Per-stage service times, plus a record of what was extrapolated."""

    terms: dict[str, ServiceTerm] = field(default_factory=dict)
    fallback: ServiceTerm = ServiceTerm(
        DECODER_ALPHA_US, DECODER_BETA_US, DECODER_GAMMA_US
    )
    eta: float = MEASURED_ETA
    valid_batch_range: tuple[int, int] = VALID_BATCH_RANGE
    valid_batch_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    valid_layer_range: tuple[int, int] = VALID_LAYER_RANGE
    profile_id: str = "legacy-mixed-2026-08-21"
    hardware_id: str = "MIXED_INVALID"
    paper_eligible: bool = False
    provenance: str = (
        "LEGACY INVALID: RTX3090 p12 + AutoDL p16 were mixed; retained only "
        "for regression while reviewer-grade profiles are rebuilt"
    )
    calibration: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    transfers: dict[str, TransferTerm] = field(default_factory=lambda: {
        "h2d": TransferTerm(),
        "d2h": TransferTerm(),
    })
    emit_warnings: bool = True
    _warned: set[str] = field(default_factory=set, repr=False)
    extrapolations: dict[str, int] = field(default_factory=dict, repr=False)

    @classmethod
    def measured(cls) -> "CostModel":
        """Legacy mixed-host model, retained only for regression.

        New quantitative experiments must use :meth:`from_profile`. Keeping the
        old constructor avoids silently changing existing semantic tests while
        its manifest makes the invalid provenance impossible to miss.
        """
        return cls(
            terms={
                name: ServiceTerm(alpha, beta, gamma, delta)
                for name, (alpha, beta, gamma, delta) in DEFAULT_TERMS.items()
            }
        )

    @classmethod
    def from_profile(cls, path: Path) -> "CostModel":
        """Load one machine-specific, gate-bearing calibration profile."""
        report = json.loads(Path(path).read_text())
        if int(report.get("schema_version", 0)) != 1:
            raise ValueError(f"unsupported cost-profile schema: {path}")
        hardware_id = str(report.get("hardware_id", "")).strip()
        if not hardware_id or "mixed" in hardware_id.lower():
            raise ValueError(
                f"cost profile must name one physical hardware domain: {path}"
            )
        raw_terms = report.get("terms")
        if not isinstance(raw_terms, dict) or not raw_terms:
            raise ValueError(f"cost profile has no service terms: {path}")
        terms = {
            name: ServiceTerm(
                float(entry["alpha_us"]),
                float(entry["beta_us_per_item"]),
                float(entry.get("gamma_us_per_layer", 0.0)),
                float(entry.get("delta_us_per_item_layer", 0.0)),
            )
            for name, entry in raw_terms.items()
        }
        raw_transfers = report.get("transfers") or {}
        transfers = {
            direction: TransferTerm(
                float(entry.get("fixed_us", 0.0)),
                float(entry.get("per_tensor_us", 0.0)),
                float(entry.get("bandwidth_gbps", 0.0)),
            )
            for direction, entry in raw_transfers.items()
        }
        transfers.setdefault("h2d", TransferTerm())
        transfers.setdefault("d2h", TransferTerm())
        calibration = report.get("calibration") or {}
        gates = calibration.get("gates") or {}
        paper_eligible = bool(gates) and all(
            bool(value) for value in gates.values()
        )
        return cls(
            terms=terms,
            fallback=terms.get(
                "decoder_generating",
                ServiceTerm(
                    DECODER_ALPHA_US, DECODER_BETA_US, DECODER_GAMMA_US
                ),
            ),
            eta=float(report.get("eta", MEASURED_ETA)),
            valid_batch_range=tuple(report.get("valid_batch_range", (1, 8))),
            valid_batch_ranges={
                name: tuple(value)
                for name, value in (report.get("valid_batch_ranges") or {}).items()
            },
            valid_layer_range=tuple(report.get("valid_layer_range", (1, 32))),
            profile_id=str(report.get("profile_id", Path(path).stem)),
            hardware_id=hardware_id,
            paper_eligible=paper_eligible,
            provenance=str(report.get("provenance", path)),
            calibration=calibration,
            environment=dict(report.get("environment") or {}),
            transfers=transfers,
        )

    @classmethod
    def from_fit(cls, path: Path) -> "CostModel":
        """Load from a p12 `--out` report, keeping the p16 per-layer term.

        p12 fits `alpha' + beta*n` on 8-layer shards, so its alpha' already
        contains eight layers of work. Re-deriving `alpha` and `gamma` from it
        keeps a re-fit consistent with the segment-count axis instead of
        silently reverting to a single-topology model.
        """
        report = json.loads(Path(path).read_text())
        terms: dict[str, ServiceTerm] = {}
        for name, entry in report.get("service_terms", {}).items():
            alpha_at_8 = float(entry["alpha_us"])
            beta = float(entry["beta_us_per_item"])
            if name.startswith("decoder"):
                gamma = max(0.0, (alpha_at_8 - DECODER_ALPHA_US) / 8.0)
                terms[name] = ServiceTerm(DECODER_ALPHA_US, beta, gamma)
            else:
                terms[name] = ServiceTerm(alpha_at_8, beta, 0.0)
        if "decoder_generating" in terms and "decoder_prefill" not in terms:
            terms["decoder_prefill"] = terms["decoder_generating"]
        for name, (alpha, beta, gamma, delta) in DEFAULT_TERMS.items():
            terms.setdefault(name, ServiceTerm(alpha, beta, gamma, delta))
        model = cls(
            terms=terms,
            provenance=f"LEGACY INVALID cross-source fit: {path} + p16",
        )
        throughput = report.get("throughput_model") or {}
        if "eta" in throughput:
            model.eta = float(throughput["eta"])
        return model

    def term(self, stage: str) -> ServiceTerm:
        return self.terms.get(stage, self.fallback)

    def per_layer_us(self, stage: str = "decoder_generating") -> float:
        """Per-decoder-layer cost, for deriving a Profile's TPS."""
        term = self.term(stage)
        return term.gamma_us_per_layer or term.alpha_us

    def duration_ns(self, stage: str, batch: int = 1, layers: int = 0) -> int:
        """Service time for one merged operation, in virtual nanoseconds."""
        valid_batch_range = self.valid_batch_ranges.get(
            stage, self.valid_batch_range
        )
        low, high = valid_batch_range
        if batch > high or batch < low:
            self._flag(
                f"{stage}:batch={batch}", stage,
                f"batch={batch} outside {valid_batch_range}",
            )
        if layers:
            low_l, high_l = self.valid_layer_range
            if layers > high_l or layers < low_l:
                self._flag(
                    f"{stage}:layers={layers}", stage,
                    f"layers={layers} outside {self.valid_layer_range}",
                )
        return self.term(stage).duration_ns(batch, layers)

    def transfer_duration_ns(
        self, direction: str, size_bytes: int, transfers: int = 1
    ) -> int:
        if direction not in self.transfers:
            raise KeyError(f"unknown transfer direction: {direction}")
        return self.transfers[direction].duration_ns(size_bytes, transfers)

    def _flag(self, key: str, stage: str, detail: str) -> None:
        self.extrapolations[key] = self.extrapolations.get(key, 0) + 1
        if key not in self._warned:
            self._warned.add(key)
            if self.emit_warnings:
                print(
                    f"[drosim.cost] WARNING: {stage!r} with {detail}; the "
                    f"service time is EXTRAPOLATED beyond the calibration "
                    f"range."
                )

    def report(self) -> dict[str, Any]:
        """Provenance for the run manifest."""
        return {
            "profile_id": self.profile_id,
            "hardware_id": self.hardware_id,
            "paper_eligible": self.paper_eligible,
            "terms": {
                name: {
                    "alpha_us": term.alpha_us,
                    "beta_us_per_item": term.beta_us_per_item,
                    "gamma_us_per_layer": term.gamma_us_per_layer,
                    "delta_us_per_item_layer": term.delta_us_per_item_layer,
                }
                for name, term in self.terms.items()
            },
            "transfers": {
                direction: {
                    "fixed_us": term.fixed_us,
                    "per_tensor_us": term.per_tensor_us,
                    "bandwidth_gbps": term.bandwidth_gbps,
                }
                for direction, term in self.transfers.items()
            },
            "eta_measured": self.eta,
            "valid_batch_range": list(self.valid_batch_range),
            "valid_batch_ranges": {
                name: list(value)
                for name, value in self.valid_batch_ranges.items()
            },
            "valid_layer_range": list(self.valid_layer_range),
            "extrapolations": dict(self.extrapolations),
            "source": self.provenance,
            "calibration": dict(self.calibration),
            "environment": dict(self.environment),
        }
