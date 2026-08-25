from __future__ import annotations

import dataclasses
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .clock import VirtualClock

# Real `tc` defaults from orbit/examples/tc/Makefile
TC_LATENCY_MS = 50.0
TC_BANDWIDTH_MBIT = 100.0
# Same-host loopback, measured order of magnitude (traces show ~240 us median for
# same-host `leave -> dealer-recv`, which includes serialisation, not just wire).
LOOPBACK_LATENCY_MS = 0.05
LOOPBACK_BANDWIDTH_MBIT = 10_000.0


@dataclass(frozen=True)
class Link:
    """One directed link's characteristics."""

    latency_ms: float
    bandwidth_mbit: float

    @property
    def latency_ns(self) -> int:
        return int(self.latency_ms * 1e6)

    def transmit_ns(self, size_bytes: int) -> int:
        if self.bandwidth_mbit <= 0:
            return 0
        bits = size_bytes * 8
        seconds = bits / (self.bandwidth_mbit * 1e6)
        return int(seconds * 1e9)


LOOPBACK = Link(LOOPBACK_LATENCY_MS, LOOPBACK_BANDWIDTH_MBIT)
WAN = Link(TC_LATENCY_MS, TC_BANDWIDTH_MBIT)


@dataclass(frozen=True)
class HostCpuCost:
    """CPU serialization/deserialization cost around packet transport."""

    cores: int = 1
    serialize_fixed_us: float = 0.0
    serialize_bandwidth_gbps: float = 0.0
    deserialize_fixed_us: float = 0.0
    deserialize_bandwidth_gbps: float = 0.0

    @staticmethod
    def _duration_ns(size: int, fixed_us: float, bandwidth_gbps: float) -> int:
        transfer_us = (
            size * 8 / (bandwidth_gbps * 1e3)
            if bandwidth_gbps > 0 else 0.0
        )
        return int(round((fixed_us + transfer_us) * 1000))

    def serialize_ns(self, size: int) -> int:
        return self._duration_ns(
            size, self.serialize_fixed_us, self.serialize_bandwidth_gbps
        )

    def deserialize_ns(self, size: int) -> int:
        return self._duration_ns(
            size, self.deserialize_fixed_us, self.deserialize_bandwidth_gbps
        )


@dataclass
class FabricStats:
    packets: int = 0
    bytes_total: int = 0
    dropped: int = 0
    by_api: dict[str, int] = field(default_factory=dict)
    bytes_by_api: dict[str, int] = field(default_factory=dict)
    packets_by_link: dict[str, int] = field(default_factory=dict)
    queue_delay_ns: int = 0
    max_queue_delay_ns: int = 0
    host_cpu_service_ns: dict[str, int] = field(default_factory=dict)
    host_cpu_queue_ns: dict[str, int] = field(default_factory=dict)
    host_cpu_operations: dict[str, int] = field(default_factory=dict)

    def record(self, api: str, size: int, link: Link, queue_delay_ns: int) -> None:
        self.packets += 1
        self.bytes_total += size
        self.by_api[api] = self.by_api.get(api, 0) + 1
        self.bytes_by_api[api] = self.bytes_by_api.get(api, 0) + size
        link_key = f"{link.latency_ms:g}ms/{link.bandwidth_mbit:g}mbit"
        self.packets_by_link[link_key] = self.packets_by_link.get(link_key, 0) + 1
        self.queue_delay_ns += queue_delay_ns
        self.max_queue_delay_ns = max(self.max_queue_delay_ns, queue_delay_ns)


def _meta_tensor_payload_bytes(payload: Any) -> int:
    """Bytes omitted when a meta tensor is pickled.

    A meta tensor's pickle contains only shape/dtype metadata. Real Orbit
    transport sends the tensor storage out of band, so treating the pickle size
    as the wire size undercounts a 4096-wide fp16 activation by roughly 50x.
    CPU tensors already include their storage in the pickle and are not added.
    """
    try:
        import torch
    except Exception:
        return 0

    seen: set[int] = set()

    def visit(value: Any) -> int:
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        if isinstance(value, torch.Tensor):
            if value.device.type == "meta":
                return int(value.numel() * value.element_size())
            return 0
        if (
            value.__class__.__name__ == "DeviceTensorCarrier"
            and value.__class__.__module__ == "dealers.common.base_types"
        ):
            return visit(getattr(value, "t", None))
        if isinstance(value, dict):
            return sum(visit(k) + visit(v) for k, v in value.items())
        if isinstance(value, (list, tuple, set, frozenset)):
            return sum(visit(item) for item in value)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return sum(
                # Some production task objects deliberately delete large
                # fields after consuming them. Their dataclass schema remains,
                # so missing fields mean zero retained wire payload.
                visit(getattr(value, item.name, None))
                for item in dataclasses.fields(value)
            )
        return 0

    return visit(payload)


def wire_bytes(payload: Any) -> int:
    """Modelled serialised size, including meta tensor storage.

    Pickle alone is insufficient for meta tensors: their storage is deliberately
    absent. Add that logical storage exactly once while retaining pickle's
    object/control metadata. Falls back to retained size if unpicklable.
    """
    omitted_tensor_bytes = _meta_tensor_payload_bytes(payload)
    try:
        return (
            len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
            + omitted_tensor_bytes
        )
    except Exception:
        try:
            from utils.size_metrics import retained_bytes

            return max(int(retained_bytes(payload)), omitted_tensor_bytes)
        except Exception:
            return max(512, omitted_tensor_bytes)


class SimFabric:
    """Delivers packets between logical nodes on the virtual timeline."""

    def __init__(
        self,
        clock: VirtualClock,
        schedule_at_ns: Callable[[int, Callable[[], None]], None],
        default_link: Link = WAN,
        *,
        force_default_link: bool = False,
        share_host_interfaces: bool = True,
        host_cpu_cost: HostCpuCost = HostCpuCost(),
    ) -> None:
        self.clock = clock
        self._schedule_at_ns = schedule_at_ns
        self.default_link = default_link
        self.force_default_link = force_default_link
        self.share_host_interfaces = share_host_interfaces
        if host_cpu_cost.cores <= 0:
            raise ValueError("host_cpu_cost.cores must be positive")
        self.host_cpu_cost = host_cpu_cost
        self._links: dict[tuple[str, str], Link] = {}
        self._host_of: dict[str, str] = {}
        self._cpu_domain_of: dict[str, str] = {}
        self._link_free_at_ns: dict[tuple[str, str], int] = {}
        self._delivery_free_at_ns: dict[tuple[str, str], int] = {}
        # Logical nodes on one host do not each get a free independent NIC.
        # Cross-host traffic shares host egress/ingress; same-host traffic
        # shares one loopback interface. Without these cursors, adding nodes
        # creates O(N^2) aggregate bandwidth by construction.
        self._interface_free_at_ns: dict[tuple[str, str], int] = {}
        self._host_cpu_free_at_ns: dict[str, list[int]] = {}
        self._receivers: dict[str, Callable[[tuple[str, str, Any]], None]] = {}
        self._partitions: set[tuple[str, str]] = set()
        self.stats = FabricStats()

    # -- topology ---------------------------------------------------------
    def register(
        self,
        name: str,
        receiver: Callable[[tuple[str, str, Any]], None],
        host: str = "host0",
        cpu_domain: Optional[str] = None,
    ) -> None:
        self._receivers[name] = receiver
        self._host_of[name] = host
        self._cpu_domain_of[name] = cpu_domain or f"{host}:{name}"

    def unregister(self, name: str) -> None:
        self._receivers.pop(name, None)

    def known(self, name: str) -> bool:
        return name in self._receivers

    def set_link(self, src: str, dst: str, link: Link) -> None:
        self._links[(src, dst)] = link

    def link(self, src: str, dst: str) -> Link:
        explicit = self._links.get((src, dst))
        if explicit is not None:
            return explicit
        if self.force_default_link:
            return self.default_link
        if self._host_of.get(src, "?") == self._host_of.get(dst, "??"):
            return LOOPBACK
        return self.default_link

    def _interface_keys(self, src: str, dst: str) -> tuple[tuple[str, str], ...]:
        if not self.share_host_interfaces:
            return ()
        src_host = self._host_of.get(src, "unknown-src")
        dst_host = self._host_of.get(dst, "unknown-dst")
        if src_host == dst_host:
            return ((src_host, "loopback"),)
        # Full-duplex NIC model: outgoing and incoming directions are distinct,
        # while all logical nodes on a host share each direction.
        return ((src_host, "egress"), (dst_host, "ingress"))

    def _reserve_host_cpu(
        self, cpu_domain: str, ready_ns: int, duration_ns: int, phase: str
    ) -> int:
        cores = self._host_cpu_free_at_ns.setdefault(
            cpu_domain, [0 for _ in range(self.host_cpu_cost.cores)]
        )
        core = min(range(len(cores)), key=cores.__getitem__)
        start = max(ready_ns, cores[core])
        finish = start + duration_ns
        cores[core] = finish
        key = f"{cpu_domain}:{phase}"
        self.stats.host_cpu_service_ns[key] = (
            self.stats.host_cpu_service_ns.get(key, 0) + duration_ns
        )
        self.stats.host_cpu_queue_ns[key] = (
            self.stats.host_cpu_queue_ns.get(key, 0) + start - ready_ns
        )
        self.stats.host_cpu_operations[key] = (
            self.stats.host_cpu_operations.get(key, 0) + 1
        )
        return finish

    # -- failure injection (default off, docs/04 §3) -----------------------
    def partition(self, src: str, dst: str, both_ways: bool = True) -> None:
        self._partitions.add((src, dst))
        if both_ways:
            self._partitions.add((dst, src))

    def heal(self, src: str, dst: str, both_ways: bool = True) -> None:
        self._partitions.discard((src, dst))
        if both_ways:
            self._partitions.discard((dst, src))

    # -- delivery ---------------------------------------------------------
    def send(self, src: str, dst: str, api: str, payload: Any) -> bool:
        """Queue a packet. Returns False if it cannot be delivered at all."""
        receiver = self._receivers.get(dst)
        if receiver is None or (src, dst) in self._partitions:
            self.stats.dropped += 1
            return False

        size = wire_bytes(payload)
        link = self.link(src, dst)
        key = (src, dst)
        now = self.clock.now_ns
        src_cpu_domain = self._cpu_domain_of.get(src, "unknown-src")
        serialized_at = self._reserve_host_cpu(
            src_cpu_domain,
            now,
            self.host_cpu_cost.serialize_ns(size),
            "serialize",
        )
        interface_keys = self._interface_keys(src, dst)
        # Serialise both on the directed path and on the physical host
        # interfaces shared by all logical nodes mapped there.
        start = max(
            serialized_at,
            self._link_free_at_ns.get(key, serialized_at),
            *(
                self._interface_free_at_ns.get(item, serialized_at)
                for item in interface_keys
            ),
        )
        transmit = link.transmit_ns(size)
        finish = start + transmit
        self._link_free_at_ns[key] = finish
        for item in interface_keys:
            self._interface_free_at_ns[item] = finish
        arrival = start + transmit + link.latency_ns

        self.stats.record(api, size, link, start - serialized_at)

        def arrive() -> None:
            dst_cpu_domain = self._cpu_domain_of.get(dst, "unknown-dst")
            # TCP framing preserves order within one logical connection. Host
            # CPU cores may deserialize different connections concurrently,
            # but a later task on this flow cannot overtake create/init packets.
            delivery_ready = max(
                self.clock.now_ns,
                self._delivery_free_at_ns.get(key, self.clock.now_ns),
            )
            delivered_at = self._reserve_host_cpu(
                dst_cpu_domain,
                delivery_ready,
                self.host_cpu_cost.deserialize_ns(size),
                "deserialize",
            )
            self._delivery_free_at_ns[key] = delivered_at

            def deliver() -> None:
                handler = self._receivers.get(dst)
                if handler is not None:
                    handler((src, api, payload))

            if delivered_at <= self.clock.now_ns:
                deliver()
            else:
                self._schedule_at_ns(delivered_at, deliver)

        self._schedule_at_ns(arrival, arrive)
        return True

    def report(self) -> dict[str, Any]:
        return {
            "packets": self.stats.packets,
            "bytes_total": self.stats.bytes_total,
            "dropped": self.stats.dropped,
            "packets_by_api": dict(self.stats.by_api),
            "bytes_by_api": dict(self.stats.bytes_by_api),
            "packets_by_link": dict(self.stats.packets_by_link),
            "queue_delay_ms_total": self.stats.queue_delay_ns / 1e6,
            "queue_delay_ms_max": self.stats.max_queue_delay_ns / 1e6,
            "host_cpu_service_ms": {
                key: value / 1e6
                for key, value in self.stats.host_cpu_service_ns.items()
            },
            "host_cpu_queue_ms": {
                key: value / 1e6
                for key, value in self.stats.host_cpu_queue_ns.items()
            },
            "host_cpu_operations": dict(self.stats.host_cpu_operations),
            "default_link": {
                "latency_ms": self.default_link.latency_ms,
                "bandwidth_mbit": self.default_link.bandwidth_mbit,
            },
            "link_policy": (
                "global_override_for_all_non_self_links"
                if self.force_default_link
                else "same_host_loopback_else_default"
            ),
            "shared_host_interfaces": self.share_host_interfaces,
            "host_cpu_cost": dataclasses.asdict(self.host_cpu_cost),
            "cpu_domain_of_node": dict(self._cpu_domain_of),
            "source": (
                "explicit scenario link override"
                if self.force_default_link
                else "same-host loopback plus scenario inter-host default"
            ),
        }
