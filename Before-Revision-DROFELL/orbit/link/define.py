from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Tuple
from ..core import LocationByName


@dataclass
class LinkPerfDescribe:
    latency_ms: float = 0   # ms
    bandwidth_mbps: float = 2**10  # mbps (1Gbps -> 128MiB/s)

    @property
    def bandwidth_m(self):
        return self.bandwidth_mbps

    @property
    def bandwidth_byte_s(self):
        return self.bandwidth_mbps*(2**(20-3))

    def __str__(self) -> str:
        return f"Latency ≈ {self.latency_ms:.3f} ms, Bandwidth ≈ {self.bandwidth_mbps:.1f}Mbps ({self.bandwidth_mbps/8:.1f}MiB/s)"


SendPerf = LinkPerfDescribe
RecvPref = LinkPerfDescribe


LinkType = Literal["virtual", "tcp"]
LinkDefine = Tuple[LinkType, LinkPerfDescribe]
PacketTypeNewNode = "new_node"
PacketTypeNodeDrop = "drop_node"


class Builder(ABC):
    @abstractmethod
    async def create_link(self, src: LocationByName, dst: LocationByName, link: LinkDefine) -> None:
        raise NotImplementedError

    @abstractmethod
    def notify_node_join(self, recv_node: LocationByName, new_node: LocationByName) -> None:
        raise NotImplementedError


class Extender(ABC):
    @abstractmethod
    async def pref_net(self, target_node: LocationByName, repeat_times: int = 3) -> Tuple[SendPerf, RecvPref]:
        raise NotImplementedError
