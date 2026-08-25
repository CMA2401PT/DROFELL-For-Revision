from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from dealers.common.base_types import group_model_with_split

# 24 GiB, matching the RTX 3090 the shaped experiments ran on. Used as the twin's
# root memory so simulated admission hits the same quotas as hardware.
RTX3090_BYTES = 24 * (1 << 30)


class Placement(str, Enum):
    DEDICATED = "dedicated"   # one device per logical node
    SHARED = "shared"         # logical nodes share a fixed pool of devices


@dataclass
class DealerPlan:
    """One simulated Dealer: which model segments it holds, and on what device."""

    name: str
    host: str
    device_id: str
    group_ids: List[int]
    segments: List[str] = field(default_factory=list)
    is_head_holder: bool = False
    is_tail_holder: bool = False


@dataclass
class Topology:
    """A complete simulated deployment."""

    model_name: str
    num_decoders: int
    hidden_size: int
    segment_layers: List[int]
    head: str
    tail: str
    decoder_groups: List[str]
    dealers: List[DealerPlan]
    placement: Placement
    device_count: int
    redundancy: int
    agent_name: str = "agent"
    aux_name: str = "aux"
    request_name: str = "request"

    @property
    def segments(self) -> int:
        return len(self.decoder_groups)

    @property
    def base_models(self) -> List[str]:
        return [self.head, *self.decoder_groups, self.tail]

    def providers_for(self, model: str) -> List[str]:
        return [d.name for d in self.dealers if model in d.segments]

    def report(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "num_decoders": self.num_decoders,
            "segments": self.segments,
            "segment_layers": list(self.segment_layers),
            "logical_nodes": len(self.dealers),
            "placement": self.placement.value,
            "device_count": self.device_count,
            "redundancy": self.redundancy,
            "nodes_per_device": len(self.dealers) / max(1, self.device_count),
            # Stated on every result: a scale curve means different things under
            # the two placements, and the difference is the whole point (docs/04).
            "placement_meaning": (
                "dedicated: one device per logical node, so only orchestration "
                "cost grows"
                if self.placement is Placement.DEDICATED
                else "shared: logical nodes contend for a fixed device pool, "
                "reproducing the hardware regime"
            ),
        }


def build(
    model_name: str,
    num_decoders: int,
    hidden_size: int,
    segments: int,
    redundancy: int = 1,
    placement: Placement = Placement.DEDICATED,
    device_count: Optional[int] = None,
    hosts: int = 1,
    device_memory_bytes: int = RTX3090_BYTES,
) -> Topology:
    """Lay out `segments` model shards over `segments * redundancy` logical nodes.

    Splitting uses the production `group_model_with_split`, so segment names and
    layer ranges are byte-identical to what a real `network_spec` produces --
    which is what lets a simulated run be compared with a measured one at all.
    """
    if num_decoders % segments != 0:
        raise ValueError(
            f"{num_decoders} decoders do not divide into {segments} equal "
            f"segments; the measured configurations all do, and an uneven split "
            f"would silently change the per-segment cost."
        )
    per_segment = num_decoders // segments
    head, groups, tail = group_model_with_split(
        base_model_name=model_name,
        decoders=num_decoders,
        group_fn=lambda layers: [per_segment for _ in range(layers // per_segment)],
    )
    if len(groups) != segments:
        raise AssertionError(
            f"production splitter produced {len(groups)} groups, expected "
            f"{segments}"
        )

    total_nodes = segments * redundancy
    if placement is Placement.DEDICATED:
        devices = total_nodes if device_count is None else device_count
    else:
        if device_count is None:
            raise ValueError("shared placement requires an explicit device_count")
        devices = device_count

    dealers: List[DealerPlan] = []
    for replica in range(redundancy):
        for group_index, group in enumerate(groups):
            node_index = replica * segments + group_index
            name = f"dealer-{node_index + 1}"
            dealers.append(
                DealerPlan(
                    name=name,
                    host=f"host{node_index % max(1, hosts)}",
                    device_id=f"device-{node_index % devices}",
                    group_ids=[group_index],
                    segments=[group],
                    # The head sits with the first segment and the tail with the
                    # last, mirroring every measured network spec: a separate
                    # head/tail node would add a hop the hardware never paid.
                    is_head_holder=group_index == 0,
                    is_tail_holder=group_index == segments - 1,
                )
            )
    for dealer in dealers:
        if dealer.is_head_holder:
            dealer.segments.insert(0, head)
        if dealer.is_tail_holder:
            dealer.segments.append(tail)

    return Topology(
        model_name=model_name,
        num_decoders=num_decoders,
        hidden_size=hidden_size,
        segment_layers=[per_segment] * segments,
        head=head,
        tail=tail,
        decoder_groups=groups,
        dealers=dealers,
        placement=placement,
        device_count=devices,
        redundancy=redundancy,
    )
