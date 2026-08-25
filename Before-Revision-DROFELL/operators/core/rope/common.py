from dataclasses import dataclass
from typing import Optional


@dataclass
class RopeConfig:
    rope_theta: int
    head_dim: int
    max_seq_len: int
    rope_scaling_type: Optional[str] = None
    rope_scaling_factor: Optional[float] = None

    def __post_init__(self):
        if self.rope_scaling_type is not None:
            assert self.rope_scaling_factor is not None
