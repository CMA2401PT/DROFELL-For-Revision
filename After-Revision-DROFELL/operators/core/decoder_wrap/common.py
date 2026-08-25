from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from ..attention.common import AttentionConfig


@dataclass
class FuseConfig:
    hidden_size: int
    intermediate_size: int
    num_query_heads: int
    num_key_value_heads: int
    head_dim: int

    def __post_init__(self):
        assert self.head_dim*self.num_query_heads == self.hidden_size


@dataclass
class DecoderConfig:
    hidden_size: int
    head_dim: int
    num_query_heads: int
    num_key_value_heads: int
    num_key_value_groups: int
    intermediate_size: int
    hidden_act: str
    rms_norm_eps: float

    def __post_init__(self):
        assert self.hidden_size == self.head_dim*self.num_query_heads
        assert self.num_query_heads == self.num_key_value_heads*self.num_key_value_groups

    @property
    def attention_config(self) -> AttentionConfig:
        return AttentionConfig(hidden_size=self.hidden_size, head_dim=self.head_dim, num_query_heads=self.num_query_heads, num_key_value_heads=self.num_key_value_heads, num_key_value_groups=self.num_key_value_groups)


LoraHook = \
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]  # (hidden) -> LoraQ,LoraK,LoraV , shape=[num_tokens,hidden_size]
