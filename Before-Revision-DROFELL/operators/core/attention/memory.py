import torch
from ..memory import MemoryModule, MemoryTensor
from ..linear.memory import LinearMakerMemoryModel
from .common import AttentionConfig


class AttentionDataMemoryModel(MemoryModule):
    def __init__(self, config: AttentionConfig, linear_maker: LinearMakerMemoryModel) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_query_heads = config.num_query_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.q_proj = linear_maker.make_linear(self.hidden_size, self.num_query_heads * self.head_dim, bias=False)
        self.k_proj = linear_maker.make_linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = linear_maker.make_linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = linear_maker.make_linear(self.num_query_heads * self.head_dim, self.hidden_size, bias=False)


class PagedAttentionMemoryModel(MemoryModule):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.cfg = config
        self.head_mapping = MemoryTensor((self.cfg.num_key_value_heads,), dtype=torch.int32).repeat_interleave(repeats=self.cfg.num_key_value_groups)
