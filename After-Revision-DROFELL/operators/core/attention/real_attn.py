from torch import nn
from ..linear.real import LinearMaker
from .common import AttentionConfig


class AttentionData(nn.Module):
    def __init__(self, config: AttentionConfig, linear_maker: LinearMaker):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_query_heads = config.num_query_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups

        if (self.head_dim * self.num_query_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_query_heads})."
            )
        self.q_proj = linear_maker.make_linear(self.hidden_size, self.num_query_heads * self.head_dim, bias=False)
        self.k_proj = linear_maker.make_linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = linear_maker.make_linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = linear_maker.make_linear(self.num_query_heads * self.head_dim, self.hidden_size, bias=False)
