from dataclasses import dataclass
_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]


@dataclass
class AttentionConfig:
    hidden_size: int
    head_dim: int
    num_query_heads: int
    num_key_value_heads: int
    num_key_value_groups: int

    def __post_init__(self):
        assert self.hidden_size == self.head_dim*self.num_query_heads
        assert self.num_key_value_heads*self.num_key_value_groups == self.num_query_heads
        assert self.head_dim in _SUPPORTED_HEAD_SIZES
