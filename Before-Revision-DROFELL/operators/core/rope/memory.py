import torch
from ..memory import MemoryModule, MemoryTensor
from .common import RopeConfig


class RotaryEmbeddingMemoryModel(MemoryModule):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = MemoryTensor((self.dim//2,), torch.float32)
        self.cos_cached = MemoryTensor((max_position_embeddings, (self.dim // 2) * 2), torch.float16)
        self.sin_cached = MemoryTensor((max_position_embeddings, (self.dim // 2) * 2), torch.float16)


def make_rope_embedding_memory_model(config: RopeConfig):
    return RotaryEmbeddingMemoryModel(
        dim=config.head_dim, max_position_embeddings=config.max_seq_len,
        base=config.rope_theta
    )
