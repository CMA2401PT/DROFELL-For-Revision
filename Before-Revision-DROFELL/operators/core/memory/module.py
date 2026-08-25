from .env import Env, MemoryHolder
from .static import MemoryFloat32, MemoryModule, MemoryTensor


class LinearMemoryModel(MemoryModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = MemoryTensor((in_features, out_features), dtype=MemoryFloat32)
        if bias:
            self.bias = MemoryTensor((out_features,), dtype=MemoryFloat32)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, env: Env, x: MemoryHolder[MemoryTensor]):
        assert not x.item.is_cpu
        assert x.item.size[-1] == self.in_features
        return env.trace(MemoryTensor((*x.item.size[:-1], self.out_features), MemoryFloat32))


class EmbeddingMemoryModel(MemoryModule):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = MemoryTensor((num_embeddings, embedding_dim), dtype=MemoryFloat32)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, env: Env, x: MemoryHolder[MemoryTensor]):
        assert not x.item.is_cpu
        return env.trace(MemoryTensor((*x.item.size, self.embedding_dim), MemoryFloat32))
