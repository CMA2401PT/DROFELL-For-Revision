import torch

from ..memory.static import MemoryType

from ..memory.env import Env, MemoryHolder
from ..memory.module import EmbeddingMemoryModel
from ..memory import MemoryModule, MemoryTensor
from .common import ModelHeadConfig


class ModelHeadMemoryModel(MemoryModule):
    def __init__(self, config: ModelHeadConfig) -> None:
        super().__init__()
        self.cfg = config
        self.embed_tokens = EmbeddingMemoryModel(num_embeddings=self.cfg.vocab_size, embedding_dim=self.cfg.hidden_size)

    def train_to_embedding_tokens(self, env: Env, input_tokens_tensor: MemoryHolder[MemoryTensor], dtype: torch.dtype) -> MemoryHolder[MemoryTensor]:
        assert input_tokens_tensor.item.is_cpu
        device_input = env.trace(input_tokens_tensor.item.device())
        out = self.embed_tokens.forward(env, device_input)
        out = env.trace(out.item.to(dtype))
        return out

    def infer_to_embedding_tokens(self, env: Env, input_tokens_tensor: MemoryHolder[MemoryTensor], dtype: MemoryType) -> MemoryHolder[MemoryTensor]:
        assert input_tokens_tensor.item.is_cpu
        device_input = env.trace(input_tokens_tensor.item.device())
        out = self.embed_tokens.forward(env, device_input)
        out = env.trace(out.item.to(dtype))
        return out
