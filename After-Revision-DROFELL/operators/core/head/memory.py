import torch

from ..memory.static import MemoryDType, MemoryFloat32, MemoryType, get_dtype

from ..memory.env import Env, MemoryHolder
from ..memory.module import EmbeddingMemoryModel
from ..memory import MemoryModule, MemoryTensor
from .common import ModelHeadConfig


class ModelHeadMemoryModel(MemoryModule):
    def __init__(
        self,
        config: ModelHeadConfig,
        dtype: MemoryDType = MemoryFloat32,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.embed_tokens = EmbeddingMemoryModel(
            num_embeddings=self.cfg.vocab_size,
            embedding_dim=self.cfg.hidden_size,
            dtype=dtype,
        )

    def train_to_embedding_tokens(self, env: Env, input_tokens_tensor: MemoryHolder[MemoryTensor], dtype: torch.dtype) -> MemoryHolder[MemoryTensor]:
        assert input_tokens_tensor.item.is_cpu
        device_input = env.trace(input_tokens_tensor.item.device())
        out = self.embed_tokens.forward(env, device_input)
        if out.item.dtype != get_dtype(dtype):
            out = env.trace(out.item.to(dtype))
        return out

    def infer_to_embedding_tokens(self, env: Env, input_tokens_tensor: MemoryHolder[MemoryTensor], dtype: MemoryType) -> MemoryHolder[MemoryTensor]:
        assert input_tokens_tensor.item.is_cpu
        device_input = env.trace(input_tokens_tensor.item.device())
        out = self.embed_tokens.forward(env, device_input)
        if out.item.dtype != get_dtype(dtype):
            out = env.trace(out.item.to(dtype))
        return out
