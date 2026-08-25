import os
import torch
from torch import nn
from torch.nn import Module
from safetensors import safe_open  # type: ignore
from .common import ModelHeadConfig
from ...container import Container


class ModelHead(Module):
    def __init__(
        self,
        config: ModelHeadConfig,
        dtype: torch.dtype | None = None,
    ) -> None:
        Module.__init__(self)
        self.cfg = config
        self.embed_tokens = nn.Embedding(self.cfg.vocab_size,
                                         self.cfg.hidden_size,
                                         self.cfg.pad_token_id,
                                         dtype=dtype)
        self._device = torch.zeros(0).device

    @property
    def vocab_size(self):
        return self.cfg.vocab_size

    def load_from_data_dir(self, model_data: str):
        with safe_open(os.path.join(model_data, "embed_tokens.safetensors"), framework="pt") as f:
            self.embed_tokens.load_state_dict({k: f.get_tensor(k) for k in f.keys()})

        def freeze(module: nn.Module):
            for _, parameter in module.named_parameters():
                parameter.requires_grad_(False)
            for _, buffer in module.named_buffers():
                buffer.requires_grad_(False)
            for _, submodule in module.named_children():
                freeze(submodule)

        self.embed_tokens.eval()
        freeze(self.embed_tokens)

    @property
    def device(self) -> torch.device:
        return self._device

    async def infer_to_embedding_tokens(self, container: Container, input_tokens_tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        def _to_embedding_tokens(input_tokens_tensor: torch.Tensor, dtype: torch.dtype):
            return self.embed_tokens(input_tokens_tensor).to(dtype)
        with container.mode("inference"):
            return await container.step(
                "to_embedding_tokens",
                _to_embedding_tokens,
                input_tokens_tensor=input_tokens_tensor,
                dtype=dtype
            )

    async def train_to_embedding_tokens(self, container: Container, input_tokens_tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        def _to_embedding_tokens(input_tokens_tensor: torch.Tensor, dtype: torch.dtype):
            return self.embed_tokens(input_tokens_tensor).to(dtype)
        with container.mode("no_grad"):
            return await container.step(
                "to_embedding_tokens",
                _to_embedding_tokens,
                input_tokens_tensor=input_tokens_tensor,
                dtype=dtype
            )
