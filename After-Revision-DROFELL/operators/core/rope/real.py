from typing import Dict, Tuple
import torch
from .common import RopeConfig


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.use_fused_cuda_ops = False
        self.position_ids_dtype = torch.int32
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float16)
        self._dtype_cached: Dict[torch.dtype, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype),
            persistent=False,
        )
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def check_len(self, current_len: int):
        if current_len == self.max_seq_len_cached-1:
            self._set_cos_sin_cache(self.max_seq_len_cached*2, self.cos_cached.device, self.cos_cached.dtype)

    def get_pregen_sin_cos(self, t: torch.dtype):
        ret = self._dtype_cached.get(t)
        if ret is not None:
            return ret
        else:
            ret = (self.cos_cached.to(dtype=t), self.sin_cached.to(dtype=t))
            self._dtype_cached[t] = ret
            return ret

    @staticmethod
    def rotate_half(x: torch.Tensor):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: [seq_len,num_query_heads,head_dim]
        k: [seq_len,num_key_heads,head_dim]
        position_ids: [seq_len]
        """
        assert len(position_ids.shape) == 1
        assert len(q.shape) == 3 and len(k.shape) == 3
        if (
            self.use_fused_cuda_ops
            and q.is_cuda
            and hasattr(self, "cos_sin_cache")
            and q.dtype == self.cos_sin_cache.dtype
            and k.dtype == self.cos_sin_cache.dtype
            and position_ids.dtype == self.position_ids_dtype
            and not (
                torch.is_grad_enabled()
                and (q.requires_grad or k.requires_grad)
            )
        ):
            from cuda_extends import vllm_ops

            vllm_ops.rotary_embedding(
                position_ids,
                q,
                k,
                self.dim,
                self.cos_sin_cache,
                True,
            )
            return q, k
        cos, sin = self.get_pregen_sin_cos(q.dtype)
        cos_selected = cos[position_ids]  # [seq_len,dim]
        sin_selected = sin[position_ids]  # [seq_len,dim]
        cos = cos_selected.unsqueeze(1)  # [seq_len, 1, dim]
        sin = sin_selected.unsqueeze(1)  # [seq_len, 1, dim]
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, scaling_factor: float = 1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, scaling_factor: float = 1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


def make_rope_embedding(config: RopeConfig) -> RotaryEmbedding:
    if config.rope_scaling_type is None:
        return RotaryEmbedding(
            dim=config.head_dim, max_position_embeddings=config.max_seq_len,
            base=config.rope_theta
        )
    else:
        assert config.rope_scaling_factor is not None
        if config.rope_scaling_type == "linear":
            return LinearScalingRotaryEmbedding(
                dim=config.head_dim, max_position_embeddings=config.max_seq_len,
                base=config.rope_theta, scaling_factor=config.rope_scaling_factor
            )
        elif config.rope_scaling_type == "dynamic":
            return DynamicNTKScalingRotaryEmbedding(
                dim=config.head_dim, max_position_embeddings=config.max_seq_len,
                base=config.rope_theta, scaling_factor=config.rope_scaling_factor
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {config.rope_scaling_type}")
