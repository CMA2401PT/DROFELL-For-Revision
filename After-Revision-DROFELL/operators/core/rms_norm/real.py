import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype | None = None,
    ):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps
        self.use_fused_cuda_ops = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        needs_autograd = torch.is_grad_enabled() and hidden_states.requires_grad
        can_use_fused = (
            self.use_fused_cuda_ops
            and hidden_states.is_cuda
            and hidden_states.dtype == self.weight.dtype
            and not needs_autograd
        )
        if can_use_fused:
            from cuda_extends import vllm_ops

            output = torch.empty_like(hidden_states)
            vllm_ops.rms_norm(
                output,
                hidden_states,
                self.weight,
                self.variance_epsilon,
            )
            return output
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def forward_add(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add residual and normalize using the in-place fused CUDA kernel."""
        if not hidden_states.is_cuda:
            residual = hidden_states + residual
            return self.forward(residual), residual

        from cuda_extends import vllm_ops

        vllm_ops.fused_add_rms_norm(
            hidden_states,
            residual,
            self.weight,
            self.variance_epsilon,
        )
        return hidden_states, residual
