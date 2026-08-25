from torch import nn
import torch
from ..linear.real import LinearMaker
from ..activations.real import SiLU


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, linear_maker: LinearMaker):
        nn.Module.__init__(self)
        assert hidden_act == "silu"
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = linear_maker.make_linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = linear_maker.make_linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = linear_maker.make_linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = SiLU()
        self.use_fused_cuda_ops = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up_proj = getattr(self, "gate_up_proj", None)
        if gate_up_proj is None:
            hidden_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        else:
            gate_up = gate_up_proj(x)
            needs_autograd = torch.is_grad_enabled() and gate_up.requires_grad
            if self.use_fused_cuda_ops and gate_up.is_cuda and not needs_autograd:
                from cuda_extends import vllm_ops

                hidden_states = torch.empty(
                    *gate_up.shape[:-1],
                    self.intermediate_size,
                    dtype=gate_up.dtype,
                    device=gate_up.device,
                )
                vllm_ops.silu_and_mul(hidden_states, gate_up)
            else:
                gate, up = gate_up.split(self.intermediate_size, dim=-1)
                hidden_states = self.act_fn(gate) * up
        down_proj = self.down_proj(hidden_states)
        return down_proj
