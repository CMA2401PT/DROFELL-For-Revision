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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
