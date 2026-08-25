from torch import nn
from abc import ABC, abstractmethod
import torch


class Activation(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SiLU(Activation):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.silu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.silu(x)
