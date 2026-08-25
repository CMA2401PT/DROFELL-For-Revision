from abc import ABC, abstractmethod
from torch import nn
import torch


class Linear(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearMaker(ABC):
    @abstractmethod
    def make_linear(self, in_features: int, out_features: int, bias: bool) -> Linear:
        raise NotImplementedError
