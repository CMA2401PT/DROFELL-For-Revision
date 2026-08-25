from abc import ABC, abstractmethod
import math
from typing import Callable, List, Optional

import torch

from cuda_extends.exllama import TuningParams

from ..memory import MemoryModule, MemoryTensor


class LinearMakerMemoryModel(ABC):
    @abstractmethod
    def make_linear(self, in_features: int, out_features: int, bias: bool) -> MemoryModule:
        raise NotImplementedError


class CacheV2MemoryModel(MemoryModule):
    def __init__(self) -> None:
        super().__init__()
        self.temp_weight_size: int = 0
        self._temp_weight: Optional[MemoryTensor] = None
        self.cache_created = False

    def update_temp_weight_size(self, temp_weight_size: int):
        if temp_weight_size > self.temp_weight_size:
            self.cache_created = False
            self.temp_weight_size = temp_weight_size

    @property
    def temp_weight(self) -> MemoryTensor:
        assert self.cache_created
        assert self._temp_weight is not None
        return self._temp_weight

    def init_cache(self):
        self._temp_weight = MemoryTensor(size=(1, self.temp_weight_size), dtype=torch.float16)
        self.cache_created = True


class QuantLinearMemoryModel(MemoryModule):
    def __init__(self, group_size: int, in_features: int, out_features: int, bias: bool, use_act_order: bool):
        super().__init__()
        self.infeatures = in_features
        self.outfeatures = out_features
        self.use_act_order = use_act_order
        self.bits = 4
        self.group_size = group_size if group_size != -1 else in_features
        self.maxq = 2 ** self.bits - 1
        self.maxq = 2 ** self.bits - 1

        assert in_features % 32 == 0
        assert in_features % self.group_size == 0
        assert out_features % 32 == 0
        self.qweight = MemoryTensor((in_features // 32 * self.bits, out_features), dtype=torch.int32)
        self.qzeros = MemoryTensor((math.ceil(in_features / self.group_size), out_features // 32 * self.bits), dtype=torch.int32)
        self.scales = MemoryTensor((math.ceil(in_features / self.group_size), out_features), dtype=torch.float16)
        self.g_idx = MemoryTensor((in_features,), dtype=torch.int32)
        self.cache_v2: CacheV2MemoryModel

    def post_init(self, tunning_params: TuningParams, cache: CacheV2MemoryModel):
        self.cache_v2 = cache
        cache.update_temp_weight_size(self.qweight.numel()*8)


class QuantLinearMakerMemoryModel(LinearMakerMemoryModel):
    def __init__(self, quantization_group_size: int, use_act_order: bool) -> None:
        self.quantization_group_size = quantization_group_size
        self.use_act_order = use_act_order
        self.post_inited: bool = False
        self.temp_weight_size = 0
        self.quant_linears: List[QuantLinearMemoryModel] = []

    def make_linear(self, in_features: int, out_features: int, bias: bool = False) -> QuantLinearMemoryModel:
        linear = QuantLinearMemoryModel(group_size=self.quantization_group_size,
                                        in_features=in_features, out_features=out_features,
                                        bias=bias,
                                        use_act_order=self.use_act_order)
        self.quant_linears.append(linear)
        return linear

    def _apply(self, func: Callable[[QuantLinearMemoryModel], None]):
        [func(l) for l in self.quant_linears]
        return self

    def post_init(self, tunning_params: TuningParams, shared_cache: CacheV2MemoryModel):
        assert not self.post_inited
        self.post_inited = True
        self._apply(lambda quant_linear: quant_linear.post_init(tunning_params, shared_cache))
        return self
