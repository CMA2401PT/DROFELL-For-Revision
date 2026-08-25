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


class MarlinQuantLinearMemoryModel(MemoryModule):
    """Long-lived tensors retained by one repacked Marlin projection."""

    def __init__(
        self,
        group_size: int,
        in_features: int,
        out_features: int,
        bias: bool,
        use_act_order: bool,
        workspace_elements: int,
    ) -> None:
        super().__init__()
        if bias:
            raise ValueError("the Marlin memory model does not support bias")
        if in_features % 32 != 0 or out_features % 32 != 0:
            raise ValueError("Marlin GPTQ dimensions must be multiples of 32")
        self.infeatures = in_features
        self.outfeatures = out_features
        self.group_size = in_features if group_size == -1 else group_size
        # Match MarlinQuantLinear: act order is meaningless when there is only
        # one quantization group, and vLLM rejects that representation.
        self.use_act_order = use_act_order and self.group_size < in_features
        groups = math.ceil(in_features / self.group_size)

        # gptq_marlin_repack changes the shape but preserves qweight numel.
        self.qweight = MemoryTensor(
            (in_features // 16, out_features * 2), dtype=torch.int32
        )
        self.qzeros = MemoryTensor((0,), dtype=torch.int32)
        self.scales = MemoryTensor((groups, out_features), dtype=torch.float16)
        index_size = in_features if self.use_act_order else 0
        self.g_idx = MemoryTensor((index_size,), dtype=torch.int32)
        self.g_idx_sort_indices = MemoryTensor((index_size,), dtype=torch.int32)
        self.workspace = MemoryTensor((workspace_elements,), dtype=torch.int32)
        # Persistent lookup tables used by the tiled Marlin dequantizer.  They
        # are module-owned rather than process-global CUDA caches so the memory
        # model and tensor lifetime agree.
        self.dequant_weight_permutation = MemoryTensor(
            (1024,), dtype=torch.int16
        )
        self.dequant_scale_column_map = MemoryTensor(
            (out_features,), dtype=torch.int32
        )


class MarlinDenseCacheMemoryModel(MemoryModule):
    """One shared FP16 reconstruction workspace per decoder group."""

    def __init__(self) -> None:
        super().__init__()
        self.temp_weight_elements = 0
        self._temp_weight: Optional[MemoryTensor] = None
        self.cache_created = False

    def update(self, in_features: int, out_features: int) -> None:
        if self.cache_created:
            raise RuntimeError("cannot resize initialized Marlin dense cache")
        self.temp_weight_elements = max(
            self.temp_weight_elements,
            in_features * out_features,
        )

    def init_cache(self) -> None:
        if self.cache_created:
            raise RuntimeError("Marlin dense cache initialized twice")
        self._temp_weight = MemoryTensor(
            (self.temp_weight_elements,), dtype=torch.float16
        )
        self.cache_created = True


class MarlinQuantLinearMakerMemoryModel(LinearMakerMemoryModel):
    def __init__(
        self,
        quantization_group_size: int,
        use_act_order: bool,
        workspace_elements: int,
    ) -> None:
        self.quantization_group_size = quantization_group_size
        self.use_act_order = use_act_order
        self.workspace_elements = workspace_elements
        self.dense_cache = MarlinDenseCacheMemoryModel()
        self.quant_linears: List[MarlinQuantLinearMemoryModel] = []
        self.post_inited = False

    def make_linear(
        self, in_features: int, out_features: int, bias: bool = False
    ) -> MarlinQuantLinearMemoryModel:
        linear = MarlinQuantLinearMemoryModel(
            group_size=self.quantization_group_size,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            use_act_order=self.use_act_order,
            workspace_elements=self.workspace_elements,
        )
        self.quant_linears.append(linear)
        self.dense_cache.update(in_features, out_features)
        return linear

    def post_init(self) -> None:
        if self.post_inited:
            raise RuntimeError(
                "MarlinQuantLinearMakerMemoryModel.post_init called twice"
            )
        self.post_inited = True
        self.dense_cache.init_cache()

    def merge(
        self, linears: List[MarlinQuantLinearMemoryModel]
    ) -> MarlinQuantLinearMemoryModel:
        if not linears:
            raise ValueError("cannot merge an empty list of Marlin linears")
        first = linears[0]
        if any(linear.infeatures != first.infeatures for linear in linears):
            raise ValueError("merged Marlin linears must share in_features")
        merged = self.make_linear(
            in_features=first.infeatures,
            out_features=sum(linear.outfeatures for linear in linears),
            bias=False,
        )
        for linear in linears:
            self.quant_linears.remove(linear)
        return merged
