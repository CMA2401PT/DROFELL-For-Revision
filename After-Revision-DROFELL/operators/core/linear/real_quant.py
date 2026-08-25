import math
from typing import Callable, List, cast
from torch import nn
import torch
from cuda_extends.exllama import CacheV2, ExllamaQ4ForwardOnlyLinearFunction, ExllamaQ4LinearFunction, Q4MatrixV2, TuningParams
from ... import flags
from .real import Linear, LinearMaker


class QuantLinear(Linear):
    def __init__(self, group_size: int, in_features: int, out_features: int, bias: bool, use_act_order: bool):
        nn.Module.__init__(self)
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

        self.register_buffer(
            'qweight',
            torch.zeros((in_features // 32 * self.bits,
                        out_features), dtype=torch.int32)
        )
        self.register_buffer(
            'qzeros',
            torch.zeros((math.ceil(in_features / self.group_size),
                        out_features // 32 * self.bits), dtype=torch.int32)
        )
        self.register_buffer(
            'scales',
            torch.zeros((math.ceil(in_features / self.group_size),
                        out_features), dtype=torch.float16)
        )
        self.register_buffer(
            'g_idx',
            torch.tensor(
                [i // self.group_size for i in range(in_features)], dtype=torch.int32)
        )

        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def post_init(self, tunning_params: TuningParams, cache: CacheV2):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None
        self.q4_matrix = Q4MatrixV2(
            self.qweight,
            self.qzeros,
            self.scales,
            self.infeatures, self.outfeatures,
            self.g_idx.cpu() if self.use_act_order else None,
            self.use_act_order,
            tuning_params=tunning_params,
            cache=cache
        )
        if flags.RECONSTRUCTE_WEIGTH_FLOAT16:
            self.q4_matrix.reconstruct_weight(dtype=torch.float16)
        if flags.RECONSTRUCTE_WEIGTH_BFLOAT16:
            self.q4_matrix.reconstruct_weight(dtype=torch.bfloat16)

    def get_reconstruct_weights(self):
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0)
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        torch.bitwise_and(zeros, (2 ** self.bits) - 1, out=zeros)

        zeros = zeros + 1
        zeros = zeros.reshape(self.scales.shape)

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
            self.wf.unsqueeze(-1)
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        torch.bitwise_and(weight, (2 ** self.bits) - 1, out=weight)

        weight = weight.reshape(
            weight.shape[0] * weight.shape[1], weight.shape[2])
        num_itr = self.g_idx.shape[0] // self.infeatures
        if num_itr == 1:
            weights = (self.scales[self.g_idx.long()] *
                       (weight - zeros[self.g_idx.long()]))
        else:
            num_dim = self.g_idx.shape[0] // num_itr
            weights_splits: List[torch.Tensor] = []
            for i in range(num_itr):
                scale_i = self.scales[:, i * num_dim:(i + 1) * num_dim]
                weight_i = weight[:, i * num_dim:(i + 1) * num_dim]
                zeros_i = zeros[:, i * num_dim:(i + 1) * num_dim]
                g_idx_i = self.g_idx[i * num_dim:(i + 1) * num_dim]
                weights_splits.append(scale_i[g_idx_i.long()]
                                      * (weight_i - zeros_i[g_idx_i.long()]))
            weights = torch.cat(weights_splits, dim=1)
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.requires_grad:
            return cast(torch.Tensor, ExllamaQ4LinearFunction.apply(x, self.q4_matrix, self.bias, flags.REUSE_RECONSTRUCT_CACHE_WHEN_GRAD))
        else:
            return cast(torch.Tensor, ExllamaQ4ForwardOnlyLinearFunction.apply(x, self.q4_matrix, self.bias))


class QuantLinearMaker(LinearMaker):
    def __init__(self, quantization_group_size: int, use_act_order: bool) -> None:
        self.quantization_group_size = quantization_group_size
        self.use_act_order = use_act_order
        self.quant_linears: list[QuantLinear] = []
        self.post_inited: bool = False

    def _apply(self, func: Callable[[QuantLinear], None]):
        [func(l) for l in self.quant_linears]
        return self

    def post_init(self, tunning_params: TuningParams, shared_cache: CacheV2):
        assert not self.post_inited
        self.post_inited = True
        self._apply(lambda quant_linear: quant_linear.post_init(tunning_params, shared_cache))
        return self

    def set_tuning_param(self, tuning_param: TuningParams):
        self._apply(lambda quant_linear: quant_linear.q4_matrix.set_tuning_params(tuning_param))
        return self

    def make_linear(self, in_features: int, out_features: int, bias: bool = False) -> QuantLinear:
        linear = QuantLinear(group_size=self.quantization_group_size,
                             in_features=in_features, out_features=out_features,
                             bias=bias,
                             use_act_order=self.use_act_order)
        self.quant_linears.append(linear)
        return linear
