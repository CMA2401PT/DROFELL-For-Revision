"""Thin operator adapter for the Marlin GPTQ runtime."""

from __future__ import annotations

import math
from typing import Callable, cast

import torch
from torch import nn

from cuda_extends.marlin import (
    MarlinDenseCache,
    MarlinQ4ForwardOnlyLinearFunction,
    MarlinQ4LinearFunction,
    MarlinQ4Matrix,
    MarlinTuningParams,
    make_marlin_workspace,
)

from ... import flags
from .real import Linear, LinearMaker


class MarlinQuantLinear(Linear):
    """Checkpoint-facing GPTQ linear backed by ``MarlinQ4Matrix``."""

    def __init__(
        self,
        group_size: int,
        in_features: int,
        out_features: int,
        bias: bool,
        use_act_order: bool,
    ) -> None:
        nn.Module.__init__(self)
        if bias:
            raise ValueError("the Marlin decode backend does not support bias")
        if in_features % 32 != 0 or out_features % 32 != 0:
            raise ValueError("Marlin GPTQ dimensions must be multiples of 32")

        self.infeatures = in_features
        self.outfeatures = out_features
        self.group_size = in_features if group_size == -1 else group_size
        if in_features % self.group_size != 0:
            raise ValueError("in_features must be divisible by group_size")
        self.use_act_order = use_act_order and self.group_size < in_features

        self.register_buffer(
            "qweight",
            torch.zeros((in_features // 8, out_features), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (math.ceil(in_features / self.group_size), out_features // 8),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(in_features / self.group_size), out_features),
                dtype=torch.float16,
            ),
        )
        self.register_buffer(
            "g_idx",
            torch.tensor(
                [i // self.group_size for i in range(in_features)],
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "workspace",
            make_marlin_workspace(self.qweight.device),
            persistent=False,
        )
        self.bias = None
        self.q4_matrix: MarlinQ4Matrix | None = None

    def post_init(
        self,
        tuning_params: MarlinTuningParams,
        cache: MarlinDenseCache,
    ) -> None:
        if self.q4_matrix is not None:
            raise RuntimeError("MarlinQuantLinear.post_init called twice")
        q4_matrix = MarlinQ4Matrix(
            self.qweight,
            self.qzeros,
            self.scales,
            self.infeatures,
            self.outfeatures,
            self.g_idx,
            self.use_act_order,
            self.workspace,
            tuning_params,
            cache,
        )

        # Keep every persistent runtime tensor registered on the module so
        # swapping, memory accounting, and module lifetime see the same state.
        self.qweight = q4_matrix.qweight
        self.qzeros = q4_matrix.qzeros
        self.scales = q4_matrix.scales
        self.g_idx = q4_matrix.g_idx
        self.register_buffer(
            "g_idx_sort_indices", q4_matrix.g_idx_sort_indices
        )
        self.register_buffer(
            "dequant_weight_permutation",
            q4_matrix.dequant_weight_permutation,
        )
        self.register_buffer(
            "dequant_scale_column_map",
            q4_matrix.dequant_scale_column_map,
        )
        self.q4_matrix = q4_matrix

    def _matrix(self) -> MarlinQ4Matrix:
        if self.q4_matrix is None:
            raise RuntimeError("MarlinQuantLinear must be repacked before use")
        return self.q4_matrix

    @property
    def tuning_params(self) -> MarlinTuningParams:
        return self._matrix().tuning_params

    def reconstruct_weight(self, output: torch.Tensor) -> torch.Tensor:
        return self._matrix().reconstruct_weight(output)

    def dense_forward(
        self,
        x: torch.Tensor,
        dense_weight: torch.Tensor,
    ) -> torch.Tensor:
        return self._matrix().dense_matmul(x, dense_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        matrix = self._matrix()
        input_dtype = x.dtype
        marlin_input = x.to(matrix.scales.dtype)
        if torch.is_grad_enabled() and marlin_input.requires_grad:
            cache_weight = matrix.tuning_params.reuse_reconstruct_cache(
                flags.REUSE_RECONSTRUCT_CACHE_WHEN_GRAD
            )
            output = MarlinQ4LinearFunction.apply(
                marlin_input,
                matrix,
                cache_weight,
            )
        else:
            output = MarlinQ4ForwardOnlyLinearFunction.apply(
                marlin_input,
                matrix,
            )
        return cast(torch.Tensor, output).to(input_dtype)


class MarlinQuantLinearMaker(LinearMaker):
    def __init__(
        self,
        quantization_group_size: int,
        use_act_order: bool,
    ) -> None:
        self.quantization_group_size = quantization_group_size
        self.use_act_order = use_act_order
        self.quant_linears: list[MarlinQuantLinear] = []
        self.post_inited = False

    def _apply(self, func: Callable[[MarlinQuantLinear], None]) -> None:
        for linear in self.quant_linears:
            func(linear)

    def post_init(
        self,
        tuning_params: MarlinTuningParams,
        shared_cache: MarlinDenseCache,
    ) -> "MarlinQuantLinearMaker":
        if self.post_inited:
            raise RuntimeError("MarlinQuantLinearMaker.post_init called twice")
        if not self.quant_linears:
            raise RuntimeError("Marlin maker contains no quantized linears")
        self.post_inited = True
        self._apply(
            lambda linear: linear.post_init(tuning_params, shared_cache)
        )
        return self

    def set_tuning_param(
        self,
        tuning_params: MarlinTuningParams,
    ) -> "MarlinQuantLinearMaker":
        self._apply(
            lambda linear: linear._matrix().set_tuning_params(tuning_params)
        )
        return self

    def merge(
        self,
        linears: list[MarlinQuantLinear],
    ) -> MarlinQuantLinear:
        if not linears:
            raise ValueError("cannot merge an empty list of Marlin linears")
        first = linears[0]
        if any(linear.infeatures != first.infeatures for linear in linears):
            raise ValueError("merged Marlin linears must share in_features")
        if any(linear.group_size != first.group_size for linear in linears):
            raise ValueError("merged Marlin linears must share group_size")
        if any(linear.use_act_order != first.use_act_order for linear in linears):
            raise ValueError("merged Marlin linears must share act-order mode")
        if any(not torch.equal(linear.g_idx, first.g_idx) for linear in linears[1:]):
            raise ValueError("merged Marlin linears must share g_idx")

        merged = self.make_linear(
            in_features=first.infeatures,
            out_features=sum(linear.outfeatures for linear in linears),
            bias=False,
        )
        merged.qweight = torch.cat(
            [linear.qweight for linear in linears], dim=1
        ).contiguous()
        merged.qzeros = torch.cat(
            [linear.qzeros for linear in linears], dim=1
        ).contiguous()
        merged.scales = torch.cat(
            [linear.scales for linear in linears], dim=1
        ).contiguous()
        merged.g_idx = first.g_idx

        for linear in linears:
            self.quant_linears.remove(linear)
        return merged

    def make_linear(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> MarlinQuantLinear:
        linear = MarlinQuantLinear(
            group_size=self.quantization_group_size,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            use_act_order=self.use_act_order,
        )
        self.quant_linears.append(linear)
        return linear
