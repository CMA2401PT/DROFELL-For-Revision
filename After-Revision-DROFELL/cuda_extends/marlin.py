"""Marlin GPTQ runtime, dispatch policy, and frozen-weight autograd support."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.autograd import Function

from . import vllm_ops

from .prebuilt_marlin_extend import (
    marlin_dense_matmul_out as _marlin_dense_matmul_out,
)
from .prebuilt_marlin_extend import marlin_dequant_into as _marlin_dequant_into


@dataclass(frozen=True)
class MarlinTuningParams:
    """Choose packed Marlin or reconstruct-then-cuBLAS by flattened M.

    ``reconstruct_threshold`` applies to inference and no-cache PEFT;
    ``grad_reconstruct_threshold`` applies when PEFT retains the reconstructed
    forward weight for backward. ``None`` disables dense forward dispatch for
    that path. Defaults are the measured RTX 3090 Llama2-70B crossover values.
    """

    reconstruct_threshold: int | None = 768
    grad_reconstruct_threshold: int | None = 128
    reuse_reconstruct_cache_when_grad: bool | None = None

    def __post_init__(self) -> None:
        for name in ("reconstruct_threshold", "grad_reconstruct_threshold"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")

    def use_reconstructed_forward(
        self,
        flattened_tokens: int,
        *,
        cache_weight: bool,
    ) -> bool:
        threshold = (
            self.grad_reconstruct_threshold
            if cache_weight
            else self.reconstruct_threshold
        )
        return threshold is not None and flattened_tokens >= threshold

    def reuse_reconstruct_cache(self, default: bool) -> bool:
        override = self.reuse_reconstruct_cache_when_grad
        return default if override is None else override


class MarlinDenseCache:
    """Shared stream-ordered FP16 reconstruction workspace."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.temp_weight_elements = 0
        self._temp_weight: torch.Tensor | None = None

    def update(self, in_features: int, out_features: int) -> None:
        if self._temp_weight is not None:
            raise RuntimeError("cannot resize initialized Marlin dense cache")
        self.temp_weight_elements = max(
            self.temp_weight_elements,
            in_features * out_features,
        )

    def init_cache(self) -> None:
        if self._temp_weight is not None:
            raise RuntimeError("Marlin dense cache initialized twice")
        self._temp_weight = torch.empty(
            self.temp_weight_elements,
            dtype=torch.float16,
            device=self.device,
        )

    def weight(self, in_features: int, out_features: int) -> torch.Tensor:
        if self._temp_weight is None:
            raise RuntimeError("Marlin dense cache is not initialized")
        elements = in_features * out_features
        if elements > self._temp_weight.numel():
            raise RuntimeError("Marlin dense cache is too small")
        return self._temp_weight[:elements].view(in_features, out_features)


def make_marlin_workspace(device: torch.device) -> torch.Tensor:
    """Allocate the small per-matrix lock workspace used by Marlin GEMM."""

    if device.type != "cuda":
        raise ValueError("Marlin workspace must be initialized on CUDA")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * 4, dtype=torch.int32, device=device)


def _permute_scales(
    scales: torch.Tensor,
    in_features: int,
    out_features: int,
    group_size: int,
) -> torch.Tensor:
    if group_size < in_features:
        permutation = [i + 8 * j for i in range(8) for j in range(8)]
    else:
        permutation = [
            2 * i + j
            for i in range(4)
            for j in (0, 1, 8, 9, 16, 17, 24, 25)
        ]
    return scales.reshape(-1, len(permutation))[:, permutation].reshape(
        -1, out_features
    ).contiguous()


def _make_dequant_weight_permutation(
    device: torch.device,
) -> torch.Tensor:
    """Build the module-owned Marlin dequantization permutation."""

    permutation: list[int] = []
    for index in range(32):
        tile: list[int] = []
        column = index // 4
        for block in (0, 1):
            for row in (
                2 * (index % 4),
                2 * (index % 4) + 1,
                2 * (index % 4 + 4),
                2 * (index % 4 + 4) + 1,
            ):
                tile.append(16 * row + column + 8 * block)
        for offset in range(4):
            permutation.extend(value + 256 * offset for value in tile)
    interleave = (0, 2, 4, 6, 1, 3, 5, 7)
    permutation = [
        permutation[start + index]
        for start in range(0, len(permutation), len(interleave))
        for index in interleave
    ]
    return torch.tensor(permutation, dtype=torch.int16, device=device)


def _make_dequant_scale_column_map(
    out_features: int,
    group_size: int,
    in_features: int,
    device: torch.device,
) -> torch.Tensor:
    group_permutation = [i + 8 * j for i in range(8) for j in range(8)]
    channel_permutation = [
        2 * i + j
        for i in range(4)
        for j in (0, 1, 8, 9, 16, 17, 24, 25)
    ]
    permutation = (
        group_permutation if group_size < in_features else channel_permutation
    )
    permutation_tensor = torch.tensor(permutation, dtype=torch.long)
    inverse = torch.empty(len(permutation), dtype=torch.long)
    inverse[permutation_tensor] = torch.arange(len(permutation))
    columns = torch.arange(out_features)
    block = columns // len(permutation)
    within = inverse[columns % len(permutation)]
    return (block * len(permutation) + within).to(
        dtype=torch.int32,
        device=device,
    )


def _dense_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    transpose_weight: bool,
) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    output_features = weight.shape[0] if transpose_weight else weight.shape[1]
    output = torch.empty(
        (x_2d.shape[0], output_features),
        dtype=torch.float16,
        device=x.device,
    )
    _marlin_dense_matmul_out(x_2d, weight, output, transpose_weight)
    return output.reshape(*x.shape[:-1], output_features)


class MarlinQ4Matrix:
    """Repacked Marlin matrix plus its dispatch and reconstruction metadata."""

    def __init__(
        self,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        in_features: int,
        out_features: int,
        g_idx: torch.Tensor,
        use_act_order: bool,
        workspace: torch.Tensor,
        tuning_params: MarlinTuningParams,
        cache: MarlinDenseCache,
    ) -> None:
        if qweight.device.type != "cuda":
            raise ValueError("Marlin weights must be initialized on CUDA")
        if not bool(torch.all(qzeros == 0x77777777).item()):
            raise ValueError("Marlin requires symmetric GPTQ zero points")

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = in_features // scales.shape[0]
        self.use_act_order = use_act_order and self.group_size < in_features
        self.tuning_params = tuning_params
        self.cache = cache

        if self.use_act_order:
            sort_indices = torch.argsort(g_idx).to(torch.int32)
            marlin_g_idx = g_idx[sort_indices].to(torch.int32)
        else:
            sort_indices = torch.empty(
                0, dtype=torch.int32, device=qweight.device
            )
            marlin_g_idx = torch.empty(
                0, dtype=torch.int32, device=qweight.device
            )

        self.qweight = vllm_ops.gptq_marlin_repack(
            qweight.contiguous(),
            sort_indices,
            in_features,
            out_features,
            4,
            False,
        )
        self.scales = _permute_scales(
            scales.contiguous(),
            in_features,
            out_features,
            self.group_size,
        )
        self.qzeros = torch.empty(
            0, dtype=torch.int32, device=qweight.device
        )
        self.g_idx = marlin_g_idx
        self.g_idx_sort_indices = sort_indices
        if workspace.device != qweight.device or workspace.dtype != torch.int32:
            raise ValueError("Marlin workspace has the wrong device or dtype")
        self.workspace = workspace
        self.dequant_weight_permutation = _make_dequant_weight_permutation(
            qweight.device
        )
        self.dequant_scale_column_map = _make_dequant_scale_column_map(
            out_features,
            self.group_size,
            in_features,
            qweight.device,
        )
        cache.update(in_features, out_features)

    def set_tuning_params(self, tuning_params: MarlinTuningParams) -> None:
        self.tuning_params = tuning_params

    def use_reconstructed_forward(
        self,
        flattened_tokens: int,
        *,
        cache_weight: bool,
    ) -> bool:
        return self.tuning_params.use_reconstructed_forward(
            flattened_tokens,
            cache_weight=cache_weight,
        )

    def reconstruct_weight(
        self,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty(
                (self.in_features, self.out_features),
                dtype=torch.float16,
                device=self.qweight.device,
            )
        _marlin_dequant_into(
            self.qweight,
            self.scales,
            self.dequant_weight_permutation,
            self.dequant_scale_column_map,
            self.g_idx_sort_indices,
            output,
            self.in_features,
            self.out_features,
            self.group_size,
        )
        return output

    def direct_matmul(self, x: torch.Tensor) -> torch.Tensor:
        x_2d = x.reshape(-1, self.in_features).contiguous()
        output = vllm_ops.marlin_gemm(
            x_2d,
            None,
            self.qweight,
            None,
            self.scales,
            None,
            None,
            self.qzeros,
            self.g_idx,
            self.g_idx_sort_indices,
            self.workspace,
            vllm_ops.UINT4B8_ID,
            x_2d.shape[0],
            self.out_features,
            self.in_features,
            True,
            False,
            True,
            False,
        )
        return output.reshape(*x.shape[:-1], self.out_features)

    def dense_matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        transpose_weight: bool = False,
    ) -> torch.Tensor:
        return _dense_matmul(x, weight, transpose_weight=transpose_weight)

    def forward_matmul(
        self,
        x: torch.Tensor,
        *,
        cache_weight: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        flattened_tokens = x.numel() // self.in_features
        if not self.use_reconstructed_forward(
            flattened_tokens,
            cache_weight=cache_weight,
        ):
            return self.direct_matmul(x), None

        weight = (
            self.reconstruct_weight()
            if cache_weight
            else self.cache.weight(self.in_features, self.out_features)
        )
        if not cache_weight:
            self.reconstruct_weight(weight)
        return self.dense_matmul(x, weight), weight if cache_weight else None

    def transpose_matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if weight is None:
            weight = self.cache.weight(self.in_features, self.out_features)
            self.reconstruct_weight(weight)
        return self.dense_matmul(x, weight, transpose_weight=True)


class MarlinQ4LinearFunction(Function):
    """Autograd for a frozen Marlin matrix; only the activation has a grad."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q4_matrix: MarlinQ4Matrix,
        cache_weight: bool,
    ) -> torch.Tensor:
        output, cached_weight = q4_matrix.forward_matmul(
            x,
            cache_weight=cache_weight,
        )
        ctx.q4_matrix = q4_matrix
        ctx.x_dtype = x.dtype
        if cached_weight is None:
            ctx.save_for_backward()
        else:
            ctx.save_for_backward(cached_weight)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q4_matrix: MarlinQ4Matrix = ctx.q4_matrix
        cached_weight = ctx.saved_tensors[0] if ctx.saved_tensors else None
        grad_x = q4_matrix.transpose_matmul(
            grad_output.to(q4_matrix.scales.dtype).contiguous(),
            cached_weight,
        ).to(ctx.x_dtype)
        return grad_x, None, None


class MarlinQ4ForwardOnlyLinearFunction(Function):
    """Forward-only wrapper matching the ExLlama backend interface."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q4_matrix: MarlinQ4Matrix,
    ) -> torch.Tensor:
        output, _ = q4_matrix.forward_matmul(x, cache_weight=False)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise RuntimeError("MarlinQ4ForwardOnlyLinearFunction has no backward")
