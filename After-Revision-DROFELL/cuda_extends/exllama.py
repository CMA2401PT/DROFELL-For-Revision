import torch
from torch.autograd import Function

from typing import Any, NewType, Optional, Union, Tuple
import os
from .prebuilt_exllama_extend import make_q4
from .prebuilt_exllama_extend import q4_to_half_matrix, q4_to_bfloat16_matrix
from .prebuilt_exllama_extend import q4_to_half_matrix_v2, q4_to_bfloat16_matrix_v2
from .prebuilt_exllama_extend import column_remap, column_restore
from .prebuilt_exllama_extend import q4_matmul_cuda, q4_matmul_blas
from .prebuilt_exllama_extend import q4_matmul_cuda_v2, q4_matmul_blas_v2
from .prebuilt_exllama_extend import q4_transpose_matmul_blas
from .prebuilt_exllama_extend import q4_transpose_matmul_blas_v2
from .prebuilt_exllama_extend import mlp_forward_v2
from .prebuilt_exllama_extend import attention_forward_qkv_stage_v2, attention_forward_rope_stage_v2
from .prebuilt_exllama_extend import lora
from dataclasses import dataclass

Q4Ptr = NewType('Q4Ptr', int)


@dataclass
class TuningParams:
    # each block contains <<<RECONSTRUCT_THREADS_X,RECONSTRUCT_THREADS_Y>>> threads
    # each thread is in charge of 4 values (an int32) (4,1)
    RECONSTRUCT_THREADS_X: int = 128
    RECONSTRUCT_THREADS_Y: int = 1
    # each block contains <<<COLUMN_SHUFFLE_BLOCK_X,1>>> threads
    # each thread is in charge of COLUMN_SHUFFLE_BLOCK_Y values (COLUMN_SHUFFLE_BLOCK_Y,1)
    COLUMN_SHUFFLE_BLOCK_X: int = 256
    COLUMN_SHUFFLE_BLOCK_Y: int = 16
    # if X has shape [M,K], W has shape [K,N], X@W=>O[M,N]
    # each block contains <<<MATMUL_THREADS_X,MATMUL_THREADS_Y,1>>> threads
    # each thead is in charge of [1,MATMUL_BLOCK_SIZE_Z] in X and [MATMUL_BLOCK_SIZE_Z,1] in W and produce [1,1] value in O
    MATMUL_THREADS_X: int = 32
    MATMUL_THREADS_Y: int = 1
    MATMUL_BLOCK_SIZE_Z: Optional[int] = None
    MATMUL_GROUP_STEP: int = 32

    MATMAL_RECONSTRUCT_THRESHOLD: int = 8
    MATMAL_RECONSTRUCT_THRESHOLD_V2: int = 50

    NO_HALF2: bool = False
    NO_FUSE_REMAP: bool = True
    BLOCK_KN_SIZE: int = 128
    BLOCK_M_SIZE_MAX: int = 8


def make_q4_py(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, g_idx: Optional[torch.Tensor], cuda_x_map: Optional[torch.Tensor], device_idx: int, use_act: bool, v2_mode: bool) -> Q4Ptr:
    return make_q4(qweight, qzeros, scales, g_idx, cuda_x_map, device_idx, use_act, v2_mode)


def q4_to_half_matrix_py(q4: Q4Ptr, out_matrix: torch.Tensor, RECONSTRUCT_THREADS_X: int, RECONSTRUCT_THREADS_Y: int) -> None:
    return q4_to_half_matrix(q4, out_matrix, RECONSTRUCT_THREADS_X, RECONSTRUCT_THREADS_Y)


def q4_to_half_matrix_v2_py(q4: Q4Ptr, out_matrix: torch.Tensor, BLOCK_KN_SIZE: int) -> None:
    return q4_to_half_matrix_v2(q4, out_matrix, BLOCK_KN_SIZE)


def q4_to_bfloat16_matrix_py(q4: Q4Ptr, out_matrix: torch.Tensor, RECONSTRUCT_THREADS_X: int, RECONSTRUCT_THREADS_Y: int) -> None:
    return q4_to_bfloat16_matrix(q4, out_matrix, RECONSTRUCT_THREADS_X, RECONSTRUCT_THREADS_Y)


def q4_to_bfloat16_matrix_v2_py(q4: Q4Ptr, out_matrix: torch.Tensor, BLOCK_KN_SIZE: int) -> None:
    return q4_to_bfloat16_matrix_v2(q4, out_matrix, BLOCK_KN_SIZE)


def column_remap_py(x: torch.Tensor, remapped_x: torch.Tensor, x_map: torch.Tensor, COLUMN_SHUFFLE_BLOCK_X: int, COLUMN_SHUFFLE_BLOCK_Y: int) -> None:
    return column_remap(x, remapped_x, x_map, COLUMN_SHUFFLE_BLOCK_X, COLUMN_SHUFFLE_BLOCK_Y)


def column_restore_py(x: torch.Tensor, remapped_x: torch.Tensor, x_map: torch.Tensor, COLUMN_SHUFFLE_BLOCK_X: int, COLUMN_SHUFFLE_BLOCK_Y: int) -> None:
    return column_restore(x, remapped_x, x_map, COLUMN_SHUFFLE_BLOCK_X, COLUMN_SHUFFLE_BLOCK_Y)

# no_zero,no_half2 false no_fused_remap true


# def half_matmul_py(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, use_half2: bool, do_add: bool):
#     x_shape = x.shape
#     x = x.view(-1, x_shape[-1])
#     out = out.view(-1, w.shape[-1])
#     half_matmul(use_half2, do_add, x.data_ptr(), w.data_ptr(), out.data_ptr(), x.shape[0], x.shape[1], w.shape[1])


# def half_matmul_create_py(x: torch.Tensor, w: torch.Tensor,  use_half2: bool = True, do_add: bool = False):
#     x_shape = x.shape
#     x = x.view(-1, x_shape[-1])
#     out = torch.zeros((*x_shape[:-1], w.shape[-1]), dtype=torch.float16, device=x.device)
#     half_matmul(use_half2, do_add, x.data_ptr(), w.data_ptr(), out.data_ptr(), x.shape[0], x.shape[1], w.shape[1])
#     return out


def q4_matmul_cuda_py(x: torch.Tensor, q4: Q4Ptr, out: torch.Tensor, temp_x: torch.Tensor, no_zero: bool, no_half2: bool, no_fuse_remap: bool, MATMUL_THREADS_X: int, MATMUL_THREADS_Y: int, MATMUL_BLOCK_SIZE_Z: int, MATMUL_GROUP_STEP: int, COLUMN_SHUFFLE_BLOCK_X: int, COLUMN_SHUFFLE_BLOCK_Y: int, skip_remap: bool) -> None:
    return q4_matmul_cuda(x, q4, out, temp_x, no_zero, no_half2, no_fuse_remap, MATMUL_THREADS_X, MATMUL_THREADS_Y, MATMUL_BLOCK_SIZE_Z, MATMUL_GROUP_STEP, COLUMN_SHUFFLE_BLOCK_X, COLUMN_SHUFFLE_BLOCK_Y, skip_remap)


def q4_matmul_cuda_v2_py(x: torch.Tensor, q4: Q4Ptr, out: torch.Tensor, BLOCK_KN_SIZE: int, BLOCK_M_SIZE_MAX: int, no_zero: bool) -> None:
    return q4_matmul_cuda_v2(x, q4, out, BLOCK_KN_SIZE, BLOCK_M_SIZE_MAX, no_zero)

# # no_zero false


def q4_matmul_blas_py(x: torch.Tensor, q4: Q4Ptr, out: torch.Tensor, temp_weight: torch.Tensor, temp_x: torch.Tensor, no_zero: bool, COLUMN_SHUFFLE_BLOCK_X: int, COLUMN_SHUFFLE_BLOCK_Y: int, RECONSTRUCT_THREADS_X: int, RECONSTRUCT_THREADS_Y: int, skip_remap: bool, skip_reconstruct: bool, bfloat16: bool) -> None:
    return q4_matmul_blas(x, q4, out, temp_weight, temp_x, no_zero, COLUMN_SHUFFLE_BLOCK_X, COLUMN_SHUFFLE_BLOCK_Y, RECONSTRUCT_THREADS_X, RECONSTRUCT_THREADS_Y, skip_remap, skip_reconstruct, bfloat16)


def q4_matmul_blas_v2_py(x: torch.Tensor, q4: Q4Ptr, out: torch.Tensor, temp_weight: torch.Tensor, no_zero: bool, BLOCK_KN_SIZE: int, skip_reconstruct: bool, bfloat16: bool) -> None:
    return q4_matmul_blas_v2(x, q4, out, temp_weight, no_zero, BLOCK_KN_SIZE, skip_reconstruct, bfloat16)


def q4_transpose_matmul_blas_py(x: torch.Tensor, q4: Q4Ptr, out: torch.Tensor, temp_weight: torch.Tensor, temp_x: torch.Tensor, no_zero: bool, COLUMN_SHUFFLE_BLOCK_X: int, COLUMN_SHUFFLE_BLOCK_Y: int, RECONSTRUCT_THREADS_X: int, RECONSTRUCT_THREADS_Y: int, skip_remap: bool, skip_reconstruct: bool, bfloat16: bool) -> None:
    return q4_transpose_matmul_blas(x, q4, out, temp_weight, temp_x, no_zero, COLUMN_SHUFFLE_BLOCK_X, COLUMN_SHUFFLE_BLOCK_Y, RECONSTRUCT_THREADS_X, RECONSTRUCT_THREADS_Y, skip_remap, skip_reconstruct, bfloat16)


def q4_transpose_matmul_blas_v2_py(x: torch.Tensor, q4: Q4Ptr, out: torch.Tensor, temp_weight: torch.Tensor, no_zero: bool, BLOCK_KN_SIZE: int, skip_reconstruct: bool, bfloat16: bool) -> None:
    return q4_transpose_matmul_blas_v2(x, q4, out, temp_weight, no_zero, BLOCK_KN_SIZE, skip_reconstruct, bfloat16)


def mlp_forward_v2_py(layernorm: torch.Tensor, norm_epsilon: float, gate_q4: Q4Ptr, up_q4: Q4Ptr, down_q4: Q4Ptr, temp_rms_output: torch.Tensor, temp_gate_output: torch.Tensor, temp_up_output: torch.Tensor, temp_weight: torch.Tensor, x: torch.Tensor, output: torch.Tensor, BLOCK_KN_SIZE: int, BLOCK_M_SIZE_MAX: int, MATMAL_RECONSTRUCT_THRESHOLD_V2: int) -> None:
    return mlp_forward_v2(layernorm, norm_epsilon, gate_q4, up_q4, down_q4, temp_rms_output, temp_gate_output, temp_up_output, temp_weight, x, output, BLOCK_KN_SIZE, BLOCK_M_SIZE_MAX, MATMAL_RECONSTRUCT_THRESHOLD_V2)


def attention_forward_qkv_stage_v2_py(layernorm: torch.Tensor, norm_epsilon: float,
                                      q_proj: Q4Ptr, k_proj: Q4Ptr, v_proj: Q4Ptr,
                                      temp_rms_output: torch.Tensor, temp_q_output: torch.Tensor, temp_k_output: torch.Tensor, temp_v_output: torch.Tensor, temp_weight: torch.Tensor,
                                      x: torch.Tensor, BLOCK_KN_SIZE: int, BLOCK_M_SIZE_MAX: int, MATMAL_RECONSTRUCT_THRESHOLD_V2: int,
                                      q_len: int, batch_size: int):
    attention_forward_qkv_stage_v2(layernorm, norm_epsilon, q_proj, k_proj, v_proj, temp_rms_output, temp_q_output, temp_k_output, temp_v_output, temp_weight,
                                   x, BLOCK_KN_SIZE, BLOCK_M_SIZE_MAX, MATMAL_RECONSTRUCT_THRESHOLD_V2, q_len, batch_size)


def attention_forward_rope_stage_v2_py(temp_q_output: torch.Tensor, temp_k_output: torch.Tensor,  sin: torch.Tensor, cos: torch.Tensor,
                                       q_len: int, batch_size: int, num_heads: int, num_kv_heads: int, head_dim: int, past_len: int, past_lens: torch.Tensor):
    attention_forward_rope_stage_v2(temp_q_output, temp_k_output, sin, cos, q_len, batch_size, num_heads, num_kv_heads, head_dim, past_len, past_lens)


def attention_forward_first_stage_v2_py(layernorm: torch.Tensor, norm_epsilon: float,
                                        q_proj: Q4Ptr, k_proj: Q4Ptr, v_proj: Q4Ptr,
                                        temp_rms_output: torch.Tensor, temp_q_output: torch.Tensor, temp_k_output: torch.Tensor, temp_v_output: torch.Tensor, temp_weight: torch.Tensor,
                                        x: torch.Tensor, BLOCK_KN_SIZE: int, BLOCK_M_SIZE_MAX: int, MATMAL_RECONSTRUCT_THRESHOLD_V2: int, sin: torch.Tensor, cos: torch.Tensor,
                                        q_len: int, batch_size: int, num_heads: int, num_kv_heads: int, head_dim: int, past_len: int, past_lens: torch.Tensor):
    attention_forward_qkv_stage_v2(layernorm, norm_epsilon, q_proj, k_proj, v_proj, temp_rms_output, temp_q_output, temp_k_output, temp_v_output, temp_weight,
                                   x, BLOCK_KN_SIZE, BLOCK_M_SIZE_MAX, MATMAL_RECONSTRUCT_THRESHOLD_V2, q_len, batch_size)
    attention_forward_rope_stage_v2(temp_q_output, temp_k_output, sin, cos, q_len, batch_size, num_heads, num_kv_heads, head_dim, past_len, past_lens)


class CacheV1(object):
    def __init__(self, device: torch.device, x_height: int) -> None:
        self.temp_weight_size: int = 0
        self.temp_x_features: int = 0
        self.temp_x_height: int = x_height
        self._temp_weight: Optional[torch.Tensor] = None
        self._temp_x: Optional[torch.Tensor] = None
        self.device:  torch.device = device
        self.cache_created = False

    def update_temp_weight_size(self, temp_weight_size: int):
        if temp_weight_size > self.temp_weight_size:
            self.cache_created = False
            self.temp_weight_size = temp_weight_size

    def update_temp_x_size(self, temp_x_size: int):
        if temp_x_size > self.temp_x_features:
            self.cache_created = False
            self.temp_x_features = temp_x_size

    def update_x_height(self, x_height: int):
        if x_height > self.temp_x_height:
            self.cache_created = False
            self.temp_x_height = x_height

    @property
    def temp_weight(self) -> torch.Tensor:
        assert self.cache_created
        assert self._temp_weight is not None
        return self._temp_weight

    @property
    def temp_x(self) -> torch.Tensor:
        assert self.cache_created
        assert self._temp_x is not None
        return self._temp_x

    def init_cache(self):
        self._temp_weight = torch.empty(size=(1, self.temp_weight_size), dtype=torch.float16, requires_grad=False, device=self.device)
        self._temp_x = torch.empty((1, self.temp_x_features*self.temp_x_height), dtype=torch.float16, requires_grad=False, device=self.device)


class Q4MatrixV1(object):
    def __init__(self,
                 qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor,
                 height: int, width: int,
                 g_idx: Optional[torch.Tensor], use_act: bool,
                 tuning_params: TuningParams,
                 cache: CacheV1) -> None:
        self.use_act = use_act
        self.cuda_x_map: Optional[torch.Tensor] = None
        if self.use_act:
            assert g_idx is not None
            # memalloc
            self.cuda_x_map = torch.zeros(size=g_idx.size(), dtype=g_idx.dtype, device=qweight.device)
        self.cuda_q4: Q4Ptr = make_q4_py(
            qweight=qweight, qzeros=qzeros, scales=scales,
            g_idx=g_idx if self.use_act else torch.empty((1, 1), device="meta"),
            cuda_x_map=self.cuda_x_map if self.use_act else torch.empty((1, 1), device="meta"),
            device_idx=qweight.device.index,
            use_act=self.use_act,
            v2_mode=False
        )
        self.height: int = height
        self.width: int = width
        self.device: torch.device = qweight.device
        self.tuning_params: TuningParams = tuning_params
        self.cache: CacheV1 = cache
        cache.update_temp_weight_size(qweight.numel()*8)
        cache.update_temp_x_size(height)
        cache.update_temp_x_size(width)

    def reconstruct_weight(self, dtype: torch.dtype):
        assert dtype in [torch.float16, torch.bfloat16]
        if dtype == torch.float16:
            self.reconstructed_weight_float16 = self.get_unremapped_float16_weights_matrix()
        else:
            self.reconstructed_weight_bfloat16 = self.get_unremapped_bfloat16_weights_matrix()

    def release_weight(self, dtype: Optional[torch.dtype] = None):
        if dtype is None:
            self.reconstructed_weight_float16 = None
            self.reconstructed_weight_bfloat16 = None
        if dtype == torch.float16:
            self.reconstructed_weight_float16 = None
        elif dtype == torch.bfloat16:
            self.reconstructed_weight_bfloat16 = None

    def set_tuning_params(self, tuning_params: TuningParams):
        self.tuning_params = tuning_params

    def get_weight_matrix_shape(self):
        return (self.height, self.width)

    def get_x_features(self):
        return max(self.height, self.width)

    def get_unremapped_float16_weights_matrix(self, out_matrix: Optional[torch.Tensor] = None):
        '''cuda_x_map is not applied'''
        if out_matrix is None:
            out_matrix = torch.empty(self.height, self.width, dtype=torch.half, device=self.device, requires_grad=False)
        q4_to_half_matrix_py(self.cuda_q4, out_matrix, RECONSTRUCT_THREADS_X=self.tuning_params.RECONSTRUCT_THREADS_X,
                             RECONSTRUCT_THREADS_Y=self.tuning_params.RECONSTRUCT_THREADS_Y)
        return out_matrix

    def get_unremapped_bfloat16_weights_matrix(self, out_matrix: Optional[torch.Tensor] = None):
        '''cuda_x_map is not applied'''
        if out_matrix is None:
            out_matrix = torch.empty(self.height, self.width, dtype=torch.bfloat16, device=self.device, requires_grad=False)
        q4_to_bfloat16_matrix_py(self.cuda_q4, out_matrix, RECONSTRUCT_THREADS_X=self.tuning_params.RECONSTRUCT_THREADS_X,
                                 RECONSTRUCT_THREADS_Y=self.tuning_params.RECONSTRUCT_THREADS_Y)
        return out_matrix

    def x_column_remap(self, x: torch.Tensor, remapped_x: Optional[torch.Tensor] = None):
        '''apply cuda_x_map on x'''
        if not self.use_act:
            return x
        assert x.device == self.device
        assert self.cuda_x_map is not None
        outshape = x.shape[:-1] + (self.height,)
        x = x.view(-1, x.shape[-1])
        if remapped_x is None:
            remapped_x = torch.empty_like(x, requires_grad=x.requires_grad, device=self.device, dtype=x.dtype)
        column_remap_py(x, remapped_x, self.cuda_x_map, COLUMN_SHUFFLE_BLOCK_X=self.tuning_params.COLUMN_SHUFFLE_BLOCK_X,
                        COLUMN_SHUFFLE_BLOCK_Y=self.tuning_params.COLUMN_SHUFFLE_BLOCK_Y)
        return remapped_x.view(outshape)

    def x_column_restore(self, x: torch.Tensor, restored_x: Optional[torch.Tensor] = None):
        '''revert the apply of cuda_x_map on x'''
        if not self.use_act:
            return x
        assert x.device == self.device
        outshape = x.shape[:-1] + (self.height,)
        x = x.view(-1, x.shape[-1])
        if restored_x is None:
            restored_x = torch.empty_like(x, requires_grad=x.requires_grad, device=self.device, dtype=x.dtype)
        assert self.cuda_x_map is not None
        column_restore_py(x, restored_x, self.cuda_x_map, COLUMN_SHUFFLE_BLOCK_X=self.tuning_params.COLUMN_SHUFFLE_BLOCK_X,
                          COLUMN_SHUFFLE_BLOCK_Y=self.tuning_params.COLUMN_SHUFFLE_BLOCK_Y)
        return restored_x.view(outshape)

    def matmul(self, x: torch.Tensor, output: Optional[torch.Tensor] = None, temp_weight: Optional[torch.Tensor] = None, must_blas: bool = False):
        assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
        temp_x = self.cache.temp_x
        outshape = x.shape[:-1] + (self.width,)
        x = x.view(-1, x.shape[-1])
        x_height = x.shape[0]
        if output is None:
            output = torch.empty((x_height, self.width), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        if x.dtype == torch.float16 and self.reconstructed_weight_float16 is not None:
            q4_matmul_blas_py(x, self.cuda_q4, output, self.reconstructed_weight_float16, temp_x,
                              False, self.tuning_params.COLUMN_SHUFFLE_BLOCK_X, self.tuning_params.COLUMN_SHUFFLE_BLOCK_Y,
                              self.tuning_params.RECONSTRUCT_THREADS_X, self.tuning_params.RECONSTRUCT_THREADS_Y, False, True, False)
            return output.view(outshape)
        elif x.dtype == torch.bfloat16 and self.reconstructed_weight_bfloat16 is not None:
            q4_matmul_blas_py(x, self.cuda_q4, output, self.reconstructed_weight_bfloat16, temp_x,
                              False, self.tuning_params.COLUMN_SHUFFLE_BLOCK_X, self.tuning_params.COLUMN_SHUFFLE_BLOCK_Y,
                              self.tuning_params.RECONSTRUCT_THREADS_X, self.tuning_params.RECONSTRUCT_THREADS_Y, False, True, True)
            return output.view(outshape)
        use_blas = False
        if must_blas or x.dtype == torch.bfloat16 or x_height > self.tuning_params.MATMAL_RECONSTRUCT_THRESHOLD:
            use_blas = True
        if use_blas:
            if temp_weight is None:
                temp_weight = self.cache.temp_weight
            q4_matmul_blas_py(x, self.cuda_q4, output, temp_weight, temp_x,
                              False, self.tuning_params.COLUMN_SHUFFLE_BLOCK_X, self.tuning_params.COLUMN_SHUFFLE_BLOCK_Y,
                              self.tuning_params.RECONSTRUCT_THREADS_X, self.tuning_params.RECONSTRUCT_THREADS_Y, False, False, x.dtype == torch.bfloat16)
        else:
            block_size_z = self.tuning_params.MATMUL_BLOCK_SIZE_Z
            if block_size_z is None:
                block_size_z = 256
                # 7B (4096,11008) # 13B (5120,13824) #70B (6656,17920)
                if self.width in (4096, 5120):
                    block_size_z = 384
                elif self.width == 17920:
                    block_size_z = 128
            q4_matmul_cuda_py(x, self.cuda_q4, output, temp_x,
                              False, self.tuning_params.NO_HALF2, self.tuning_params.NO_FUSE_REMAP,
                              self.tuning_params.MATMUL_THREADS_X, self.tuning_params.MATMUL_THREADS_Y, block_size_z, self.tuning_params.MATMUL_GROUP_STEP,
                              self.tuning_params.COLUMN_SHUFFLE_BLOCK_X, self.tuning_params.COLUMN_SHUFFLE_BLOCK_Y, skip_remap=False)
        return output.view(outshape)

    def transpose_matmul(self, x: torch.Tensor, output: Optional[torch.Tensor] = None, temp_weight: Optional[torch.Tensor] = None, skip_reconstruct: bool = False):
        assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
        temp_x = self.cache.temp_x
        outshape = x.shape[:-1] + (self.height,)
        x = x.view(-1, x.shape[-1])
        x_height = x.shape[0]
        if output is None:
            output = torch.empty((x_height, self.height), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        if x.dtype == torch.float16 and self.reconstructed_weight_float16 is not None:
            q4_transpose_matmul_blas_py(x, self.cuda_q4, output, self.reconstructed_weight_float16, temp_x,
                                        False, self.tuning_params.COLUMN_SHUFFLE_BLOCK_X, self.tuning_params.COLUMN_SHUFFLE_BLOCK_Y,
                                        self.tuning_params.RECONSTRUCT_THREADS_X, self.tuning_params.RECONSTRUCT_THREADS_Y, False, True, False)
            return output.view(outshape)
        elif x.dtype == torch.bfloat16 and self.reconstructed_weight_bfloat16 is not None:
            q4_transpose_matmul_blas_py(x, self.cuda_q4, output, self.reconstructed_weight_bfloat16, temp_x,
                                        False, self.tuning_params.COLUMN_SHUFFLE_BLOCK_X, self.tuning_params.COLUMN_SHUFFLE_BLOCK_Y,
                                        self.tuning_params.RECONSTRUCT_THREADS_X, self.tuning_params.RECONSTRUCT_THREADS_Y, False, True, True)
            return output.view(outshape)
        if temp_weight is None:
            assert not skip_reconstruct
            temp_weight = self.cache.temp_weight
        q4_transpose_matmul_blas_py(x, self.cuda_q4, output, temp_weight, temp_x,
                                    False, self.tuning_params.COLUMN_SHUFFLE_BLOCK_X, self.tuning_params.COLUMN_SHUFFLE_BLOCK_Y,
                                    self.tuning_params.RECONSTRUCT_THREADS_X, self.tuning_params.RECONSTRUCT_THREADS_Y, False, skip_reconstruct, x.dtype == torch.bfloat16)
        return output.view(outshape)


class CacheV2(object):
    def __init__(self, device: torch.device) -> None:
        self.temp_weight_size: int = 0
        self._temp_weight: Optional[torch.Tensor] = None
        self.device:  torch.device = device
        self.cache_created = False

    def update_temp_weight_size(self, temp_weight_size: int):
        if temp_weight_size > self.temp_weight_size:
            self.cache_created = False
            self.temp_weight_size = temp_weight_size

    @property
    def temp_weight(self) -> torch.Tensor:
        assert self.cache_created
        assert self._temp_weight is not None
        return self._temp_weight

    def init_cache(self):
        self._temp_weight = torch.empty(size=(1, self.temp_weight_size), dtype=torch.float16, requires_grad=False, device=self.device)
        self.cache_created = True


class Q4MatrixV2(object):
    def __init__(self,
                 qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor,
                 height: int, width: int,
                 g_idx: Optional[torch.Tensor], use_act: bool,
                 tuning_params: TuningParams,
                 cache: CacheV2) -> None:
        self.use_act = use_act
        self.cuda_x_map: Optional[torch.Tensor] = None
        if self.use_act:
            assert g_idx is not None
            # memalloc
            self.cuda_x_map = torch.zeros(size=g_idx.size(), dtype=g_idx.dtype, device=qweight.device)
        self.cuda_q4: Q4Ptr = make_q4_py(
            qweight=qweight, qzeros=qzeros, scales=scales,
            g_idx=g_idx if self.use_act else torch.empty((1, 1), device="meta"),
            cuda_x_map=self.cuda_x_map if self.use_act else torch.empty((1, 1), device="meta"),
            device_idx=qweight.device.index,
            use_act=self.use_act,
            v2_mode=True
        )
        self.height = height
        self.width = width
        self.device = qweight.device
        self.tuning_params = tuning_params
        self.cache = cache
        cache.update_temp_weight_size(qweight.numel()*8)
        self.reconstructed_weight_float16: Optional[torch.Tensor] = None
        self.reconstructed_weight_bfloat16: Optional[torch.Tensor] = None

    def reconstruct_weight(self, dtype: torch.dtype):
        assert dtype in [torch.float16, torch.bfloat16]
        if dtype == torch.float16:
            self.reconstructed_weight_float16 = self.get_float16_weights_matrix_v2()
        else:
            self.reconstructed_weight_bfloat16 = self.get_bfloat16_weights_matrix_v2()

    def release_weight(self, dtype: Optional[torch.dtype] = None):
        if dtype is None:
            self.reconstructed_weight_float16 = None
            self.reconstructed_weight_bfloat16 = None
        elif dtype == torch.float16:
            self.reconstructed_weight_float16 = None
        elif dtype == torch.bfloat16:
            self.reconstructed_weight_bfloat16 = None

    def set_tuning_params(self, tuning_params: TuningParams):
        self.tuning_params = tuning_params

    def get_weight_matrix_shape(self):
        return (self.height, self.width)

    def get_x_features(self):
        return max(self.height, self.width)

    def get_float16_weights_matrix_v2(self, out_matrix: Optional[torch.Tensor] = None):
        '''cuda_x_map applied'''
        if out_matrix is None:
            out_matrix = torch.zeros(self.height, self.width, dtype=torch.half, device=self.device, requires_grad=False)
        q4_to_half_matrix_v2_py(self.cuda_q4, out_matrix, BLOCK_KN_SIZE=self.tuning_params.BLOCK_KN_SIZE)
        return out_matrix

    def get_bfloat16_weights_matrix_v2(self, out_matrix: Optional[torch.Tensor] = None):
        '''cuda_x_map applied'''
        if out_matrix is None:
            out_matrix = torch.zeros(self.height, self.width, dtype=torch.bfloat16, device=self.device, requires_grad=False)
        q4_to_bfloat16_matrix_v2_py(self.cuda_q4, out_matrix, BLOCK_KN_SIZE=self.tuning_params.BLOCK_KN_SIZE)
        return out_matrix

        # get_device_buffers(self.device).\
        #     update_max_dq_buffer_size(qweight.numel() * 8).\
        #     update_max_inner_outer_dim(height,width).\
        #     update_max_input_size(max_input_size)

    # def matmul_cuda(self,x:torch.Tensor,output:Optional[torch.Tensor],temp_x:Optional[torch.Tensor]):
    #     assert x.dtype==torch.float16
    #     if self.use_act:
    #         if temp_x is None:
    #             temp_x=torch.empty_like(x,requires_grad=False)
    #     outshape = x.shape[:-1] + (self.width,)
    #     x = x.view(-1, x.shape[-1])
    #     if output is None:
    #         output = torch.empty((x.shape[0], self.width), dtype=torch.float16, device=x.device)
    #     BLOCK_SIZE_Z=self.tunning_params.MATMUL_BLOCK_SIZE_Z
    #     if BLOCK_SIZE_Z is None:
    #         BLOCK_SIZE_Z=256
    #         # 7B (4096,11008) # 13B (5120,13824) #70B (6656,17920)
    #         if self.width in (4096,5120):
    #             BLOCK_SIZE_Z=384
    #         elif self.width == 17920:
    #             BLOCK_SIZE_Z=128
    #     q4_matmul_cuda_py(x, self.cuda_q4, output,temp_x,
    #                       False,self.tunning_params.NO_HALF2,self.tunning_params.NO_FUSE_REMAP,
    #                       self.tunning_params.MATMUL_THREADS_X,self.tunning_params.MATMUL_THREADS_Y,BLOCK_SIZE_Z,
    #                       self.tunning_params.COLUMN_SHUFFLE_BLOCK_X,self.tunning_params.COLUMN_SHUFFLE_BLOCK_Y,skip_remap=False)
    #     return output.view(outshape)

    # def matmul_blas(self,x:torch.Tensor,output:Optional[torch.Tensor],temp_weight:Optional[torch.Tensor],temp_x:Optional[torch.Tensor]):
    #     assert x.dtype==torch.float16 or x.dtype==torch.bfloat16
    #     if self.use_act:
    #         if temp_x is None:
    #             temp_x=torch.empty_like(x,requires_grad=False)
    #     outshape = x.shape[:-1] + (self.width,)
    #     x = x.view(-1, x.shape[-1])
    #     if output is None:
    #         output = torch.empty((x.shape[0], self.width), dtype=torch.float16, device=x.device)
    #     q4_matmul_blas_py(x,self.cuda_q4,output,temp_weight,temp_x,
    #                       False,self.tunning_params.COLUMN_SHUFFLE_BLOCK_X,self.tunning_params.COLUMN_SHUFFLE_BLOCK_Y,
    #                       self.tunning_params.RECONSTRUCT_THREADS_X,self.tunning_params.RECONSTRUCT_THREADS_Y,False,False)
    #     return output.view(outshape)

    def matmul(self, x: torch.Tensor, output: Optional[torch.Tensor], temp_weight: Optional[torch.Tensor], must_blas: bool = False):
        assert x.dtype == torch.float16 or x.dtype == torch.bfloat16, f"{x.dtype}"
        outshape = x.shape[:-1] + (self.width,)
        x = x.view(-1, x.shape[-1])
        x_height = x.shape[0]
        if output is None:
            output = torch.empty((x_height, self.width), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        if x.dtype == torch.float16 and self.reconstructed_weight_float16 is not None:
            q4_matmul_blas_v2_py(x, self.cuda_q4, output, self.reconstructed_weight_float16, False, self.tuning_params.BLOCK_KN_SIZE, True, False)
            return output.view(outshape)
        elif x.dtype == torch.bfloat16 and self.reconstructed_weight_bfloat16 is not None:
            q4_matmul_blas_v2_py(x, self.cuda_q4, output, self.reconstructed_weight_bfloat16, False, self.tuning_params.BLOCK_KN_SIZE, True, True)
            return output.view(outshape)
        use_blas = False
        if must_blas or x.dtype == torch.bfloat16 or x_height > self.tuning_params.MATMAL_RECONSTRUCT_THRESHOLD_V2:
            use_blas = True
        if use_blas:
            if temp_weight is None:
                temp_weight = self.cache.temp_weight
            q4_matmul_blas_v2_py(x, self.cuda_q4, output, temp_weight, False, self.tuning_params.BLOCK_KN_SIZE, False, x.dtype == torch.bfloat16)
        else:
            q4_matmul_cuda_v2_py(x, self.cuda_q4, output, self.tuning_params.BLOCK_KN_SIZE, self.tuning_params.BLOCK_M_SIZE_MAX, False)
            # BLOCK_SIZE_Z=self.tuning_params.MATMUL_BLOCK_SIZE_Z
            # if BLOCK_SIZE_Z is None:
            #     BLOCK_SIZE_Z=256
            #     # 7B (4096,11008) # 13B (5120,13824) #70B (6656,17920)
            #     if self.width in (4096,5120):
            #         BLOCK_SIZE_Z=384
            #     elif self.width == 17920:
            #         BLOCK_SIZE_Z=128
            # q4_matmul_cuda_py(x, self.cuda_q4, output,temp_x,
            #                     False,self.tuning_params.NO_HALF2,self.tuning_params.NO_FUSE_REMAP,
            #                     self.tuning_params.MATMUL_THREADS_X,self.tuning_params.MATMUL_THREADS_Y,BLOCK_SIZE_Z,self.tuning_params.MATMUL_GROUP_STEP,
            #                     self.tuning_params.COLUMN_SHUFFLE_BLOCK_X,self.tuning_params.COLUMN_SHUFFLE_BLOCK_Y,skip_remap=False)
        return output.view(outshape)

    def transpose_matmul(self, x: torch.Tensor, output: Optional[torch.Tensor], temp_weight: Optional[torch.Tensor], skip_reconstruct: bool = False):
        assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
        outshape = x.shape[:-1] + (self.height,)
        x = x.reshape(-1, x.shape[-1])
        x_height = x.shape[0]
        if output is None:
            output = torch.empty((x_height, self.height), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        if x.dtype == torch.float16 and self.reconstructed_weight_float16 is not None:
            q4_transpose_matmul_blas_v2_py(x, self.cuda_q4, output, self.reconstructed_weight_float16, False, self.tuning_params.BLOCK_KN_SIZE, True, False)
            return output.view(outshape)
        elif x.dtype == torch.bfloat16 and self.reconstructed_weight_bfloat16 is not None:
            q4_transpose_matmul_blas_v2_py(x, self.cuda_q4, output, self.reconstructed_weight_bfloat16, False, self.tuning_params.BLOCK_KN_SIZE, True, True)
            return output.view(outshape)
        if temp_weight is None:
            temp_weight = self.cache.temp_weight
        q4_transpose_matmul_blas_v2_py(x, self.cuda_q4, output, temp_weight, False, self.tuning_params.BLOCK_KN_SIZE, skip_reconstruct, x.dtype == torch.bfloat16)
        return output.view(outshape)

# from .exllama_cuda_kernels import column_remap
# from .exllama_cuda_kernels import column_restore

# class DeviceBuffers(object):
#     def __init__(self, device) -> None:
#         self.device = device
#         self.max_dq_buffer_size = 0
#         self.max_inner_outer_dim = 0
#         self.max_input_size = 0
#         self.temp_state = None
#         self.tmep_dq = None

#     def update_max_dq_buffer_size(self, *args):
#         for size in args:
#             self.max_dq_buffer_size = max(size, self.max_dq_buffer_size)
#         return self

#     def update_max_inner_outer_dim(self, *args):
#         for size in args:
#             self.max_inner_outer_dim = max(size, self.max_inner_outer_dim)
#         return self

#     def update_max_input_size(self, *args):
#         for size in args:
#             self.max_input_size = max(size, self.max_input_size)
#         return self

#     def prepare_buffers(self):
#         self.temp_state = torch.zeros(
#             (self.max_input_size, self.max_inner_outer_dim), dtype=torch.float16, device=self.device)
#         self.temp_dq = torch.zeros((1, self.max_dq_buffer_size),
#                                    dtype=torch.float16, device=self.device)
#         prepare_buffers(self.device, self.temp_state, self.temp_dq)

#     # def immediately_update_max_input_size(self, *args):
#     #     update=False
#     #     for size in args:
#     #         if size>self.max_input_size:
#     #             update=True
#     #     if update:
#     #         self.temp_state = torch.zeros((self.max_input_size, self.max_inner_outer_dim), dtype=torch.float16, device=self.device)
#     #         prepare_buffers(self.device, self.temp_state, self.temp_dq)
#     #     return self

# devices_buffers = {}

# def get_device_buffers(device):
#     global devices_buffers
#     if not device.index in devices_buffers.keys():
#         device_buffer = DeviceBuffers(device)
#         devices_buffers[device.index] = device_buffer
#         return device_buffer
#     return devices_buffers[device.index]


# def prepare_device_buffers(device):
#     global devices_buffers
#     if device is not None:
#         devices_buffers[device.index].prepare_buffers()
#     else:
#         for devices_buffer in devices_buffers.values():
#             prepare_device_buffers(devices_buffer.device)


# def set_tuning_params(matmul_reconstruct_threshold = 8,matmul_fused_remap = False,matmul_no_half2 = False):
#     _set_tuning_params(matmul_reconstruct_threshold, matmul_fused_remap, matmul_no_half2)
    # def matmul(self, x):
    #     global devices_buffers
    #     assert x.dtype==torch.float16
    #     """Matrix multiplication, returns x @ q4"""
    #     outshape = x.shape[:-1] + (self.width,)
    #     x = x.view(-1, x.shape[-1])
    #     assert self.max_input_size>=x.size(0),f"need increase max input size to {x.size(0)}"
    #     output = torch.empty(
    #         (x.shape[0], self.width), dtype=torch.float16, device=x.device)
    #     q4_matmul(x, self.cuda_q4, output)
    #     return output.view(outshape)

    # def matmul_t(self, x):
    #     global devices_buffers
    #     assert x.dtype==torch.float16
    #     """Matrix multiplication, returns x @ q4.T"""
    #     outshape = x.shape[:-1] + (self.height,)
    #     x = x.view(-1, x.shape[-1])
    #     assert self.max_input_size>=x.size(0),f"need increase max input size to {x.size(0)}"
    #     output = torch.empty((x.shape[0], self.height), dtype=torch.float16, device=x.device)
    #     q4_t_matmul(x, self.cuda_q4, output)
    #     # output = q4_t_matmul_torch_backward(x, self.cuda_q4)
    #     # print(output.shape)
    #     return output.view(outshape)

# def prepare_device_buffers(idx=None):
#     global devices
#     if idx is None:
#         for k in devices:
#             prepare_device_buffers(k)
#     else:
#         prepare_buffers(
#             devices[idx].device,
#             torch.zeros((1, devices[idx].temp_state_size), dtype=torch.float16, device=devices[idx].device),
#             torch.zeros((1, devices[idx].temp_dq_size), dtype=torch.float16, device=devices[idx].device)
#         )

class ExllamaQ4LinearFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, weight_q4: Union[Q4MatrixV1, Q4MatrixV2], bias: Optional[torch.Tensor], cache_weight: bool) -> torch.Tensor:  # type: ignore
        ctx.weight_q4 = weight_q4
        if cache_weight:
            if input.dtype == torch.float16 and (weight_q4.reconstructed_weight_float16 is not None):
                cache_weight = False
            elif input.dtype == torch.bfloat16 and (weight_q4.reconstructed_weight_bfloat16 is not None):
                cache_weight = False
        ctx.cache_weight = cache_weight
        output = None
        if cache_weight:
            weight_restored = torch.empty(weight_q4.get_weight_matrix_shape(), requires_grad=False, device=weight_q4.device, dtype=input.dtype)
            output = weight_q4.matmul(input, output=None, temp_weight=weight_restored, must_blas=True)
            ctx.save_for_backward(weight_restored)
        else:
            output = weight_q4.matmul(input, output=None, temp_weight=None, must_blas=False)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None, None]:  # type: ignore
        grad_input = None
        if ctx.needs_input_grad[0]:
            cached_weight = None
            if ctx.cache_weight:
                cached_weight, = ctx.saved_tensors
            weight_q4: Union[Q4MatrixV1, Q4MatrixV2] = ctx.weight_q4
            grad_input = weight_q4.transpose_matmul(grad_output, output=None, temp_weight=cached_weight, skip_reconstruct=cached_weight is not None)
            # if ctx.cuda_backward:
            #     holder = grad_output.clone()
            #     grad_input = ctx.weight_q4.matmul_t(holder)
            # else:
            #     weight=None
            #     if ctx.cache_weight:
            #         weight, = ctx.saved_tensors
            #     else:
            #         weight=ctx.weight_q4.get_weights_matrix()
            #         if ctx.bfloat16:
            #             weight=weight.to(torch.bfloat16)
            #     if ctx.bfloat16:
            #         assert grad_output.dtype==torch.bfloat16
            #     else:
            #         assert grad_output.dtype==torch.float16
            #     grad_input = torch.matmul(grad_output, weight.t())
            #     grad_input = ctx.weight_q4.x_column_restore(grad_input)
        return grad_input, None, None, None

# class ExllamaQ4LinearFunctionSafe(Function):
#     @staticmethod
#     def forward(input, weight_q4: Q4MatrixV1, bias):
#         input = weight_q4.x_column_remap(input)
#         if input.dtype == torch.float16:
#             weight = weight_q4.get_float16_weights_matrix()
#         elif input.dtype == torch.bfloat16:
#             weight = weight_q4.get_bfloat16_weights_matrix()
#         output = torch.matmul(input, weight)
#         if bias is not None:
#             output += bias.to(input.dtype).unsqueeze(0).expand_as(output)
#         return output

#     @staticmethod
#     def setup_context(ctx, inputs, output):
#         input, weight_q4, bias = inputs
#         ctx.weight_q4 = weight_q4

#     # This function has only a single output, so it gets only one gradient
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = None
#         if ctx.needs_input_grad[0]:
#             if grad_output.dtype == torch.float16:
#                 weight = ctx.weight_q4.get_float16_weights_matrix()
#             elif grad_output.dtype == torch.bfloat16:
#                 weight = ctx.weight_q4.get_bfloat16_weights_matrix()
#             grad_input = torch.matmul(grad_output, weight.t())
#             grad_input = ctx.weight_q4.x_column_restore(grad_input)
#         return grad_input, None, None


class ExllamaQ4ForwardOnlyLinearFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, weight_q4: Union[Q4MatrixV1, Q4MatrixV2], bias: Optional[torch.Tensor]) -> torch.Tensor:  # type:ignore
        output = weight_q4.matmul(input, None, None, False)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, None, None]:  # type:ignore
        return None, None, None
# class Q4LinearFunctionSafeBFloat16WithCache(Function):
#     @staticmethod
#     def forward(ctx,input, weight_q4, bias):
#         assert input.dtype==torch.bfloat16
#         weight_bfloat16=weight_q4.get_weights_matrix().to(torch.bfloat16)
#         ctx.save_for_backward(weight_bfloat16)
#         ctx.weight_q4 = weight_q4
#         output = torch.matmul(input,weight_bfloat16)
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#         return output

#     # This function has only a single output, so it gets only one gradient
#     @staticmethod
#     def backward(ctx, grad_output):
#         assert grad_output.dtype==torch.bfloat16
#         grad_input = None
#         if ctx.needs_input_grad[0]:
#             weight_bfloat16, = ctx.saved_tensors
#             grad_input = torch.matmul(grad_output, weight_bfloat16.t())
#         return grad_input, None, None

# class Q4LinearFunctionSafe(Function):
#     @staticmethod
#     def forward(input, weight_q4, bias):
#         assert input.dtype==torch.float16
#         weight_float16 = weight_q4.get_weights_matrix()
#         output = torch.matmul(input,weight_float16)
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#         return output

#     @staticmethod
#     def setup_context(ctx, inputs, output):
#         input, weight_q4, bias = inputs
#         ctx.weight_q4 = weight_q4

#     # This function has only a single output, so it gets only one gradient
#     @staticmethod
#     def backward(ctx, grad_output):
#         assert grad_output.dtype==torch.float16
#         grad_input = None
#         if ctx.needs_input_grad[0]:
#             weight_float16 = ctx.weight_q4.get_weights_matrix()
#             grad_input = torch.matmul(grad_output, weight_float16.t())
#         return grad_input, None, None

# class Q4LinearFunctionSafeWithCache(Function):
#     @staticmethod
#     def forward(ctx,input, weight_q4, bias):
#         assert input.dtype==torch.float16
#         weight_float16 = weight_q4.get_weights_matrix()
#         output = torch.matmul(input,weight_float16)
#         ctx.weight_q4 = weight_q4
#         ctx.save_for_backward(weight_float16)
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#         return output

#     # This function has only a single output, so it gets only one gradient
#     @staticmethod
#     def backward(ctx, grad_output):
#         assert grad_output.dtype==torch.float16
#         grad_input = None
#         if ctx.needs_input_grad[0]:
#             weight_float16, = ctx.saved_tensors
#             grad_input = torch.matmul(grad_output, weight_float16.t())
#         return grad_input, None, None


# class Q4LinearFunctionQuick(Function):
#     @staticmethod
#     def forward(input, weight_q4, bias):
#         assert input.dtype==torch.float16
#         output = weight_q4.matmul(input)
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#         return output

#     @staticmethod
#     def setup_context(ctx, inputs, output):
#         input, q4_matrix, bias = inputs
#         ctx.q4_matrix = q4_matrix

#     # This function has only a single output, so it gets only one gradient
#     @staticmethod
#     def backward(ctx, grad_output):
#         assert grad_output.dtype==torch.float16
#         holder = grad_output.clone()
#         grad_input = None
#         if ctx.needs_input_grad[0]:
#             # grad_input=
#             grad_input = ctx.q4_matrix.matmul_t(holder)
#             # weight_tensor=ctx.weight_q4.get_weights_matrix()
#             # grad_input=torch.matmul(grad_output.half(),weight_tensor.t())
#         return grad_input, None, None


# from torch.nn import Function
# # Inherit from Function
# class LinearFunction(Function):
#     # Note that forward, setup_context, and backward are @staticmethods
#     @staticmethod
#     def forward(input, weight_q4, bias):
#         output = weight_q4.matmul(input.half())
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#         return output

#     @staticmethod
#     # inputs is a Tuple of all of the inputs passed to forward.
#     # output is the output of the forward().
#     def setup_context(ctx, inputs, output):
#         input, weight_q4, bias = inputs
#         ctx.weight_q4=weight_q4
#         ctx.save_for_backward(input)

#     # This function has only a single output, so it gets only one gradient
#     @staticmethod
#     def backward(ctx, grad_output):
#         # This is a pattern that is very convenient - at the top of backward
#         # unpack saved_tensors and initialize all gradients w.r.t. inputs to
#         # None. Thanks to the fact that additional trailing Nones are
#         # ignored, the return statement is simple even when the function has
#         # optional inputs.
#         input, = ctx.saved_tensors
#         grad_input = None

#         # These needs_input_grad checks are optional and there only to
#         # improve efficiency. If you want to make your code simpler, you can
#         # skip them. Returning gradients for inputs that don't require it is
#         # not an error.
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output.mm(weight)
#         if ctx.needs_input_grad[1]:
#             grad_weight = grad_output.t().mm(input)
#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = grad_output.sum(0)

#         return grad_input, grad_weight, grad_bias

# from exllama_ext import q4_matmul_lora
# from exllama_ext import half_matmul
# from exllama_ext import half_matmul_cublas
# from exllama_ext import q4_mlp
# from exllama_ext import rms_norm
# from exllama_ext import rope_
# from exllama_ext import rep_penalty
# from exllama_ext import apply_rep_penalty

# # Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension

# none_tensor = torch.empty((1, 1), device = "meta")


# # Construct Q4Matrix, return handle

# def ext_make_q4(qweight, qzeros, scales, g_idx, device):

#     return make_q4(qweight,
#                    qzeros,
#                    scales,
#                    g_idx if g_idx is not None else none_tensor,
#                    device)


# # Matrix multiplication, returns x @ q4

# def ext_q4_matmul(x, q4, q4_width, lora_A = None, lora_B = None):

#     outshape = x.shape[:-1] + (q4_width,)
#     x = x.view(-1, x.shape[-1])
#     output = torch.empty((x.shape[0], q4_width), dtype = torch.float16, device = x.device)

#     if lora_A is None:
#         q4_matmul(x, q4, output)
#     else:
#         lora_temp = torch.empty((x.shape[0], lora_A.shape[1]), dtype = torch.float16, device = x.device)
#         q4_matmul_lora(x, q4, output, lora_A, lora_B, lora_temp)

#     return output.view(outshape)


# # Matrix multiplication, returns x @ w, both half-precision tensors

# def ext_half_matmul(x, w, cublas = False):

#     outshape = x.shape[:-1] + (w.shape[1],)
#     x = x.view(-1, x.shape[-1])

#     if cublas:
#         output = torch.empty((x.shape[0], w.shape[1]), dtype = torch.float16, device = x.device)
#         half_matmul_cublas(x, w, output)
#     else:
#         output = torch.zeros((x.shape[0], w.shape[1]), dtype = torch.float16, device = x.device)
#         half_matmul(x, w, output)

#     return output.view(outshape)  ##


# # RoPE embeddings, in_place

# def ext_rope_(x, sin, cos, past_len, num_heads, head_dim):

#     rope_(x, sin, cos, past_len, num_heads, head_dim)


# # RMS norm: x = x * w / sqrt(row_mean(x * x) + epsilon)

# def ext_rms_norm(x, w, epsilon):

#     outshape = x.shape
#     x = x.view(-1, x.shape[-1])
#     output = torch.empty_like(x)
#     rms_norm(x, w, output, epsilon)

#     return output.view(outshape)

# def ext_rms_norm_(x, w, epsilon):

#     outshape = x.shape
#     x = x.view(-1, x.shape[-1])
#     rms_norm(x, w, x, epsilon)


# # Repetition penalty

# def ext_rep_penalty_mask_cpu(vocab_size, sequence, penalty_max, sustain, decay):

#     rep_mask = torch.empty(vocab_size, dtype = torch.float32)
#     rep_penalty(sequence, rep_mask, penalty_max, sustain, decay)
#     return rep_mask


# def ext_apply_rep_penalty_mask_cpu(sequence, penalty_max, sustain, decay, logits):

#     apply_rep_penalty(sequence, penalty_max, sustain, decay, logits)
