from typing import Dict, List, Tuple, Optional, Union
import torch
from cuda_extends import vllm_cache as cache_ops
from operators import flags
from .common import BlockInfo

TensorOrDataPtr = Union[torch.Tensor, int]


def swap_blocks_list(src_to_dst: List[Tuple[int, int]], direction: int, per_block_bytes: int, alt_cuda_stream: Optional[int] = None):
    if alt_cuda_stream is None:
        cache_ops.swap_blocks_list(src_to_dst, direction, per_block_bytes, 0, False)
    else:
        cache_ops.swap_blocks_list(src_to_dst, direction, per_block_bytes, alt_cuda_stream, True)


class LayerBlocksHolder:
    def __init__(self,
                 block_info: BlockInfo,
                 layer_key_ptr_table_device: torch.Tensor,
                 layer_val_ptr_table_device: torch.Tensor,
                 layer_key_ptr_table_cpu: torch.Tensor,
                 layer_val_ptr_table_cpu: torch.Tensor,
                 layer_key_ptr_table_list: List[int],
                 layer_val_ptr_table_list: List[int],
                 contigious_kv_tensor: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                 ) -> None:
        self.block_info = block_info
        self.layer_key_ptr_table_device = layer_key_ptr_table_device
        self.layer_val_ptr_table_device = layer_val_ptr_table_device
        self.layer_key_ptr_table_cpu = layer_key_ptr_table_cpu
        self.layer_val_ptr_table_cpu = layer_val_ptr_table_cpu
        self.layer_key_ptr_table_list = layer_key_ptr_table_list
        self.layer_val_ptr_table_list = layer_val_ptr_table_list
        assert not self.layer_key_ptr_table_device.requires_grad
        assert not self.layer_val_ptr_table_device.requires_grad
        self._contigious_kv_tensor = contigious_kv_tensor
        self._contigious_kv_tensor_bak: Optional[Tuple[torch.Tensor, torch.Tensor]]
        self.device = layer_key_ptr_table_device.device

    def as_contigious(self, contigious: bool):
        if contigious:
            if self._contigious_kv_tensor is not None:
                return
            else:
                assert self._contigious_kv_tensor_bak is not None
                self._contigious_kv_tensor = self._contigious_kv_tensor_bak
        else:
            if self._contigious_kv_tensor_bak is not None:
                return
            else:
                assert self._contigious_kv_tensor is not None
                self._contigious_kv_tensor_bak = self._contigious_kv_tensor

    @property
    def is_contigious(self) -> bool:
        return self._contigious_kv_tensor is not None

    @property
    def contigious_kv_tenosr(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # automatic switch implement basing on this prop in page attetion
        # ~/projects/LLM/island/operators/core/block/real.py line 38
        assert self._contigious_kv_tensor is not None
        return self._contigious_kv_tensor

    def kv_tensor_to_slots_with_convert(
        self,
            key_states: torch.Tensor,
            val_states: torch.Tensor,
            slot_mapping: torch.Tensor,
    ) -> None:
        """
        put kv format tensor to cache by slots and format change
        """
        # key_states [num_tokens,num_kv_heads,head_dim]
        # slot mapping [num_tokens]
        assert key_states.shape[0] == slot_mapping.shape[0]
        cache_ops.reshape_and_cache(
            key_states,
            val_states,
            self.layer_key_ptr_table_device,
            self.layer_val_ptr_table_device,
            slot_mapping,
            flags.BLOCK_SIZE,
            self.block_info.x
        )

    def kv_tensor_from_slots_with_convert(
        self,
            key_states: torch.Tensor,
            val_states: torch.Tensor,
            slot_mapping: torch.Tensor,
    ) -> None:
        """
        get kv format tensor from cache by slots and format change
        """
        assert key_states.shape[0] == slot_mapping.shape[0]
        cache_ops.restore_from_cache(
            key_states,
            val_states,
            self.layer_key_ptr_table_device,
            self.layer_val_ptr_table_device,
            slot_mapping,
            flags.BLOCK_SIZE,
            self.block_info.x
        )

    def kv_tensor_from_slots_with_convert_with_create(self, slot_mapping: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        get kv format tensor from cache by slots and format change with create
        """
        seq_len: int = slot_mapping.shape[0]
        num_kv_heads, head_dim = self.block_info.num_key_value_heads, self.block_info.head_dim
        layer_key_cache = torch.empty(seq_len, num_kv_heads, head_dim, dtype=torch.float16, requires_grad=False)
        layer_value_cache = torch.empty(seq_len, num_kv_heads, head_dim, dtype=torch.float16, requires_grad=False)
        self.kv_tensor_from_slots_with_convert(layer_key_cache, layer_value_cache, slot_mapping)
        return layer_key_cache, layer_value_cache

    def _swap(self,
              idx_data_ptr_mapping: Dict[int, Tuple[TensorOrDataPtr, TensorOrDataPtr]],
              cache_to_tensor: bool,
              tensor_is_cpu: bool,
              alt_stream: Optional[int] = None):
        """
        idx_data_ptr_mapping={block_i:(key_tensor_data_ptr,val_tensor_data_ptr)}
        cache_to_tensor:
            True:  block_i->(key_tensor_data_ptr,val_tensor_data_ptr)
            False: (key_tensor_data_ptr,val_tensor_data_ptr)->block_i
        """
        src_is_cpu = False if cache_to_tensor else tensor_is_cpu
        dst_is_cpu = tensor_is_cpu if cache_to_tensor else False

        if src_is_cpu and dst_is_cpu:
            raise ValueError
        elif src_is_cpu and (not dst_is_cpu):
            mode = cache_ops.CPU_TO_GPU
        elif (not src_is_cpu) and (dst_is_cpu):
            mode = cache_ops.GPU_TO_CPU
        else:
            mode = cache_ops.GPU_TO_GPU

        if len(idx_data_ptr_mapping) == 0:
            return
        src_to_dst: List[Tuple[int, int]] = []
        for cache_block_idx, (key_tensor, val_tesnor) in idx_data_ptr_mapping.items():
            key_tensor = key_tensor if isinstance(key_tensor, int) else key_tensor.data_ptr()
            val_tesnor = val_tesnor if isinstance(val_tesnor, int) else val_tesnor.data_ptr()
            ck = self.layer_key_ptr_table_list[cache_block_idx]
            cv = self.layer_val_ptr_table_list[cache_block_idx]

            if cache_to_tensor:
                src_to_dst.append((ck, key_tensor))
                src_to_dst.append((cv, val_tesnor))

            else:
                src_to_dst.append((key_tensor, ck))
                src_to_dst.append((val_tesnor, cv))
        swap_blocks_list(src_to_dst, mode, self.block_info.bytes_per_block, alt_stream)

    def swap_in_from_list(self, key_ptrs: List[TensorOrDataPtr], val_ptrs: List[TensorOrDataPtr], target_blocks: List[int], src_is_cpu: bool = True, alt_stream: Optional[int] = None):
        mapping = {
            t: (k, v) for
            t, k, v in zip(target_blocks, key_ptrs, val_ptrs)
        }
        self._swap(mapping, False, src_is_cpu, alt_stream)

    def swap_in_from_mapping(self, idx_data_ptr_mapping: Dict[int, Tuple[TensorOrDataPtr, TensorOrDataPtr]], src_is_cpu: bool = True, alt_stream: Optional[int] = None):
        self._swap(idx_data_ptr_mapping, False, src_is_cpu, alt_stream)

    def swap_out_to_list(self, key_ptrs: List[TensorOrDataPtr], val_ptrs: List[TensorOrDataPtr], target_blocks: List[int], dst_is_cpu: bool = True, alt_stream: Optional[int] = None):
        mapping = {
            t: (k, v) for
            t, k, v in zip(target_blocks, key_ptrs, val_ptrs)
        }
        self._swap(mapping, True, dst_is_cpu, alt_stream)  # type:ignore

    def swap_out_to_mapping(self, idx_data_ptr_mapping: Dict[int, Tuple[TensorOrDataPtr, TensorOrDataPtr]], dst_is_cpu: bool = True, alt_stream: Optional[int] = None):
        self._swap(idx_data_ptr_mapping, True, dst_is_cpu, alt_stream)

    def swap_out_to_separate_tensors(self, target_blocks: List[int], alt_stream: Optional[int] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        num_blocks = len(target_blocks)
        key_size = self.block_info.key_block_shape
        val_size = self.block_info.value_block_shape
        device = torch.device("cpu")
        kv_blocks = [(
            torch.empty(size=key_size, dtype=self.block_info.dtype, device=device),
            torch.empty(size=val_size, dtype=self.block_info.dtype, device=device)
        ) for _ in range(num_blocks)]
        self.swap_out_to_mapping({
            i: kv
            for i, kv in zip(target_blocks, kv_blocks)
        }, alt_stream=alt_stream)
        return kv_blocks

    def swap_in_from_single_block(self, src: torch.Tensor, target_blocks: List[int], alt_stream: Optional[int] = None):
        '''
        with no data order change, put src to specific blocks
        '''
        kv_hidden_size = self.block_info.head_dim*self.block_info.num_key_value_heads
        expect_block_shape = (2, len(target_blocks)*flags.BLOCK_SIZE, kv_hidden_size)
        assert src.shape == expect_block_shape, f"expect shape: {expect_block_shape}, actual shape: {src.shape}"
        assert src.is_contiguous()
        assert src.dtype == torch.float16
        ptr0 = src.data_ptr()
        per_block_k_or_v_bytes = kv_hidden_size*flags.BLOCK_SIZE*2  # float16 -> 2bytes
        ptr1 = ptr0+len(target_blocks) * per_block_k_or_v_bytes
        self.swap_in_from_mapping({
            b: (ptr0 + i * per_block_k_or_v_bytes, ptr1 + i * per_block_k_or_v_bytes)
            for i, b in enumerate(target_blocks)
        }, src.is_cpu, alt_stream=alt_stream)

    def swap_out_to_single_block(self, dst: torch.Tensor, src_blocks: List[int], alt_stream: Optional[int] = None):
        '''
        with no data order change, read specific blocks to dst
        '''
        kv_hidden_size = self.block_info.head_dim*self.block_info.num_key_value_heads
        expect_block_shape = (2, len(src_blocks)*flags.BLOCK_SIZE, kv_hidden_size)
        assert dst.shape == expect_block_shape
        assert dst.is_contiguous()
        assert dst.dtype == torch.float16
        ptr0 = dst.data_ptr()
        per_block_k_or_v_bytes = kv_hidden_size*flags.BLOCK_SIZE*2  # float16 -> 2bytes
        ptr1 = ptr0+len(src_blocks) * per_block_k_or_v_bytes
        self.swap_out_to_mapping({
            b: (ptr0 + i * per_block_k_or_v_bytes, ptr1 + i * per_block_k_or_v_bytes)
            for i, b in enumerate(src_blocks)
        }, dst.is_cpu, alt_stream=alt_stream)

    def swap_out_to_single_block_with_create(self, src_blocks: List[int], device: torch.device, alt_stream: Optional[int] = None):
        '''
        with no data order change, read specific blocks to dst
        '''
        kv_hidden_size = self.block_info.head_dim*self.block_info.num_key_value_heads
        expect_block_shape = (2, len(src_blocks)*flags.BLOCK_SIZE, kv_hidden_size)
        dst = torch.empty(expect_block_shape, dtype=torch.float16, device=device)
        self.swap_out_to_single_block(dst, src_blocks=src_blocks, alt_stream=alt_stream)
        return dst

    @staticmethod
    def create_contigious_cacher(block_info: BlockInfo, num_blocks: int, device: torch.device, dtype: torch.dtype = torch.float16):
        assert dtype.itemsize == block_info.element_size
        key_pre_allocate_blocks = torch.empty(size=(num_blocks, *block_info.key_block_shape), dtype=dtype, device=device)
        val_pre_allocate_blocks = torch.empty(size=(num_blocks, *block_info.value_block_shape), dtype=dtype, device=device)
        key_ptrs: List[int] = [key_pre_allocate_blocks[block_i].data_ptr() for block_i in range(num_blocks)]
        val_ptrs: List[int] = [val_pre_allocate_blocks[block_i].data_ptr() for block_i in range(num_blocks)]
        key_ptr_table_cpu = torch.tensor(key_ptrs, device="cpu", dtype=torch.int64)
        val_ptr_table_cpu = torch.tensor(val_ptrs, device="cpu", dtype=torch.int64)
        key_ptr_table_device = key_ptr_table_cpu.to(device=device)
        val_ptr_table_device = val_ptr_table_cpu.to(device=device)
        return LayerBlocksHolder(
            block_info=block_info,
            layer_key_ptr_table_device=key_ptr_table_device, layer_val_ptr_table_device=val_ptr_table_device,
            layer_key_ptr_table_cpu=key_ptr_table_cpu, layer_val_ptr_table_cpu=val_ptr_table_cpu,
            layer_key_ptr_table_list=key_ptrs,
            layer_val_ptr_table_list=val_ptrs,
            contigious_kv_tensor=(key_pre_allocate_blocks, val_pre_allocate_blocks)
        )


# def swap(
#     src_to_dst_ptr: List[Tuple[int, int]],
#     src_is_cpu: bool,
#     dst_is_cpu: bool,
#     per_block_bytes: int
# ):
#     if src_is_cpu and dst_is_cpu:
#         raise ValueError
#     elif src_is_cpu and (not dst_is_cpu):
#         mode = cache_ops.CPU_TO_GPU
#     elif (not src_is_cpu) and (dst_is_cpu):
#         mode = cache_ops.GPU_TO_CPU
#     else:
#         mode = cache_ops.GPU_TO_GPU
#     cache_ops.swap_blocks_list(src_to_dst_ptr, mode, per_block_bytes)
