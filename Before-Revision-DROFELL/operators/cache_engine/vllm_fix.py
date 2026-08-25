from typing import Dict, List, Literal, Tuple, Union, cast
import torch

from dealers.common.base_types import DeviceTensorCarrier
from operators import flags
from operators.assemble.decoders.common import LayersCacheProvider
from operators.cache_engine.common import Translator

from ..core.block.real import LayerBlocksHolder
from ..core.block.common import BlockInfo
from .common import BlockManager, CacheEngine
# from cuda_extends import vllm_cache as cache_ops


class VllmFixCacheEngine(CacheEngine, Translator, BlockManager, LayersCacheProvider):
    def __init__(
        self,
        block_info: BlockInfo,
        num_cpu_blocks: int,
        num_device_blocks: int,
        num_layers: int,
        device: torch.device,
        mode: Literal["vllm", "dual_table"] = "dual_table",
        translate: bool = True,
    ) -> None:
        self._block_info = block_info
        self._num_layers = num_layers
        self._num_device_blocks = num_device_blocks
        self._num_cpu_blocks = num_cpu_blocks
        self._device_device = device
        self.mode = mode
        self._cpu_device = torch.zeros((0,), device=torch.device("cpu")).device
        self._events: List[torch.cuda.streams.Event] = [
            cast(torch.cuda.Event, torch.cuda.streams.Event())
            for _ in range(self._num_layers)
        ]
        self.translate = translate
        self.available_blocks: List[int] = [
            i+1000 for i in range(num_device_blocks)
        ] if self.translate else [
            i for i in range(num_device_blocks)
        ]  # 1000 is just for debug, avoiding confusing with idx
        self.occupied_blocks: Dict[int, int] = {b: 0 for b in self.available_blocks}

        with torch.inference_mode():
            # [[[key_block,value_block]*num_blocks]*num_layers]
            self._device_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
            # [[num_blocks]*num_layers]
            self._device_key_blocks_ptr_list: List[List[int]]
            self._device_key_blocks_ptr_cpu_tensor: torch.Tensor
            self._device_key_blocks_ptr_device_tensor: torch.Tensor
            self._layer_device_key_blocks_ptr_device_tensor: List[torch.Tensor]

            self._device_val_blocks_ptr_list: List[List[int]]
            self._device_val_blocks_ptr_cpu_tensor: torch.Tensor
            self._device_val_blocks_ptr_device_tensor: torch.Tensor
            self._layer_device_val_blocks_ptr_device_tensor: List[torch.Tensor]
            self._allocate_device_cache()

            # [[[key_block,value_block]*num_blocks]*num_layers]

            self._cpu_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
            self._cpu_key_blocks_ptr_cpu_tensor: torch.Tensor
            self._cpu_value_blocks_ptr_cpu_tensor: torch.Tensor
            self._allocate_cpu_cache()

            self.layer_blocks_holders = [
                LayerBlocksHolder(
                    block_info=self._block_info,
                    layer_key_ptr_table_device=self._device_key_blocks_ptr_device_tensor[layer_i],
                    layer_val_ptr_table_device=self._device_val_blocks_ptr_device_tensor[layer_i],
                    layer_key_ptr_table_cpu=self._device_key_blocks_ptr_cpu_tensor[layer_i],
                    layer_val_ptr_table_cpu=self._device_val_blocks_ptr_cpu_tensor[layer_i],
                    layer_key_ptr_table_list=self._device_key_blocks_ptr_list[layer_i],
                    layer_val_ptr_table_list=self._device_val_blocks_ptr_list[layer_i],
                    contigious_kv_tensor=self._device_cache[layer_i] if mode == "vllm" else None,
                )
                for layer_i in range(self._num_layers)
            ]

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        self._events[0].record(stream)

    def allocate_block(self):
        i = self.available_blocks[0]
        self.available_blocks = self.available_blocks[1:]
        self.occupied_blocks[i] = 1
        return i

    def allocate_blocks(self, count: int):
        bs: List[int] = self.available_blocks[:count]
        self.available_blocks = self.available_blocks[count:]
        for b in bs:
            self.occupied_blocks[b] = 1
        return bs

    def release_block(self, block_idx: int) -> None:
        assert self.occupied_blocks[block_idx] > 0, f"{block_idx}:{self.occupied_blocks[block_idx]}"
        self.occupied_blocks[block_idx] -= 1
        if self.occupied_blocks[block_idx] == 0:
            self.available_blocks.append(block_idx)

    def release_blocks(self, blocks_idx: List[int]) -> None:
        for b in blocks_idx:
            self.release_block(b)

    def ref_block(self, block_idx: int):
        assert self.occupied_blocks[block_idx] > 0
        self.occupied_blocks[block_idx] += 1

    def translate_block(self, block_idx: int) -> int:
        if self.translate:
            out = block_idx-1000
        else:
            out = block_idx
        assert out >= 0
        return out

    def translate_slot(self, slot_i: int) -> int:
        return self.translate_block(slot_i//flags.BLOCK_SIZE)*flags.BLOCK_SIZE+(slot_i % flags.BLOCK_SIZE)

    def data_ptr(self, blocks: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        return [key_ptr:[[blocks]*num_layers],val_ptr:[[blocks]*num_layers]]
        '''
        kp = self._cpu_key_blocks_ptr_cpu_tensor[:, blocks].clone()
        vp = self._cpu_value_blocks_ptr_cpu_tensor[:, blocks].clone()
        return kp, vp

    @property
    def block_info(self):
        return self._block_info

    def get_cache_provider(self, related_blocks: List[int]) -> Tuple[LayersCacheProvider, Translator]:
        return self, self

    def get_layer_with_event_ready(self, layer_i: int) -> LayerBlocksHolder:
        self._events[layer_i].wait()
        return self.layer_blocks_holders[layer_i]

    @staticmethod
    def _allocate_blocks(
        num_layers: int, num_blocks: int,
        key_block_shape: Tuple[int, int, int, int], value_block_shape: Tuple[int, int, int],
        dtype: torch.dtype, device: torch.device) \
            -> Tuple[
                List[Tuple[torch.Tensor, torch.Tensor]],
                torch.Tensor,  # key ptrs in cpu
                torch.Tensor,  # value ptrs in cpu
                List[List[int]], List[List[int]]
    ]:
        with torch.inference_mode():
            # holder: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [[]for _ in range(num_layers)]
            vllm_style_holder: List[Tuple[torch.Tensor, torch.Tensor]] = []
            vllm_layers_key_block_ptr: List[int] = []
            vllm_layers_val_block_ptr: List[int] = []
            key_blocks_ptr: List[List[int]] = [[] for _ in range(num_layers)]  # [[[int]*num_blocks]*num_layers]
            val_blocks_ptr: List[List[int]] = [[] for _ in range(num_layers)]

            for layer_i in range(num_layers):
                large_key_pre_allocate_block = torch.empty(size=(num_blocks, *key_block_shape), dtype=dtype, device=device)
                large_val_pre_allocate_block = torch.empty(size=(num_blocks, *value_block_shape), dtype=dtype, device=device)
                vllm_style_holder.append((large_key_pre_allocate_block, large_val_pre_allocate_block))

            # for _ in range(num_blocks):
            #     for layer_i in range(num_layers):
            #         key_block = torch.empty(size=(1, *key_block_shape), dtype=dtype, device=device)
            #         value_block = torch.empty(size=(1, *value_block_shape), dtype=dtype, device=device)
            #         holder[layer_i].append((key_block, value_block))
            for layer_i, (layer_key_block, layer_val_block) in enumerate(vllm_style_holder):
                vllm_layers_key_block_ptr.append(layer_key_block.data_ptr())
                vllm_layers_val_block_ptr.append(layer_val_block.data_ptr())
                for block_i in range(num_blocks):
                    key_blocks_ptr[layer_i].append(layer_key_block[block_i].data_ptr())
                    val_blocks_ptr[layer_i].append(layer_val_block[block_i].data_ptr())
            key_blocks_ptr_in_cpu = torch.tensor(key_blocks_ptr, device="cpu", dtype=torch.int64)
            value_blocks_ptr_in_cpu = torch.tensor(val_blocks_ptr, device="cpu", dtype=torch.int64)
            return vllm_style_holder, key_blocks_ptr_in_cpu, value_blocks_ptr_in_cpu, key_blocks_ptr, val_blocks_ptr

    def _allocate_device_cache(self):
        self._device_cache, self._device_key_blocks_ptr_cpu_tensor, self._device_val_blocks_ptr_cpu_tensor, self._device_key_blocks_ptr_list, self._device_val_blocks_ptr_list =\
            self._allocate_blocks(num_layers=self._num_layers, num_blocks=self._num_device_blocks,
                                  key_block_shape=self._block_info.key_block_shape, value_block_shape=self._block_info.value_block_shape,
                                  dtype=self._block_info.dtype, device=self._device_device)
        # SYNC
        self._device_key_blocks_ptr_device_tensor = self._device_key_blocks_ptr_cpu_tensor.to(self._device_device)
        self._layer_device_key_blocks_ptr_device_tensor = [self._device_key_blocks_ptr_device_tensor[i] for i in range(self._num_layers)]
        # SYNC
        self._device_val_blocks_ptr_device_tensor = self._device_val_blocks_ptr_cpu_tensor.to(self._device_device)
        self._layer_device_val_blocks_ptr_device_tensor = [self._device_val_blocks_ptr_device_tensor[i] for i in range(self._num_layers)]

    def _allocate_cpu_cache(self):
        self._cpu_cache, self._cpu_key_blocks_ptr_cpu_tensor, self._cpu_value_blocks_ptr_cpu_tensor, _, _ =\
            self._allocate_blocks(num_layers=self._num_layers, num_blocks=self._num_cpu_blocks,
                                  key_block_shape=self._block_info.key_block_shape, value_block_shape=self._block_info.value_block_shape,
                                  dtype=self._block_info.dtype, device=self._cpu_device)

    def put_blocks(self, target_blocks: List[int], kv_cache: List[List[Union[Tuple[torch.Tensor, torch.Tensor], Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]]]):
        target_blocks = [self.translate_block(b) for b in target_blocks]
        for layer_i, layer_holder in enumerate(self.layer_blocks_holders):
            layer_holder.swap_in_from_mapping({
                i: (k.data_ptr(), v.data_ptr())
                for i, (k, v) in zip(target_blocks, kv_cache[layer_i])
            })

    def set_blocks_readonly(self, blocks: List[int]) -> None:
        pass

    def get_blocks(self, target_blocks: List[int]) -> List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]]:
        target_blocks = [self.translate_block(b) for b in target_blocks]
        layers: List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]] = []
        for layer_holder in self.layer_blocks_holders:
            tensors = layer_holder.swap_out_to_separate_tensors(target_blocks)
            layers.append([
                (DeviceTensorCarrier.from_data(kv[0]), DeviceTensorCarrier.from_data(kv[1]))
                for kv in tensors
            ])
        return layers

        # bytes_per_block = self._block_info.bytes_per_block
        # for layer_i, (kps, vps) in enumerate(kv_cache):
        #     kptr_pair = [
        #         (kp, self._device_key_blocks_ptr_list[layer_i][target_blocks[i]])
        #         for i, kp in enumerate(kps)
        #     ]
        #     swap(
        #         kptr_pair,
        #         src_is_cpu=True,
        #         dst_is_cpu=False,
        #         per_block_bytes=bytes_per_block
        #     )
        #     vptr_pair = [
        #         (kp, self._device_val_blocks_ptr_list[layer_i][target_blocks[i]])
        #         for i, kp in enumerate(vps)
        #     ]
        #     swap(
        #         vptr_pair,
        #         src_is_cpu=True,
        #         dst_is_cpu=False,
        #         per_block_bytes=bytes_per_block
        #     )

    # def get_blocks(self, target_blocks: List[int], kv_cache: List[Tuple[List[int], List[int]]]):
    #     for layer_holder, (kps, vps) in zip(self.layer_blocks_holders, kv_cache):
    #         layer_holder.swap_out(kps, vps, target_blocks)

    # def swap_in(self, blocks_to_swap_in: Dict[int, int]) -> List[torch.cuda.Event]:
    #     with torch.cuda.stream(self._cache_stream):
    #         for i in range(self._num_layers):
    #             cache_ops.swap_blocks(self._cpu_key_blocks_ptr_cpu_tensor[i], self._device_key_blocks_ptr_cpu_tensor[i],
    #                                   blocks_to_swap_in,
    #                                   cache_ops.CPU_TO_device, self._block_info.bytes_per_block)
    #             cache_ops.swap_blocks(self._cpu_value_blocks_ptr_cpu_tensor[i], self._device_val_blocks_ptr_cpu_tensor[i],
    #                                   blocks_to_swap_in,
    #                                   cache_ops.CPU_TO_device, self._block_info.bytes_per_block)
    #             event = self._events[i]
    #             event.record(stream=self._cache_stream)
    #     return self._events

    # def swap_out(self, blocks_to_swap_out: Dict[int, int]) -> List[torch.cuda.Event]:
    #     with torch.cuda.stream(self._cache_stream):
    #         for i in range(self._num_layers):
    #             cache_ops.swap_blocks(self._device_key_blocks_ptr_cpu_tensor[i], self._cpu_key_blocks_ptr_cpu_tensor[i],
    #                                   blocks_to_swap_out,
    #                                   cache_ops.device_TO_CPU, self._block_info.bytes_per_block)
    #             cache_ops.swap_blocks(self._device_val_blocks_ptr_cpu_tensor[i], self._cpu_value_blocks_ptr_cpu_tensor[i],
    #                                   blocks_to_swap_out,
    #                                   cache_ops.device_TO_CPU, self._block_info.bytes_per_block)
    #             event = self._events[i]
    #             event.record(stream=self._cache_stream)
    #     return self._events

    # def copy(self, blocks_to_copy: dict[int, List[int]]):
    #     # wait until copy done
    #     block_mapping_flatten: List[int] = []
    #     for src, dsts in blocks_to_copy.items():
    #         for dst in dsts:
    #             block_mapping_flatten += [src, dst]
    #     block_mapping_flatten_cuda = torch.tensor(
    #         block_mapping_flatten, dtype=torch.long, device=self._device_device)
    #     cache_ops.copy_blocks(
    #         self._device_key_blocks_ptr_device_tensor, self._device_val_blocks_ptr_device_tensor,
    #         block_mapping_flatten_cuda,
    #         self._block_info.elements_per_block, self._block_info.dtype
    #     )

    # def put_blocks(self, target_blocks: List[int], kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]):
    #     # check
    #     assert len(kv_cache) == self._num_layers, f"{len(kv_cache)}!={self._num_layers}"
    #     for layer_kv_cache in kv_cache:
    #         assert layer_kv_cache[0].size(0) % self._block_info.block_size == 0
    #         assert layer_kv_cache[1].size(0) % self._block_info.block_size == 0
    #         assert layer_kv_cache[0].device == self._cpu_device
    #         assert layer_kv_cache[1].device == self._cpu_device
    #         assert layer_kv_cache[0].dtype == torch.float16
    #         assert layer_kv_cache[1].dtype == torch.float16
    #         assert layer_kv_cache[0].is_contiguous()
    #         assert layer_kv_cache[1].is_contiguous()
    #         assert len(target_blocks) == layer_kv_cache[0].size(0) // self._block_info.block_size
    #         assert len(target_blocks) == layer_kv_cache[1].size(0) // self._block_info.block_size

    #     # generate ptr
    #     key_ptrs: List[List[int]] = [[] for _ in range(self._num_layers)]
    #     value_ptrs: List[List[int]] = [[] for _ in range(self._num_layers)]
    #     do_not_gc: List[Tuple[torch.Tensor, torch.Tensor]] = []
    #     for (layer_i, layer_kv_cache) in enumerate(kv_cache):
    #         key_cache = layer_kv_cache[0]  # [num_tokens, num_heads, head_size]
    #         value_cache = layer_kv_cache[1]  # [num_tokens, num_heads, head_size]
    #         num_tokens, num_heads, head_size = key_cache.size()
    #         key_cache = key_cache.view(
    #             num_tokens//self._block_info.block_size,
    #             self._block_info.block_size,
    #             num_heads,
    #             head_size//self._block_info.x,
    #             self._block_info.x
    #         )
    #         value_cache = value_cache.view(
    #             num_tokens//self._block_info.block_size,
    #             self._block_info.block_size,
    #             num_heads,
    #             head_size
    #         )
    #         key_cache = key_cache.permute(0, 2, 3, 1, 4).contiguous()
    #         value_cache = value_cache.permute(0, 2, 3, 1).contiguous()
    #         do_not_gc.append((key_cache, value_cache))

    #         for block_i in range(len(target_blocks)):
    #             # self._device_cache[layer_i][target_blocks[block_i]][0][:] = key_cache[block_i]
    #             # self._device_cache[layer_i][target_blocks[block_i]][1][:] = value_cache[block_i]
    #             key_ptrs[layer_i].append(key_cache[block_i].data_ptr())
    #             value_ptrs[layer_i].append(value_cache[block_i].data_ptr())
    #     src_key_ptrs = torch.tensor(key_ptrs, device=self._cpu_device, dtype=torch.int64)
    #     src_value_ptrs = torch.tensor(value_ptrs, device=self._cpu_device, dtype=torch.int64)
    #     # generate mapping
    #     src_to_dst = {i: t for i, t in enumerate(target_blocks)}
    #     # do swap in
    #     # with torch.cuda.stream(self._cache_stream):
    #     for layer_i in range(self._num_layers):
    #         cache_ops.swap_blocks(
    #             src_key_ptrs[layer_i],
    #             self._device_key_blocks_ptr_cpu_tensor[layer_i],
    #             src_to_dst,
    #             cache_ops.CPU_TO_device,
    #             self._block_info.bytes_per_block,
    #         )
    #         cache_ops.swap_blocks(
    #             src_value_ptrs[layer_i],
    #             self._device_val_blocks_ptr_cpu_tensor[layer_i],
    #             src_to_dst, cache_ops.CPU_TO_device,
    #             self._block_info.bytes_per_block
    #         )
    #         # event = self._events[layer_i]
    #         # event.record(stream=self._cache_stream)

    #     # def wait_complete():
    #     #     for event in self._events:
    #     #         event.wait()
    #     # return wait_complete

    # def get_blocks(self, target_blocks: List[int]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    #     num_blocks = len(target_blocks)
    #     kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = [
    #         (
    #             torch.empty(
    #                 num_blocks,
    #                 self._block_info.block_size*self._block_info.num_heads*self._block_info.head_size,
    #                 dtype=torch.float16
    #             ), torch.empty(
    #                 num_blocks,
    #                 self._block_info.block_size*self._block_info.num_heads*self._block_info.head_size,
    #                 dtype=torch.float16
    #             )
    #         )
    #         for _ in range(self._num_layers)
    #     ]
    #     key_ptrs: List[List[int]] = [[] for _ in range(self._num_layers)]
    #     value_ptrs: List[List[int]] = [[] for _ in range(self._num_layers)]
    #     for (layer_i, layer_kv_cache) in enumerate(kv_cache):
    #         key_cache = layer_kv_cache[0]  # [num_blocks, num_heads, head_size/x, block_size, x]
    #         value_cache = layer_kv_cache[1]  # [num_blocks, num_heads, head_size, block_size]
    #         for block_i in range(len(target_blocks)):
    #             key_ptrs[layer_i].append(key_cache[block_i].data_ptr())
    #             value_ptrs[layer_i].append(value_cache[block_i].data_ptr())
    #     dst_key_ptrs = torch.tensor(key_ptrs, device=self._cpu_device, dtype=torch.int64)
    #     dst_value_ptrs = torch.tensor(value_ptrs, device=self._cpu_device, dtype=torch.int64)
    #     src_to_dst = {t: i for i, t in enumerate(target_blocks)}
    #     for layer_i in range(self._num_layers):
    #         cache_ops.swap_blocks(self._device_key_blocks_ptr_cpu_tensor[layer_i], dst_key_ptrs[layer_i], src_to_dst, cache_ops.device_TO_CPU, self._block_info.bytes_per_block)
    #         cache_ops.swap_blocks(self._device_val_blocks_ptr_cpu_tensor[layer_i], dst_value_ptrs[layer_i],  src_to_dst, cache_ops.device_TO_CPU, self._block_info.bytes_per_block)
    #         layer_key_cache, layer_value_cache = kv_cache[layer_i][0], kv_cache[layer_i][1]
    #         layer_key_cache = layer_key_cache \
    #             .view(
    #                 num_blocks,
    #                 self._block_info.num_heads,
    #                 self._block_info.head_size//self._block_info.x,
    #                 self._block_info.block_size,
    #                 self._block_info.x)\
    #             .permute(0, 3, 1, 2, 4)\
    #             .contiguous()\
    #             .view(
    #                 num_blocks*self._block_info.block_size,
    #                 self._block_info.num_heads,
    #                 self._block_info.head_size
    #             )
    #         layer_value_cache = layer_value_cache \
    #             .view(
    #                 num_blocks,
    #                 self._block_info.num_heads,
    #                 self._block_info.head_size,
    #                 self._block_info.block_size
    #             )\
    #             .permute(0, 3, 1, 2)\
    #             .contiguous()\
    #             .view(
    #                 num_blocks*self._block_info.block_size,
    #                 self._block_info.num_heads,
    #                 self._block_info.head_size
    #             )
    #         kv_cache[layer_i] = (layer_key_cache, layer_value_cache)
    #     return kv_cache
    #     # kv_cache: List[Tensor4D2LHC] = [torch.empty((2, *kv_cache_shape), device=self._cpu_device, dtype=torch.float16, requires_grad=False) for _ in range(self._num_layers)]
    #     # layer_device_kv_cache = torch.empty((2, *kv_cache_shape), device=self._cpu_device, dtype=torch.float16, requires_grad=False)
    #     # for layer_i,  layer_kv_cache in enumerate(kv_cache):

    #     #     layer_ptuning_key = torch.empty(kv_cache_shape, device=self._device_device, dtype=self.info.dtype, requires_grad=False)
    #     #     layer_ptuning_value = torch.empty(kv_cache_shape, device=self._device_device, dtype=self.info.dtype, requires_grad=False)
    #     #     cache_ops.restore_from_cache(
    #     #         layer_ptuning_key,
    #     #         layer_ptuning_value,
    #     #         self.layer_key_blocks_ptr,
    #     #         self.layer_value_blocks_ptr,
    #     #         slot_mapping_tensor,
    #     #         self.info.block_size,
    #     #         self.info.x)
    # #     return layer_ptuning_key, layer_ptuning_value
