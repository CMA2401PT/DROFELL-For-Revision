from typing import Dict, List, Literal, Optional, Tuple, Union, cast
import torch
from dataclasses import dataclass

from dealers.common.base_types import DeviceTensorCarrier
from operators import flags
from operators.assemble.decoders.common import LayersCacheProvider
from operators.core.block.common import BlockInfo
from ..core.block.real import LayerBlocksHolder
from .common import CacheEngine, Translator

BlockStatus = Literal["empty", "device", "cpu", "ro_both"]
CPUTensor = List[Union[Tuple[torch.Tensor, torch.Tensor], Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]]  # layers*(k,v)


@dataclass
class LocalBlock:
    state: BlockStatus
    ro: bool
    cpu_tensor: Optional[CPUTensor] = None  # layers*(k,v)
    device_table_i: int = -1
    # Must mirror operators.cache_engine.memory.LocalBlock.generation.
    generation: int = 0


class _Translator(Translator):
    def __init__(self, mapping: Dict[int, int]) -> None:
        self.mapping = mapping

    def translate_block(self, block_idx: int):
        return self.mapping[block_idx]

    def translate_slot(self, slot_i: int) -> int:
        return self.translate_block(slot_i//flags.BLOCK_SIZE)*flags.BLOCK_SIZE+(slot_i % flags.BLOCK_SIZE)


class ScalableCacheEngine(CacheEngine, LayersCacheProvider):
    def __init__(self,
                 block_info: BlockInfo,
                 num_layers: int,
                 device: torch.device,
                 max_blocks_count: int,
                 buffer_blocks_count: int = 16,
                 preswap: bool = True,
                 ) -> None:
        self._block_info = block_info
        self._bytes_per_block = self._block_info.bytes_per_block*2*num_layers
        self._num_layers = num_layers
        self._device = device
        self._preswap = preswap
        self._swapout_stream = cast(torch.cuda.Stream, torch.cuda.Stream(self._device, priority=0))
        self._swapin_stream = cast(torch.cuda.Stream, torch.cuda.Stream(self._device, priority=0))
        self._cpu_device = torch.zeros((0,), device=torch.device("cpu")).device
        self._max_blocks_count = max_blocks_count
        # self.current_ptr_count = 0
        self._buffer_blocks_count = buffer_blocks_count
        self._next_block_idx = 1000  # 1000 is just for debug, avoiding confusing with idx
        self._local_blocks: Dict[int, LocalBlock] = {}
        self._cache_blocks: List[int] = []
        self._device_tensors: Dict[int, Tuple[int, torch.Tensor]] = {}
        # self._full_ptr_mapping: Dict[int, int] = {}  # actual ptr -> actual block id

        self._full_ptr_tensor_cpu: torch.Tensor = torch.zeros(
            (2, num_layers, self._max_blocks_count),
            dtype=torch.int64, device=self._cpu_device, requires_grad=False
        )
        self._full_ptr_tensor_device: torch.Tensor = torch.zeros(
            (2, num_layers, self._max_blocks_count),
            dtype=torch.int64, device=device, requires_grad=False
        )
        self._device_key_blocks_ptr_list: List[List[int]] = [
            [-1 for _ in range(max_blocks_count)]
            for _ in range(num_layers)
        ]
        self._device_val_blocks_ptr_list: List[List[int]] = [
            [-1 for _ in range(max_blocks_count)]
            for _ in range(num_layers)
        ]

        self.layer_blocks_holders = [
            LayerBlocksHolder(
                block_info=self._block_info,
                layer_key_ptr_table_device=self._full_ptr_tensor_device[0][layer_i],
                layer_val_ptr_table_device=self._full_ptr_tensor_device[1][layer_i],
                layer_key_ptr_table_cpu=self._full_ptr_tensor_cpu[0][layer_i],
                layer_val_ptr_table_cpu=self._full_ptr_tensor_cpu[1][layer_i],
                layer_key_ptr_table_list=self._device_key_blocks_ptr_list[layer_i],
                layer_val_ptr_table_list=self._device_val_blocks_ptr_list[layer_i],
                contigious_kv_tensor=None,
            )
            for layer_i in range(self._num_layers)
        ]

        self._events: List[torch.cuda.streams.Event] = [
            cast(torch.cuda.Event, torch.cuda.streams.Event())
            for _ in range(self._num_layers)
        ]

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        self._events[0].record(stream)

    @property
    def num_device_blocks(self):
        return len(self._device_tensors)

    def _get_empty_table(self, num: int):
        availables: List[int] = []
        for i in range(self._max_blocks_count):
            if not i in self._device_tensors:
                availables.append(i)
                if len(availables) == num:
                    return availables
        raise ValueError("not enough blocks")

    def _append_blocks_reuse(self, ptrs: List[Tuple[int, int]]):
        num_append = len(ptrs)
        append_blocks: List[int] = [0]*(2*self._num_layers*num_append)
        for append_i, (table_i, ptr) in enumerate(ptrs):
            for layer_i in range(self._num_layers):
                actual_key_ptr = ptr+(layer_i * 2) * self._block_info.bytes_per_block
                actual_val_ptr = ptr+(layer_i * 2+1) * self._block_info.bytes_per_block
                self._device_key_blocks_ptr_list[layer_i][table_i] = actual_key_ptr
                self._device_val_blocks_ptr_list[layer_i][table_i] = actual_val_ptr
                append_blocks[append_i+layer_i*num_append] = actual_key_ptr
                append_blocks[append_i+(layer_i+self._num_layers)*num_append] = actual_val_ptr
        append_table = torch.tensor(
            append_blocks, dtype=torch.int64, device=self._cpu_device, requires_grad=False
        ).view((2, self._num_layers, num_append))
        tables = [i for (i, _) in ptrs]
        self._full_ptr_tensor_cpu[:, :, tables] = append_table
        self._full_ptr_tensor_device[:, :, tables] = append_table.cuda()

    def _alloc_device_blocks(self, num_blocks: int) -> List[int]:
        if len(self._cache_blocks) >= num_blocks:
            bs = self._cache_blocks[:num_blocks]
            self._cache_blocks = self._cache_blocks[num_blocks:]
            return bs
        else:
            max_allowed_blocks = self._max_blocks_count - self.num_device_blocks
            assert num_blocks-self.num_available_caches <= max_allowed_blocks, \
                f"want: {num_blocks} blocks, available: {max_allowed_blocks}, current total: {self.num_device_blocks} max: {self._max_blocks_count}"
            pad_blocks_num = num_blocks+(self._buffer_blocks_count-self.num_available_caches)
            blocks_to_alloc = min(max_allowed_blocks, pad_blocks_num)

            table = self._get_empty_table(blocks_to_alloc)
            new_ptrs: List[Tuple[int, int]] = []
            new_blocks: List[int] = []
            for table_i in table:
                b = torch.empty(
                    size=(self._num_layers, 2, self._block_info.elements_per_block),
                    dtype=self._block_info.dtype,
                    device=self._device
                )
                ptr = b.data_ptr()
                new_blocks.append(table_i)
                new_ptrs.append((table_i, ptr))
                self._device_tensors[table_i] = (ptr, b)
            self._append_blocks_reuse(new_ptrs)
            bs = new_blocks[:num_blocks]
            self._cache_blocks += new_blocks[num_blocks:]
            # print(f"current device blocks: {len(self._device_tensors)}")
            return bs

    @property
    def num_available_caches(self):
        return len(self._cache_blocks)

    def _free_device_block(self, table_i: int):
        # print(f"free: {device_block[0]}")
        if len(self._cache_blocks) < self._buffer_blocks_count:
            # print(f"give back: {device_block[0]}")
            self._cache_blocks.append(table_i)
        else:
            del self._device_tensors[table_i]
            # print(f"current device blocks: {len(self._device_tensors)}")

    def release_block(self, block_idx: int) -> None:
        table_i = self._local_blocks[block_idx].device_table_i
        if table_i != -1:
            self._free_device_block(table_i)
        del self._local_blocks[block_idx]
        # print(f"current local blocks: {len(self._local_blocks)}")

    def release_blocks(self, blocks_idx: List[int]) -> None:
        for b in blocks_idx:
            self.release_block(b)

    def allocate_block(self) -> int:
        blk = self._next_block_idx
        # print("occupy: ", blk)
        self._next_block_idx += 1
        self._local_blocks[blk] = LocalBlock(
            state="empty", ro=False, cpu_tensor=None, device_table_i=-1,
            generation=0,
        )
        return blk

    def allocate_blocks(self, count: int) -> List[int]:
        return [self.allocate_block() for _ in range(count)]

    def put_blocks(self, target_blocks: List[int], kv_cache: List[List[Union[Tuple[torch.Tensor, torch.Tensor], Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]]]) -> None:
        for block_i, local_block_idx in enumerate(target_blocks):
            blk = self._local_blocks[local_block_idx]
            assert blk.state == "empty"
            cpu_block: List[Union[Tuple[torch.Tensor, torch.Tensor], Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]] = []
            for layer_i in range(self._num_layers):
                cpu_block.append(kv_cache[layer_i][block_i])
            blk.state = "cpu"
            blk.cpu_tensor = cpu_block

    def set_blocks_readonly(self, blocks: List[int]):
        if not self._preswap:
            for b in blocks:
                self._local_blocks[b].ro = True
            return
        to_swap: List[int] = []
        for b in blocks:
            blk = self._local_blocks[b]
            if blk.ro:
                continue
            blk.ro = True
            if blk.state == "device":
                to_swap.append(b)
        if len(to_swap) == 0:
            return
        self._pre_swap_out_blocks(target_blocks=to_swap, stream=self._swapout_stream)

    def _pre_swap_out_blocks(self, target_blocks: List[int], stream: torch.cuda.Stream):
        for block_idx in target_blocks:
            blk = self._local_blocks[block_idx]
            assert blk.state == "device" and blk.ro
        self._swap_out_blocks(target_blocks, stream=stream)
        for block_idx in target_blocks:
            blk = self._local_blocks[block_idx]
            blk.state = "ro_both"

    def _swap_out_blocks(self, target_blocks: List[int], stream: torch.cuda.Stream):
        """
        will update events
        """
        if len(target_blocks) == 0:
            return
        src_swap_table_i: List[int] = []
        dst_reciver: List[CPUTensor] = []
        for block_idx in target_blocks:
            blk = self._local_blocks[block_idx]
            assert blk.state != "empty"
            assert blk.state == "device"
            src_swap_table_i.append(blk.device_table_i)
            if blk.cpu_tensor is None:
                blk.cpu_tensor = [
                    (
                        torch.empty(size=self._block_info.key_block_shape, dtype=self._block_info.dtype, device=self._cpu_device),
                        torch.empty(size=self._block_info.value_block_shape, dtype=self._block_info.dtype, device=self._cpu_device)
                    )
                    for _ in range(self._num_layers)
                ]
            dst_reciver.append(blk.cpu_tensor)
        for layer_i, layer_holder in enumerate(self.layer_blocks_holders):
            stream.wait_event(self._events[layer_i])
            layer_holder.swap_out_to_mapping({
                i: (r[layer_i][0].data_ptr(), r[layer_i][1].data_ptr())
                for i, r in zip(src_swap_table_i, dst_reciver)
            }, alt_stream=stream.cuda_stream)
            self._events[layer_i].record(stream)

    def swap_out(self, target_blocks: List[int]):
        """
        must wait events[-1] to sync
        """
        to_swap: List[int] = []
        for block_idx in target_blocks:
            blk = self._local_blocks[block_idx]
            if blk.state == "device":
                to_swap.append(block_idx)
        self._swap_out_blocks(to_swap, self._swapout_stream)
        for block_idx in target_blocks:
            blk = self._local_blocks[block_idx]
            if blk.state == "ro_both":
                blk.state = "cpu"
            elif blk.state == "device":
                blk.state = "cpu"
            else:
                raise ValueError(blk.state)
            assert blk.device_table_i != -1
            b = blk.device_table_i
            blk.device_table_i = -1
            self._free_device_block(b)

    def commit_blocks_generation(
        self, target_blocks: set[int], generation: int
    ) -> None:
        """Stamp the same committed access recorded by the memory twin."""
        if generation <= 0:
            raise ValueError("a committed block generation must be positive")
        missing = target_blocks.difference(self._local_blocks)
        if missing:
            raise KeyError(f"real cache blocks do not exist: {sorted(missing)}")
        for block_idx in target_blocks:
            self._local_blocks[block_idx].generation = generation

    def block_generations(self, target_blocks: set[int]) -> Dict[int, int]:
        return {
            block_idx: self._local_blocks[block_idx].generation
            for block_idx in target_blocks
        }

    def get_blocks(self, target_blocks: List[int]) -> List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]]:
        recivers: List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]] = [
            [
                (DeviceTensorCarrier.place_holder(), DeviceTensorCarrier.place_holder())
                for _ in range(len(target_blocks))
            ] for _ in range(self._num_layers)
        ]
        swap_blocks: List[int] = []
        for block_idx in target_blocks:
            blk = self._local_blocks[block_idx]
            if blk.state == "device":
                swap_blocks.append(block_idx)
        self._swap_out_blocks(swap_blocks, self._swapout_stream)
        for reciver_i, block_idx in enumerate(target_blocks):
            blk = self._local_blocks[block_idx]
            assert blk.state != "empty"
            assert blk.cpu_tensor is not None
            if blk.state == "device" and blk.ro:
                blk.state = "ro_both"
            for layer_i, kv in enumerate(blk.cpu_tensor):
                recivers[layer_i][reciver_i] = (DeviceTensorCarrier.from_data(kv[0]), DeviceTensorCarrier.from_data(kv[1]))
        torch.cuda.current_stream(self._device).wait_event(self._events[-1])
        return recivers

    def get_layer_with_event_ready(self, layer_i: int) -> LayerBlocksHolder:
        self._events[layer_i].wait()
        return self.layer_blocks_holders[layer_i]

    def get_cache_provider(self, related_blocks: List[int]) -> Tuple[LayersCacheProvider, Translator]:
        """
        all swapin happens here
        """
        mapping: Dict[int, int] = {}  # local block i -> table i
        # device_blocks: List[Tuple[int, torch.Tensor]] = []

        device_blocks_to_allocate = 0
        for local_block_idx in related_blocks:
            info = self._local_blocks[local_block_idx]
            if info.state == "empty" or info.state == "cpu":
                device_blocks_to_allocate += 1
        if device_blocks_to_allocate == 0:
            for local_block_idx in related_blocks:
                info = self._local_blocks[local_block_idx]
                assert (info.state == "device") or (info.state == "ro_both")
                assert info.device_table_i != -1
                mapping[local_block_idx] = info.device_table_i
            return self, _Translator(mapping)

        device_blocks = self._alloc_device_blocks(device_blocks_to_allocate)
        swap_in: List[Tuple[int, CPUTensor]] = []  # table i -> cpu_tensor

        for local_block_idx in related_blocks:
            info = self._local_blocks[local_block_idx]
            table_i: int
            if info.state == "device" or info.state == "ro_both":
                assert info.device_table_i != -1
                table_i = info.device_table_i
            elif info.state == "empty":
                table_i = device_blocks.pop()
                info.device_table_i = table_i
                info.state = "device"
            elif info.state == "cpu":
                assert info.cpu_tensor is not None
                table_i = device_blocks.pop()
                info.device_table_i = table_i
                swap_in.append((table_i, info.cpu_tensor))
                if not info.ro:
                    info.state = "device"
                    info.cpu_tensor = None
                else:
                    info.state = "ro_both"
            else:
                raise NotImplementedError
            mapping[local_block_idx] = table_i
        if len(swap_in):
            for layer_i, layer_holder in enumerate(self.layer_blocks_holders):
                self._swapin_stream.wait_event(self._events[layer_i])
                layer_holder.swap_in_from_mapping({
                    i: (lkvs[layer_i][0].data_ptr(), lkvs[layer_i][1].data_ptr())
                    for i, lkvs in swap_in
                }, True, alt_stream=self._swapin_stream.cuda_stream)
                self._events[layer_i].record(self._swapin_stream)
        return self, _Translator(mapping)

    # def _make_device_tensor_ptr_table_in_cpu(self, ptrs: List[int]) -> torch.Tensor:
    #     blocks_ptr_in_cpu: List[int] = [0]*(2*self._num_layers*len(ptrs))
    #     for kv in (0, 1):
    #         for layer_i in range(self._num_layers):
    #             for block_i, ptr in enumerate(ptrs):
    #                 actual_ptr = ptr+(layer_i * 2 + kv) * self._bytes_per_block
    #                 if kv == 0:
    #                     self._device_key_blocks_ptr_list[layer_i][block_i] = actual_ptr
    #                 else:
    #                     self._device_val_blocks_ptr_list[layer_i][block_i] = actual_ptr
    #                 blocks_ptr_in_cpu[block_i+layer_i*len(ptrs)+kv*(len(ptrs)*self._num_layers)] = actual_ptr
    #     return torch.tensor(blocks_ptr_in_cpu, dtype=torch.int64, device=self._cpu_device, requires_grad=False).view((2, self._num_layers, len(ptrs)))

    # def _update_full_ptr_tensor(self):
    #     '''update cache engine ptr table and mapping'''
    #     mapping: Dict[int, int] = {}  # actual ptr -> actual block id
    #     ptrs: List[int] = []
    #     for blk in self.local_blocks.values():
    #         if blk.device_tensor is not None:
    #             ptr = blk.device_tensor[0]
    #             mapping[ptr] = len(ptrs)
    #             ptrs.append(ptr)
    #     assert len(mapping) < self.cache_size

    #     for ptr, _ in self._cache_blocks:
    #         mapping[ptr] = len(ptrs)
    #         ptrs.append(ptr)
    #     self.current_ptr_count = len(ptrs)
    #     remade_ptr_table = self._make_device_tensor_ptr_table_in_cpu(ptrs)
    #     self._full_ptr_tensor_cpu[:, :, : self.current_ptr_count] = remade_ptr_table
    #     self._full_ptr_tensor_device[:, :, : self.current_ptr_count].copy_(remade_ptr_table, non_blocking=True)
    #     self._full_ptr_mapping = mapping

    # def _append_blocks(self, ptrs: List[int]):
    #     '''
    #     append block ptrs to current ptr table
    #     if [self.current_ptr_count...self._max_blocks_count] < blocks to append
    #     just append block ptrs after self.current_ptr_count

    #     '''
    #     num_append = len(ptrs)
    #     if self.current_ptr_count+num_append >= self._max_blocks_count:
    #         raise ValueError
    #         # self._update_full_ptr_tensor()
    #         return
    #     append_blocks: List[int] = [0]*(2*self._num_layers*num_append)
    #     last_count = self.current_ptr_count
    #     for append_i, ptr in enumerate(ptrs):
    #         self._full_ptr_mapping[ptr] = self.current_ptr_count
    #         self.current_ptr_count += 1
    #         for layer_i in range(self._num_layers):
    #             actual_key_ptr = ptr+(layer_i * 2) * self._block_info.bytes_per_block
    #             actual_val_ptr = ptr+(layer_i * 2+1) * self._block_info.bytes_per_block
    #             self._device_key_blocks_ptr_list[layer_i][last_count+append_i] = actual_key_ptr
    #             self._device_val_blocks_ptr_list[layer_i][last_count+append_i] = actual_val_ptr
    #             append_blocks[append_i+layer_i*num_append] = actual_key_ptr
    #             append_blocks[append_i+(layer_i+self._num_layers)*num_append] = actual_val_ptr
    #     append_table = torch.tensor(
    #         append_blocks, dtype=torch.int64, device="cpu", requires_grad=False
    #     ).view((2, self._num_layers, num_append))
    #     # self.infer_stream.synchronize()
    #     self._full_ptr_tensor_cpu[:, :, last_count: last_count+num_append] = append_table
    #     self._full_ptr_tensor_device[:, :, last_count: last_count+num_append].copy_(append_table, non_blocking=True)
