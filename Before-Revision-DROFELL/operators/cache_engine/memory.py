
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set, Tuple
import torch
from .scalable import BlockStatus
from .common import MemoryCacheEngine
from ..core.memory.static import MemoryTensor
from ..core.block.common import BlockInfo
from ..core.memory.env import Env, MemoryDict


class VllmFixCacheEngineMemoryModel(MemoryCacheEngine):
    def __init__(self,
                 env: Env,
                 block_info: BlockInfo,
                 num_cpu_blocks: int,
                 num_device_blocks: int,
                 num_layers: int,
                 mode: Literal["vllm", "dual_table"] = "dual_table",
                 translate: bool = True,
                 ) -> None:
        super().__init__(env)
        self._block_info = block_info
        self._num_layers = num_layers
        self._num_device_blocks = num_device_blocks
        self._num_cpu_blocks = num_cpu_blocks
        self.mode = mode
        self.translate = translate
        self.available_blocks: List[int] = [
            i+1000 for i in range(num_device_blocks)
        ] if self.translate else [
            i for i in range(num_device_blocks)
        ]  # 1000 is just for debug, avoiding confusing with idx
        self.occupied_blocks: Dict[int, int] = {b: 0 for b in self.available_blocks}

        self._allocate_device_cache()

    def _allocate_device_cache(self):
        device_cache, key_blocks_ptr, val_blocks_ptr = self._allocate_blocks(num_layers=self._num_layers, num_blocks=self._num_device_blocks,
                                                                             key_block_shape=self._block_info.key_block_shape, value_block_shape=self._block_info.value_block_shape,
                                                                             dtype=self._block_info.dtype)
        self.set_item("key_blocks_ptr", key_blocks_ptr)
        self.set_item("val_blocks_ptr", val_blocks_ptr)
        for i, (k, v) in enumerate(device_cache):
            self.set_item(f"layer-{i}-key", k)
            self.set_item(f"layer-{i}-val", v)

    @staticmethod
    def _allocate_blocks(
            num_layers: int, num_blocks: int,
            key_block_shape: Tuple[int, int, int, int], value_block_shape: Tuple[int, int, int],
            dtype: torch.dtype):
        # holder: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [[]for _ in range(num_layers)]
        vllm_style_holder: List[Tuple[MemoryTensor, MemoryTensor]] = []
        for layer_i in range(num_layers):
            large_key_pre_allocate_block = MemoryTensor(size=(num_blocks, *key_block_shape), dtype=dtype)
            large_val_pre_allocate_block = MemoryTensor(size=(num_blocks, *value_block_shape), dtype=dtype)
            vllm_style_holder.append((large_key_pre_allocate_block, large_val_pre_allocate_block))
        key_blocks_ptr = MemoryTensor((num_layers, num_blocks), dtype=torch.int64)
        val_blocks_ptr = MemoryTensor((num_layers, num_blocks), dtype=torch.int64)
        return vllm_style_holder, key_blocks_ptr, val_blocks_ptr

    def allocate_blocks(self, count: int):
        orig = self.available_blocks.copy()
        bs: List[int] = self.available_blocks[:count]
        self.available_blocks = self.available_blocks[count:]
        for b in bs:
            assert self.occupied_blocks[b] == 0
            self.occupied_blocks[b] = 1

        def revert():
            self.available_blocks = orig
            for b in bs:
                self.occupied_blocks[b] = 0
        self.env.revert_stack.append(revert)
        return bs

    def allocate_block(self):
        orig = self.available_blocks.copy()
        i = self.available_blocks[0]
        self.available_blocks = self.available_blocks[1:]
        self.occupied_blocks[i] = 1

        def revert():
            self.available_blocks = orig
            self.occupied_blocks[i] = 0
        self.env.revert_stack.append(revert)
        return i

    def release_blocks(self, blocks_idx: List[int]) -> None:
        orig = self.available_blocks.copy()
        for b in blocks_idx:
            assert self.occupied_blocks[b] > 0
            self.occupied_blocks[b] -= 1
            if self.occupied_blocks[b] == 0:
                self.available_blocks.append(b)

        def revert():
            self.available_blocks = orig
            for b in blocks_idx:
                self.occupied_blocks[b] += 1
        self.env.revert_stack.append(revert)

    def set_blocks_readonly(self, blocks: List[int]) -> None:
        pass

    def get_cache_provider(self, related_blocks: List[int]):
        pass

    def get_blocks(self, target_blocks: List[int]) -> None:
        pass

    def put_blocks(self, target_blocks: List[int]) -> None:
        pass


@dataclass
class LocalBlock:
    state: BlockStatus
    ro: bool
    cpu_tensor: Optional[bool] = None  # layers*(k,v)
    device_table_i: int = -1

    def clone(self):
        return LocalBlock(self.state, self.ro, self.cpu_tensor, self.device_table_i)


class ScalableCacheEngineMemoryModel(MemoryCacheEngine):
    def __init__(self,
                 env: Env,
                 block_info: BlockInfo,
                 num_layers: int,
                 max_blocks_count: int,
                 buffer_blocks_count: int = 16,
                 preswap: bool = False
                 ) -> None:
        super().__init__(env)
        self.env = env
        self._block_info = block_info
        self._bytes_per_block = self._block_info.bytes_per_block*2*num_layers
        self._num_layers = num_layers
        self._max_blocks_count = max_blocks_count
        # self.current_ptr_count = 0
        self._buffer_blocks_count = buffer_blocks_count
        self.preswap = preswap
        self._next_block_idx = 1000  # 1000 is just for debug, avoiding confusing with idx
        self.local_blocks: Dict[int, LocalBlock] = {}
        self.device_local_blocks: Set[int] = set()
        self._cache_blocks: List[int] = []
        self.device_tensors = MemoryDict(env)
        self.set_item("device_ptr_table", MemoryTensor(size=(2, num_layers, self._max_blocks_count), dtype=torch.int64))
        self.set("blocks", self.device_tensors)
        self.per_block_elements = self._num_layers * 2 * self._block_info.elements_per_block
        self.per_block_bytes = self.per_block_elements*2
        self.max_buffer_bytes = self._buffer_blocks_count*self.per_block_bytes

    def allocate_block(self) -> int:
        blk = self._next_block_idx
        # print("occupy: ", blk)
        self._next_block_idx += 1
        self.local_blocks[blk] = LocalBlock(state="empty", ro=False, cpu_tensor=None, device_table_i=-1)

        def revert():
            del self.local_blocks[blk]
            self._next_block_idx -= 1
        self.env.revert_stack.append(revert)
        return blk

    def allocate_blocks(self, count: int) -> List[int]:
        bs: List[int] = [0]*count
        orig = self._next_block_idx
        for i in range(count):
            blk = self._next_block_idx
            self._next_block_idx += 1
            self.local_blocks[blk] = LocalBlock(state="empty", ro=False, cpu_tensor=None, device_table_i=-1)
            bs[i] = blk

        def revert():
            nonlocal orig
            self._next_block_idx = orig
            for i in range(count):
                blk = orig+i
                del self.local_blocks[blk]
        self.env.revert_stack.append(revert)
        return bs

    def _free_device_block(self, table_i: int):
        if len(self._cache_blocks) < self._buffer_blocks_count:
            self._cache_blocks.append(table_i)

            def revert():
                self._cache_blocks.pop()
            self.env.revert_stack.append(revert)
        else:
            del self.device_tensors[table_i]

    def release_blocks(self, blocks_idx: List[int]) -> None:
        orig: List[Optional[LocalBlock]] = [None]*len(blocks_idx)
        for i, b in enumerate(blocks_idx):
            table_i = self.local_blocks[b].device_table_i
            if table_i != -1:
                self._free_device_block(table_i)
                self.device_local_blocks.remove(b)
            orig[i] = self.local_blocks[b]
            del self.local_blocks[b]

        def revert():
            for i, b in enumerate(blocks_idx):
                _ob = orig[i]
                assert _ob is not None
                self.local_blocks[b] = _ob
                if _ob.device_table_i != -1:
                    self.device_local_blocks.add(b)
        self.env.revert_stack.append(revert)

    def set_blocks_readonly(self, blocks: List[int]) -> None:
        orig: List[bool] = [False]*len(blocks)
        if not self.preswap:
            for i, b in enumerate(blocks):
                orig[i] = self.local_blocks[b].ro
                self.local_blocks[b].ro = True
        else:
            for b in blocks:
                blk = self.local_blocks[b]
                if blk.ro:
                    continue
                blk.ro = True
                if blk.state == "device":
                    blk.state = "ro_both"

        def revert():
            for i, b in enumerate(blocks):
                self.local_blocks[b].ro = orig[i]
        self.env.revert_stack.append(revert)

    @property
    def num_device_blocks(self):
        return len(self.device_tensors)

    @property
    def num_available_caches(self):
        return len(self._cache_blocks)

    def _get_empty_table(self, num: int):
        availables: List[int] = []
        for i in range(self._max_blocks_count):
            if not i in self.device_tensors:
                availables.append(i)
                if len(availables) == num:
                    return availables
        raise ValueError("not enough blocks")

    def _alloc_device_blocks(self, num_blocks: int) -> List[int]:
        if len(self._cache_blocks) >= num_blocks:
            orig = self._cache_blocks.copy()
            bs = self._cache_blocks[:num_blocks]
            self._cache_blocks = self._cache_blocks[num_blocks:]

            def revert():
                self._cache_blocks = orig
            self.env.revert_stack.append(revert)
            return bs
        else:
            max_allowed_blocks = self._max_blocks_count - self.num_device_blocks
            assert num_blocks-self.num_available_caches <= max_allowed_blocks, \
                f"want: {num_blocks} blocks, available: {max_allowed_blocks}, current total: {self.num_device_blocks} max: {self._max_blocks_count}"
            pad_blocks_num = num_blocks+(self._buffer_blocks_count-self.num_available_caches)
            blocks_to_alloc = min(max_allowed_blocks, pad_blocks_num)

            table = self._get_empty_table(blocks_to_alloc)
            new_blocks: List[int] = []
            for table_i in table:
                b = self.env.trace(MemoryTensor((self._num_layers, 2, self._block_info.elements_per_block), dtype=self._block_info.dtype))
                new_blocks.append(table_i)
                self.device_tensors.set(table_i, b)
            bs = new_blocks[:num_blocks]
            orig = self._cache_blocks.copy()
            self._cache_blocks += new_blocks[num_blocks:]

            def revert():
                self._cache_blocks = orig
            self.env.revert_stack.append(revert)
            return bs

    def get_cache_provider(self, related_blocks: List[int]):
        device_blocks_to_allocate = 0
        for local_block_idx in related_blocks:
            info = self.local_blocks[local_block_idx]
            if info.state == "empty" or info.state == "cpu":
                device_blocks_to_allocate += 1
        if device_blocks_to_allocate == 0:
            return
        device_blocks = self._alloc_device_blocks(device_blocks_to_allocate)
        orig: Dict[int, LocalBlock] = {}
        for local_block_idx in related_blocks:
            info = self.local_blocks[local_block_idx]
            table_i: int
            if info.state == "device" or info.state == "ro_both":
                continue
            elif info.state == "empty":
                orig[local_block_idx] = info.clone()
                table_i = device_blocks.pop()
                info.device_table_i = table_i
                self.device_local_blocks.add(local_block_idx)
                info.state = "device"
            elif info.state == "cpu":
                orig[local_block_idx] = info.clone()
                assert info.cpu_tensor is not None
                table_i = device_blocks.pop()
                info.device_table_i = table_i
                self.device_local_blocks.add(local_block_idx)
                if not info.ro:
                    info.state = "device"
                    info.cpu_tensor = None
                else:
                    info.state = "ro_both"
            else:
                raise NotImplementedError

        def revert():
            for local_block_idx, b in orig.items():
                self.local_blocks[local_block_idx] = b
                if b.device_table_i == -1:
                    self.device_local_blocks.remove(local_block_idx)
        self.env.revert_stack.append(revert)

    def swap_out_block(self, target_block: int) -> None:
        blk = self.local_blocks[target_block]
        orig = blk.clone()

        blk.cpu_tensor = True
        if blk.state == "ro_both":
            blk.state = "cpu"
        elif blk.state == "device":
            blk.state = "cpu"
        else:
            raise ValueError(blk.state)
        assert blk.device_table_i != -1
        b = blk.device_table_i
        blk.device_table_i = -1
        self.device_local_blocks.remove(target_block)
        self._free_device_block(b)

        def revert():
            self.local_blocks[target_block] = orig
            self.device_local_blocks.add(target_block)
        self.env.revert_stack.append(revert)

    def swap_out(self, target_blocks: List[int]) -> None:
        for block_idx in target_blocks:
            self.swap_out_block(block_idx)

    def get_blocks(self, target_blocks: List[int]) -> None:
        origs = [
            self.local_blocks[block_idx].clone()
            for block_idx in target_blocks
        ]
        for block_idx in target_blocks:
            blk = self.local_blocks[block_idx]
            if blk.state == "device":
                if blk.cpu_tensor is None:
                    blk.cpu_tensor = True
                if blk.ro:
                    blk.state = "ro_both"

        def revert():
            for i, b in zip(target_blocks, origs):
                self.local_blocks[i] = b
        self.env.revert_stack.append(revert)

    def put_blocks(self, target_blocks: List[int]) -> None:
        origs = [
            self.local_blocks[block_idx].clone()
            for block_idx in target_blocks
        ]
        for local_block_idx in target_blocks:
            blk = self.local_blocks[local_block_idx]
            assert blk.state == "empty"
            blk.state = "cpu"
            blk.cpu_tensor = True

        def revert():
            for i, b in zip(target_blocks, origs):
                self.local_blocks[i] = b
        self.env.revert_stack.append(revert)
