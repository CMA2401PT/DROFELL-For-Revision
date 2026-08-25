from typing import Tuple
from dataclasses import dataclass
from operators import flags
import torch


@dataclass
class BlockInfo:
    head_dim: int
    num_key_value_heads: int
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        self.element_size = int(self.dtype.itemsize)
        self._x = 16 // self.element_size
        self._head_stride = flags.BLOCK_SIZE * self.head_dim
        self._elements_per_slot = self.num_key_value_heads * self.head_dim
        self._elements_per_block = flags.BLOCK_SIZE * self._elements_per_slot
        self._bytes_per_block = self._elements_per_block * self.element_size

    @property
    def key_block_shape(self) -> Tuple[int, int, int, int]:
        return (
            self.num_key_value_heads,
            self.head_dim // self._x,
            flags.BLOCK_SIZE,
            self._x,
        )

    @property
    def value_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_key_value_heads,
            self.head_dim,
            flags.BLOCK_SIZE,
        )

    @property
    def num_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def head_size(self) -> int:
        return self.head_dim

    @property
    def bytes_per_block(self) -> int:
        return self._bytes_per_block

    @property
    def x(self) -> int:
        return self._x

    @property
    def head_stride(self) -> int:
        return self._head_stride

    @property
    def elements_per_block(self) -> int:
        return self._elements_per_block

    def tensor_kv_to_block_kv(self,
                              key_states: torch.Tensor,
                              val_states: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        convert from kv format to page atten format
        key_states,val_states =[num_tokens, num_heads, head_size]
        """
        num_tokens, num_heads, head_size = key_states.shape
        assert (num_tokens % flags.BLOCK_SIZE) == 0
        key_cache = key_states.view(num_tokens // flags.BLOCK_SIZE, flags.BLOCK_SIZE, num_heads, head_size // self.x, self.x)
        val_cache = val_states.view(num_tokens // flags.BLOCK_SIZE, flags.BLOCK_SIZE, num_heads, head_size)
        key_cache = key_cache.permute(0, 2, 3, 1, 4).contiguous()
        val_cache = val_cache.permute(0, 2, 3, 1).contiguous()
        return key_cache, val_cache

    @staticmethod
    def block_kv_to_tensor_kv(key_states: torch.Tensor, val_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        convert from page atten  format to kv format
        '''
        num_blocks, num_heads, head_div_x, block_size, x = key_states.shape
        num_tokens = num_blocks * block_size
        head_size = head_div_x * x

        # key: [num_blocks, block_size, num_heads, head_div_x, x]
        key_tensor = key_states.permute(0, 3, 1, 2, 4).contiguous()
        key_tensor = key_tensor.view(num_tokens, num_heads, head_size)

        # val: [num_blocks, block_size, num_heads, head_size]
        val_tensor = val_states.permute(0, 3, 1, 2).contiguous()
        val_tensor = val_tensor.view(num_tokens, num_heads, head_size)

        return key_tensor, val_tensor
