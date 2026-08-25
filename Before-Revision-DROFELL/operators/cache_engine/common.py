from abc import abstractmethod, ABC
from typing import List, Tuple, Union

import torch

from dealers.common.base_types import DeviceTensorCarrier

from ..core.memory.env import Env

from ..core.memory.env import MemoryDict
from ..assemble.decoders.common import LayersCacheProvider


class BlockManager(ABC):
    @abstractmethod
    def allocate_block(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def allocate_blocks(self, count: int) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def release_block(self, block_idx: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def release_blocks(self, blocks_idx: List[int]) -> None:
        raise NotImplementedError


class Translator(ABC):
    @abstractmethod
    def translate_block(self, block_idx: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def translate_slot(self, slot_i: int) -> int:
        raise NotImplementedError


class CacheEngine(BlockManager):
    @abstractmethod
    def put_blocks(self, target_blocks: List[int], kv_cache: List[List[Union[Tuple[torch.Tensor, torch.Tensor], Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]]]) -> None:
        '''
        kv_cache = List[ # layers
            List[ # block_i
                Tuple[torch.Tensor, torch.Tensor] # k,v
            ]
        ]    
        '''
        raise NotImplementedError

    @abstractmethod
    def set_blocks_readonly(self, blocks: List[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_blocks(self, target_blocks: List[int]) -> List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]]:
        '''
        List[ # layers
            List[ # block_i
                Tuple[torch.Tensor, torch.Tensor] # k,v
            ]
        ]    
        '''
        raise NotImplementedError

    @abstractmethod
    def get_cache_provider(self, related_blocks: List[int]) -> Tuple[LayersCacheProvider, Translator]:
        raise NotImplementedError

    @abstractmethod
    def record_stream(self, stream: torch.cuda.Stream) -> None:
        raise NotImplementedError


class MemoryCacheEngine(MemoryDict, ABC):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

    @abstractmethod
    def allocate_blocks(self, count: int) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def allocate_block(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def set_blocks_readonly(self, blocks: List[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def release_blocks(self, blocks_idx: List[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_cache_provider(self, related_blocks: List[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_blocks(self, target_blocks: List[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def put_blocks(self, target_blocks: List[int]) -> None:
        raise NotImplementedError
