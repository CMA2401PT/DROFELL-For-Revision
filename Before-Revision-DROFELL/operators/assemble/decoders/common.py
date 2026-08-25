from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import torch


from ... import flags
from ...core.block.real import LayerBlocksHolder
from ...core.decoder_wrap.common import DecoderConfig, LoraHook
from ...core.rope.common import RopeConfig
from ...core.linear.common import QuantizationConfig


@dataclass
class QuantDecodersConfig:
    decoder_config: DecoderConfig
    rope_config: RopeConfig
    quant_config: QuantizationConfig


class LayersCacheProvider(ABC):
    @abstractmethod
    def get_layer_with_event_ready(
        self,
        layer_i: int,
    ) -> LayerBlocksHolder:
        '''
        should consider swap event
        '''
        raise NotImplementedError


class LayersPrefillCacheProvider(ABC):
    @abstractmethod
    def get_layer_with_event_ready(
        self,
        layer_i: int,
    ) -> Tuple[
        LayerBlocksHolder,
        Callable[
            [int], Optional[Tuple[torch.Tensor, torch.Tensor]]
        ]
    ]:
        '''
        should consider swap event
        '''
        raise NotImplementedError


class LayersLoraHookProvider(ABC):
    @abstractmethod
    def get_lora_hook(
        self,
        layer_i: int,
    ) -> Optional[LoraHook]:
        '''
        should consider swap event
        '''
        raise NotImplementedError


class SimpleLayersCacheProvider(LayersCacheProvider):
    def __init__(
        self,
        layers_blocks_holder: List[LayerBlocksHolder],
        events: Optional[List[Optional[torch.cuda.Event]]]
    ) -> None:
        self.layers_blocks_holder = layers_blocks_holder
        self.events = events

    def get_layer_with_event_ready(
        self,
        layer_i: int
    ) -> LayerBlocksHolder:
        if self.events is not None:
            event = self.events[layer_i]
            if event is not None:
                event.wait()
        return self.layers_blocks_holder[layer_i]


class SimpleLayersPrefillCacheProvider(LayersPrefillCacheProvider):
    def __init__(
        self,
        seq_ptunings: List[Optional[List[int]]],  # [[ptuning_blocks]? * num_seqs],
        cache_provider: LayersCacheProvider,
    ) -> None:
        self.cache_provider = cache_provider
        self.indexer, self.ptunings_slots = self._squeeze_ptunings(seq_ptunings)

    def get_layer_with_event_ready(
        self,
        layer_i: int,
    ) -> Tuple[
        LayerBlocksHolder,
        Callable[
            [int], Optional[Tuple[torch.Tensor, torch.Tensor]]
        ]
    ]:
        cacher = self.cache_provider.get_layer_with_event_ready(layer_i)
        recovered_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for ptuning_slots in self.ptunings_slots:
            kv = cacher.kv_tensor_from_slots_with_convert_with_create(
                slot_mapping=ptuning_slots
            )
            recovered_kv_caches.append(kv)

        def _getter(seq_i: int):
            cache_i = self.indexer.get(seq_i)
            if cache_i is None:
                return None
            else:
                return recovered_kv_caches[cache_i]
        return cacher, _getter

    def _squeeze_ptunings(self, seq_ptunings: List[Optional[List[int]]]):
        squeeze_ptunings: List[List[int]] = []
        new_index: Dict[Tuple[int, ...], int] = {}
        old2new_mapping: Dict[int, int] = {}
        for (old_i, ptuning_blocks_list) in enumerate(seq_ptunings):
            if ptuning_blocks_list is None:
                continue
            ptuning_blocks_tuple = tuple(ptuning_blocks_list)
            new_i = new_index.get(ptuning_blocks_tuple)
            if new_i is None:
                squeeze_ptunings.append(ptuning_blocks_list)
                new_i = len(new_index)
                new_index[ptuning_blocks_tuple] = new_i
            old2new_mapping[old_i] = new_i
        ptunings_slots: List[torch.Tensor] = []
        for ptuning_blocks in squeeze_ptunings:
            ptuning_slots = [
                b*flags.BLOCK_SIZE+i
                for b in ptuning_blocks
                for i in range(flags.BLOCK_SIZE)
            ]
            ptunings_slots.append(torch.tensor(ptuning_slots, dtype=torch.int64))
        return old2new_mapping, ptunings_slots
