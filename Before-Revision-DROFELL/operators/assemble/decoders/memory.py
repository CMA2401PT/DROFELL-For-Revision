from typing import Any, List, Literal, Optional, Set

import torch

from .real import DecodersImplement

from ...core.rope.memory import make_rope_embedding_memory_model

from ...core.linear.memory import QuantLinearMakerMemoryModel

from ...core.decoder_wrap.common import FuseConfig

from cuda_extends.exllama import TuningParams

from ...core.block.common import BlockInfo

from ..decoder.memory import DecoderMemoryModel
from ...core.decoder_wrap.memory import CacheV2MemoryModel, FuseCacheMemoryModel
from ...core.memory import MemoryModule
from .common import QuantDecodersConfig


class DecodersGroupMemoryModel(MemoryModule):
    def __init__(self,
                 decoders: List[DecoderMemoryModel],
                 fuse_cache: Optional[FuseCacheMemoryModel],
                 attached_no_gc: Any = None) -> None:
        super().__init__()
        self.decoders = decoders
        self.mount_list(decoders, "decoders")
        self.fuse_cache = fuse_cache
        self.attached_no_gc = attached_no_gc
        self.mount_list(attached_no_gc, "no_gc")
        self.non_free_addrs: Set[int] = set()
        self._block_info = BlockInfo(
            head_dim=decoders[0].config.head_dim,
            num_key_value_heads=decoders[0].config.num_key_value_heads,
            dtype=torch.float16
        )
        self._cfg = decoders[0].cfg

    @property
    def block_info(self):
        return self._block_info

    @property
    def num_layers(self):
        return len(self.decoders)

    @property
    def config(self):
        return self._cfg


def make_our_decoders_memory_model(
    config: QuantDecodersConfig,
    state_files: list[str],
    fuse: bool,
    tunning_params: Optional[TuningParams] = None,
    linear_cache: Optional[CacheV2MemoryModel] = None,
):
    if linear_cache is None:
        linear_cache = CacheV2MemoryModel()
    if tunning_params is None:
        tunning_params = TuningParams()
    fuse_cache = FuseCacheMemoryModel(config=FuseConfig(
        hidden_size=config.decoder_config.hidden_size,
        intermediate_size=config.decoder_config.intermediate_size,
        num_query_heads=config.decoder_config.num_query_heads,
        num_key_value_heads=config.decoder_config.num_key_value_heads,
        head_dim=config.decoder_config.head_dim,
    )) if fuse else None
    current_decoders: List[DecoderMemoryModel] = []
    holder = QuantLinearMakerMemoryModel(quantization_group_size=config.quant_config.group_size, use_act_order=config.quant_config.desc_act)
    rotary_emb = make_rope_embedding_memory_model(config.rope_config)
    for state_file in state_files:
        decoder = DecoderMemoryModel(
            config=config.decoder_config,
            linear_maker=holder,
            rotary_emb=rotary_emb
        )
        current_decoders.append(decoder)
    holder.post_init(tunning_params, linear_cache)
    linear_cache.init_cache()
    if fuse_cache is not None:
        for decoder in current_decoders:
            decoder.fuse(fuse_cache, linear_cache, tunning_params)
    decoders = DecodersGroupMemoryModel(decoders=current_decoders, fuse_cache=fuse_cache, attached_no_gc=(linear_cache, fuse_cache))
    return decoders


def make_decoders_memory_model(
    implement: DecodersImplement,
    config: QuantDecodersConfig,
    state_files: list[str],
    tunning_params: Optional[TuningParams] = None,
    linear_cache: Optional[CacheV2MemoryModel] = None,
) -> DecodersGroupMemoryModel:
    device = torch.zeros((0,)).device
    if implement == "our" or implement == "our_fuse":
        return make_our_decoders_memory_model(
            config=config,
            fuse=implement == "our_fuse",
            state_files=state_files,
            tunning_params=tunning_params,
            linear_cache=linear_cache
        )
    else:
        raise ValueError(implement)
