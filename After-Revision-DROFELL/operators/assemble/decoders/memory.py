from typing import Any, List, Optional, Set

import torch

from .real import DecodersImplement

from ...core.rope.memory import make_rope_embedding_memory_model

from ...core.linear.memory import (
    MarlinQuantLinearMakerMemoryModel,
    QuantLinearMakerMemoryModel,
)

from ...core.decoder_wrap.common import FuseConfig

from cuda_extends.exllama import TuningParams

from ...core.block.common import BlockInfo

from ..decoder.memory import DecoderMemoryModel
from ...core.decoder_wrap.memory import CacheV2MemoryModel, FuseCacheMemoryModel
from ...core.memory import MemoryModule, MemoryTensor
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
        # Non-fused ExLlama has no fuse cache.  Do not register that optional
        # ``None`` as a memory submodule; ``MemoryModule.parameters`` expects
        # every registered child to implement ``parameters``.
        attached = tuple(
            item for item in (attached_no_gc or ()) if item is not None
        )
        self.attached_no_gc = attached
        self.mount_list(attached, "no_gc")
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


def _merge_marlin_decoder_projections(
    decoder: DecoderMemoryModel,
    linear_maker: MarlinQuantLinearMakerMemoryModel,
) -> None:
    attention = decoder.self_attn
    attention.qkv_proj = linear_maker.merge(
        [attention.q_proj, attention.k_proj, attention.v_proj]
    )
    for name in ("q_proj", "k_proj", "v_proj"):
        attention._modules.pop(name)
        object.__setattr__(attention, name, None)

    mlp = decoder.mlp
    mlp.gate_up_proj = linear_maker.merge(
        [mlp.gate_proj, mlp.up_proj]
    )
    for name in ("gate_proj", "up_proj"):
        mlp._modules.pop(name)
        object.__setattr__(mlp, name, None)


def make_marlin_decoders_memory_model(
    config: QuantDecodersConfig,
    state_files: list[str],
    workspace_elements: int,
) -> DecodersGroupMemoryModel:
    linear_maker = MarlinQuantLinearMakerMemoryModel(
        quantization_group_size=config.quant_config.group_size,
        use_act_order=config.quant_config.desc_act,
        workspace_elements=workspace_elements,
    )
    rotary_emb = make_rope_embedding_memory_model(config.rope_config)
    # The real Marlin loader calls decoder.half() after loading.
    rotary_emb.inv_freq = MemoryTensor(
        rotary_emb.inv_freq.size, dtype=torch.float16
    )
    decoders = [
        DecoderMemoryModel(
            config=config.decoder_config,
            linear_maker=linear_maker,
            rotary_emb=rotary_emb,
        )
        for _ in state_files
    ]
    for decoder in decoders:
        _merge_marlin_decoder_projections(decoder, linear_maker)
        decoder.input_layernorm.weight = MemoryTensor(
            (config.decoder_config.hidden_size,), dtype=torch.float16
        )
        decoder.post_attention_layernorm.weight = MemoryTensor(
            (config.decoder_config.hidden_size,), dtype=torch.float16
        )
    linear_maker.post_init()
    return DecodersGroupMemoryModel(
        decoders=decoders,
        fuse_cache=None,
        attached_no_gc=(linear_maker.dense_cache,),
    )


def make_decoders_memory_model(
    implement: DecodersImplement,
    config: QuantDecodersConfig,
    state_files: list[str],
    tunning_params: Optional[TuningParams] = None,
    linear_cache: Optional[CacheV2MemoryModel] = None,
    marlin_workspace_elements: int = 0,
) -> DecodersGroupMemoryModel:
    if implement == "exllama" or implement == "exllama_fuse":
        return make_our_decoders_memory_model(
            config=config,
            fuse=implement == "exllama_fuse",
            state_files=state_files,
            tunning_params=tunning_params,
            linear_cache=linear_cache
        )
    if implement == "marlin":
        return make_marlin_decoders_memory_model(
            config=config,
            state_files=state_files,
            workspace_elements=marlin_workspace_elements,
        )
    raise ValueError(implement)
