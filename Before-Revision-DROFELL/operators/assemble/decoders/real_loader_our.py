from typing import List, Dict, Optional
import torch
from safetensors import safe_open  # type:ignore
from .common import QuantDecodersConfig

from ...core.decoder_wrap.real_cache import FuseCache
from cuda_extends.exllama import CacheV2, TuningParams
from .real_decoders_group import DecodersGroup
from ..decoder.real import Decoder
from ...core.decoder_wrap.common import FuseConfig
from ...core.rope.real import make_rope_embedding
from ...core.linear.real_quant import QuantLinearMaker


def make_our_decoders_from_files(
    config: QuantDecodersConfig,
    device: torch.device,
    state_files: list[str],
    fuse: bool,
    tunning_params: Optional[TuningParams] = None,
    linear_cache: Optional[CacheV2] = None,
) -> DecodersGroup:
    if linear_cache is None:
        linear_cache = CacheV2(device)
    if tunning_params is None:
        tunning_params = TuningParams()
    fuse_cache = FuseCache(device=device, config=FuseConfig(
        hidden_size=config.decoder_config.hidden_size,
        intermediate_size=config.decoder_config.intermediate_size,
        num_query_heads=config.decoder_config.num_query_heads,
        num_key_value_heads=config.decoder_config.num_key_value_heads,
        head_dim=config.decoder_config.head_dim,
    )) if fuse else None
    current_decoders: List[Decoder] = []
    holder = QuantLinearMaker(quantization_group_size=config.quant_config.group_size, use_act_order=config.quant_config.desc_act)
    rotary_emb = make_rope_embedding(config.rope_config)
    with torch.device(device=device):
        for state_file in state_files:
            decoder = Decoder(
                config=config.decoder_config,
                linear_maker=holder,
                rotary_emb=rotary_emb
            )
            with safe_open(state_file, framework="pt") as f:
                with torch.device("cpu"):
                    raw_state_dict: Dict[str, torch.Tensor] = {k: f.get_tensor(k) for k in f.keys()}
                decoder.load_parameters(raw_state_dict)
                current_decoders.append(decoder)
    holder.post_init(tunning_params, linear_cache)
    linear_cache.init_cache()
    if fuse_cache is not None:
        for decoder in current_decoders:
            decoder.fuse(fuse_cache, linear_cache, tunning_params)
    decoders = DecodersGroup(decoders=current_decoders, device=device, fuse_cache=fuse_cache, attached_no_gc=(linear_cache, fuse_cache))
    decoders.freeze()
    torch.cuda.empty_cache()
    return decoders
