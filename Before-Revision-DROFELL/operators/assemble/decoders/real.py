from typing import Literal, Optional
import torch

from .common import QuantDecodersConfig
from cuda_extends.exllama import CacheV2, TuningParams
from .real_decoders_group import DecodersGroup
from .real_loader_our import make_our_decoders_from_files
from .real_loader_autogptq import make_autogptq_decoders_from_files


DecodersImplement = \
    Literal["triton", "exllama_v2", "exllama_v1", "cuda_old", "cuda", "our", "our_fuse"]


def make_decoders_from_files(
    implement: DecodersImplement,
    config: QuantDecodersConfig,
    state_files: list[str],
    tunning_params: Optional[TuningParams] = None,
    linear_cache: Optional[CacheV2] = None,
) -> DecodersGroup:
    device = torch.zeros((0,)).device
    if implement == "our" or implement == "our_fuse":
        return make_our_decoders_from_files(
            config=config,
            device=device, state_files=state_files,
            fuse=implement == "our_fuse",
            tunning_params=tunning_params, linear_cache=linear_cache
        )
    else:
        return make_autogptq_decoders_from_files(
            config=config,
            quant_layer_type=implement,
            device=device, state_files=state_files,
        )
