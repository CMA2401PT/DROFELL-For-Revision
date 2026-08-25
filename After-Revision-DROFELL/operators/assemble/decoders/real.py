from typing import Literal, Optional
import torch

from .common import QuantDecodersConfig
from cuda_extends.exllama import CacheV2, TuningParams
from cuda_extends.marlin import MarlinDenseCache, MarlinTuningParams
from .real_decoders_group import DecodersGroup
from .real_loader_our import make_our_decoders_from_files
from .real_loader_marlin import make_marlin_decoders_from_files
from .real_loader_autogptq import make_autogptq_decoders_from_files


DecodersImplement = Literal[
    "triton",
    "exllama_v2",
    "exllama_v1",
    "cuda_old",
    "cuda",
    "marlin",
    "exllama",
    "exllama_fuse",
]


def make_decoders_from_files(
    implement: DecodersImplement = "marlin",
    config: Optional[QuantDecodersConfig] = None,
    state_files: Optional[list[str]] = None,
    tunning_params: Optional[TuningParams] = None,
    linear_cache: Optional[CacheV2] = None,
    marlin_tuning_params: Optional[MarlinTuningParams] = None,
    marlin_linear_cache: Optional[MarlinDenseCache] = None,
) -> DecodersGroup:
    if config is None:
        raise TypeError("make_decoders_from_files requires config")
    if state_files is None:
        raise TypeError("make_decoders_from_files requires state_files")
    device = torch.zeros((0,)).device
    if implement == "marlin":
        return make_marlin_decoders_from_files(
            config=config,
            device=device,
            state_files=state_files,
            tuning_params=marlin_tuning_params,
            linear_cache=marlin_linear_cache,
        )
    if implement == "exllama" or implement == "exllama_fuse":
        return make_our_decoders_from_files(
            config=config,
            device=device, state_files=state_files,
            fuse=implement == "exllama_fuse",
            tunning_params=tunning_params, linear_cache=linear_cache
        )
    else:
        return make_autogptq_decoders_from_files(
            config=config,
            quant_layer_type=implement,
            device=device, state_files=state_files,
        )
