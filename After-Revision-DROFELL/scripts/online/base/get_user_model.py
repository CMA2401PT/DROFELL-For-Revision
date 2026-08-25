from typing import Callable, List, Optional, Tuple, cast

import torch

import data
from dealers.common.base_types import DeviceTensorCarrier
from operators.core.lora.render import LoraRenderConfig
from operators.core.lora.real import Lora
from operators.core.ptuning.real import Ptuning
from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig
from operators.helpers.configs.quant_llama import LlamaConfig


def gen_user_model(
        user_model_name: str,
        base_model_config: LlamaConfig,
        model_path: Optional[str] = None
    ) -> Tuple[
    Optional[Tuple[
        str, Callable[[Tuple[int, int]], Tuple[
            List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]],
            Optional[LoraRenderConfig]
        ]]
    ]],
    Callable[[str], str]
]:
    if user_model_name == "":
        return None, lambda prompt=str: prompt
    if user_model_name == "advgen-ptuning":
        cfg = PtuningConfig(
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.1,
            prefix_len=128
        )
        if model_path is None:
            model_path = data.trained_model_path_of("ptuning_7B.pt")
    elif user_model_name == "advgen-lora":
        cfg = LoraConfig(
            r_q=-1,
            r_k=0,
            r_v=-1,
            build_prompt_str="[instruction]{}\n[generation]",
            dropout_p=0.0,
            len_limit=128,
        )
        if model_path is None:
            model_path = data.trained_model_path_of("lora_7B.pt")
    else:
        raise ValueError(user_model_name)
    lora_cfg = None
    is_lora = isinstance(cfg, LoraConfig)
    if is_lora:
        lora_cfg = cfg.render_config
        cfg.check_or_set_decoders_config(
            num_layers=base_model_config.num_decoders,
            head_dim=base_model_config.head_dim,
            num_key_value_heads=base_model_config.num_key_value_heads,
            num_query_heads=base_model_config.num_query_heads
        )
    else:
        cfg.check_or_set_decoders_config(
            num_layers=base_model_config.num_decoders,
            num_key_value_heads=base_model_config.num_key_value_heads,
            head_dim=base_model_config.head_dim
        )
    user_model = Lora(cfg) if is_lora else Ptuning(cfg)
    state_dict = torch.load(model_path, map_location="cpu")
    user_model.load_state_dict(state_dict)
    user_data =\
        cast(Lora, user_model).get_inference_lora_blocks(torch.float16) \
        if is_lora else \
        cast(Ptuning, user_model).get_inference_ptuning_kv_cache_with_convert(torch.float16)
    user_data = [
        [
            (DeviceTensorCarrier.from_data(k), DeviceTensorCarrier.from_data(v))
            for (k, v) in layer_blocks
        ]
        for layer_blocks in user_data
    ]

    def slice(range: Tuple[int, int]) ->\
        Tuple[
            List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]],
            Optional[LoraRenderConfig]
    ]:
        return user_data[range[0]:range[1]], lora_cfg
    return (user_model_name, slice), cfg.build_prompt
