from typing import List, Any, Dict
import torch
from safetensors import safe_open  # type:ignore

from .common import QuantDecodersConfig

from .real_decoders_group import DecodersGroup
from ..decoder.real import Decoder
from ...core.rope.real import make_rope_embedding
from ...core.linear.real_auto_gptq import AutoGPTQQuantLinearsType, QuantLinearHolder

device_aux: Dict[str, Dict[str, Any]] = {}


def get_device_aux(device: torch.device) -> Dict[str, Any]:
    device_key = f"{device}"
    if device_key in device_aux.keys():
        return device_aux[device_key]
    aux: Dict[str, Any] = {}
    device_aux[device_key] = aux
    return aux


def make_autogptq_decoders_from_files(
    config: QuantDecodersConfig,
    quant_layer_type: AutoGPTQQuantLinearsType,
    device: torch.device,
    state_files: list[str]
) -> DecodersGroup:
    current_decoders: List[Decoder] = []
    holder = QuantLinearHolder(
        group_size=config.quant_config.group_size, bits=config.quant_config.bits, desc_act=config.quant_config.desc_act,
        quant_linear_type=quant_layer_type,
    )
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
                    raw_state_dict: Dict[str, torch.Tensor] = {k: f.get_tensor(k).to(device) for k in f.keys()}
                decoder.load_parameters(raw_state_dict)
                current_decoders.append(decoder)
        no_gc = holder.post_init(max_input_length=1024)
        warmup_history = get_device_aux(device)
        holder.warmup(seqlen=1024, warmup_history=warmup_history)

        decoders = DecodersGroup(decoders=current_decoders, device=device, fuse_cache=None, attached_no_gc=no_gc)
        decoders.freeze()
        torch.cuda.empty_cache()
        return decoders
