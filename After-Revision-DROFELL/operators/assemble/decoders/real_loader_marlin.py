"""Load frozen GPTQ decoders into the Marlin inference/PEFT backend."""

from typing import Dict, List, Optional

import torch
from safetensors import safe_open  # type: ignore

from cuda_extends.marlin import MarlinDenseCache, MarlinTuningParams

from ...core.linear.real_marlin import MarlinQuantLinearMaker
from ...core.rope.real import make_rope_embedding
from ..decoder.real import Decoder
from .common import QuantDecodersConfig
from .real_decoders_group import DecodersGroup


def make_marlin_decoders_from_files(
    config: QuantDecodersConfig,
    device: torch.device,
    state_files: list[str],
    tuning_params: Optional[MarlinTuningParams] = None,
    linear_cache: Optional[MarlinDenseCache] = None,
) -> DecodersGroup:
    if config.quant_config.bits != 4:
        raise ValueError("the Marlin decode backend requires 4-bit GPTQ weights")

    if tuning_params is None:
        tuning_params = MarlinTuningParams()
    if linear_cache is None:
        linear_cache = MarlinDenseCache(device)

    current_decoders: List[Decoder] = []
    holder = MarlinQuantLinearMaker(
        quantization_group_size=config.quant_config.group_size,
        use_act_order=config.quant_config.desc_act,
    )
    rotary_emb = make_rope_embedding(config.rope_config)
    with torch.device(device=device):
        for state_file in state_files:
            decoder = Decoder(
                config=config.decoder_config,
                linear_maker=holder,
                rotary_emb=rotary_emb,
            )
            with safe_open(state_file, framework="pt") as weights:
                with torch.device("cpu"):
                    raw_state_dict: Dict[str, torch.Tensor] = {
                        key: weights.get_tensor(key) for key in weights.keys()
                    }
                decoder.load_parameters(raw_state_dict)
            current_decoders.append(decoder)

    for decoder in current_decoders:
        decoder.half()
        decoder.input_layernorm.use_fused_cuda_ops = True
        decoder.post_attention_layernorm.use_fused_cuda_ops = True
        decoder.rotary_emb.use_fused_cuda_ops = True
        decoder.rotary_emb.position_ids_dtype = torch.int64
        decoder.mlp.use_fused_cuda_ops = True

    # Match nano-vLLM's decode topology: one QKV GEMM and one gate/up GEMM
    # per layer. The split checkpoint tensors can be concatenated directly in
    # the standard GPTQ layout before Marlin repacking.
    for decoder in current_decoders:
        decoder.self_attn.qkv_proj = holder.merge(
            [
                decoder.self_attn.q_proj,
                decoder.self_attn.k_proj,
                decoder.self_attn.v_proj,
            ]
        )
        decoder.self_attn.q_proj = None
        decoder.self_attn.k_proj = None
        decoder.self_attn.v_proj = None
        decoder.mlp.gate_up_proj = holder.merge(
            [decoder.mlp.gate_proj, decoder.mlp.up_proj]
        )
        decoder.mlp.gate_proj = None
        decoder.mlp.up_proj = None

    holder.post_init(tuning_params, linear_cache)
    linear_cache.init_cache()
    decoders = DecodersGroup(
        decoders=current_decoders,
        device=device,
        fuse_cache=None,
        attached_no_gc=(holder, linear_cache),
    )
    decoders.freeze()
    torch.cuda.empty_cache()
    return decoders
