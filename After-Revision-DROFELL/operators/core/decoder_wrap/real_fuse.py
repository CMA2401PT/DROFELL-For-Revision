from typing import Callable, Optional
import torch
from .common import LoraHook

from .real_common import DecoderWrap
from cuda_extends.exllama import CacheV2, TuningParams, mlp_forward_v2_py, attention_forward_qkv_stage_v2_py, attention_forward_rope_stage_v2_py
from .real_cache import FuseCache


class DecoderWrapFuse:
    def __init__(self, base: DecoderWrap, fuse_cache: FuseCache, quant_linear_cache: CacheV2, tuning_params: TuningParams) -> None:
        self.fuse_input_layer_norm = base.input_layernorm.weight.data.half()
        self.fuse_input_norm_epslion = base.input_layernorm.variance_epsilon
        self.fuse_mlp_layer_norm = base.post_attention_layernorm.weight.data.half()
        self.fuse_mlp_norm_epslion = base.post_attention_layernorm.variance_epsilon
        self.fuse_cache: FuseCache = fuse_cache
        self.quant_linear_cache: CacheV2 = quant_linear_cache
        self.tuning_params: TuningParams = tuning_params
        self.device = self.fuse_input_layer_norm.device

        self.base = base
        self.q_q4 = self.base.self_attn.q_proj.q4_matrix.cuda_q4
        self.k_q4 = self.base.self_attn.k_proj.q4_matrix.cuda_q4
        self.v_q4 = self.base.self_attn.v_proj.q4_matrix.cuda_q4

        self.gate_q4 = self.base.mlp.gate_proj.q4_matrix.cuda_q4
        self.up_q4 = self.base.mlp.up_proj.q4_matrix.cuda_q4
        self.down_q4 = self.base.mlp.down_proj.q4_matrix.cuda_q4

    def _fuse_forward_before_stage(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        lora_hook: Optional[LoraHook],
    ):
        bsz, q_len, _ = hidden_states.size()
        assert q_len == 1
        residual = hidden_states

        temp_rms_output = self.fuse_cache.temp_rms_output  # hidden state
        temp_lora_output = self.fuse_cache.temp_lora_output
        temp_q_output = self.fuse_cache.temp_q_output
        temp_k_output = self.fuse_cache.temp_k_output
        temp_v_output = self.fuse_cache.temp_v_output

        cfg = self.fuse_cache.cfg

        attention_forward_qkv_stage_v2_py(
            self.fuse_input_layer_norm,
            self.fuse_input_norm_epslion,
            self.q_q4,
            self.k_q4,
            self.v_q4,
            temp_rms_output,
            temp_q_output,
            temp_k_output,
            temp_v_output,
            self.quant_linear_cache.temp_weight,
            hidden_states,
            self.tuning_params.BLOCK_KN_SIZE,
            self.tuning_params.BLOCK_M_SIZE_MAX,
            self.tuning_params.MATMAL_RECONSTRUCT_THRESHOLD_V2,
            q_len, bsz,
        )
        # hidden_state(temp_rms_output),q,k,v shape=[batch_size,hidden_state]
        if lora_hook:
            # with torch.autograd.profiler.record_function("Lora"):
            lora_hook(temp_rms_output, temp_q_output, temp_k_output, temp_v_output, temp_lora_output)
        attention_forward_rope_stage_v2_py(
            temp_q_output,
            temp_k_output,
            self.base.rotary_emb.sin_cached,
            self.base.rotary_emb.cos_cached,
            q_len, bsz,
            cfg.num_query_heads, cfg.num_key_value_heads, cfg.head_dim,
            -1, position_ids

        )
        query_states, key_states, value_states = self.fuse_cache.get_qkv_with_std_view()
        return query_states, key_states, value_states, residual

    def _fuse_forward_after_stage(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        hidden_states = hidden_states.view_as(residual)
        hidden_states = self.base.self_attn.o_proj(hidden_states)

        hidden_states = residual + hidden_states
        residual = hidden_states
        target_shape = hidden_states.shape
        output = torch.empty_like(hidden_states)
        mlp_forward_v2_py(
            self.fuse_mlp_layer_norm,
            self.fuse_mlp_norm_epslion,
            self.gate_q4,
            self.up_q4,
            self.down_q4,
            self.fuse_cache.temp_rms_output,
            self.fuse_cache.temp_gate_output,
            self.fuse_cache.temp_up_output,
            self.quant_linear_cache.temp_weight,
            hidden_states.view(-1, target_shape[-1]),
            output.view(-1, target_shape[-1]),
            self.tuning_params.BLOCK_KN_SIZE,
            self.tuning_params.BLOCK_M_SIZE_MAX,
            self.tuning_params.MATMAL_RECONSTRUCT_THRESHOLD_V2
        )
        hidden_states = output.view(target_shape)
        hidden_states = residual + hidden_states
        return hidden_states

    def fuse_forward_without_attention(
        self,
        hidden_states: torch.Tensor, position_ids: torch.Tensor,
        atten: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        lora_hook: Optional[LoraHook],
    ):
        """
        hidden_states=[bsz,1,hidden_size]  
        atten(q,k,v)->o  
        q,k,v,o = [flatten_seq_len/seq_len,num_heads,head_dim]  
        out=[bsz,1,hidden_size]  
        """
        query_states, key_states, value_states, residual = self._fuse_forward_before_stage(hidden_states, position_ids, lora_hook)
        out = atten(query_states, key_states, value_states)
        return self._fuse_forward_after_stage(out, residual)
