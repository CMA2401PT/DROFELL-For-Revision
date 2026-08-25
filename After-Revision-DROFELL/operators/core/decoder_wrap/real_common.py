from typing import Optional, Tuple
from torch import nn
import torch

from ...container import Container

from ...core.attention.real_attn import AttentionData
from ...core.rms_norm.real import RMSNorm
from ...core.mlp.real import MLP
from ...core.rope.real import RotaryEmbedding
from ...core.linear.real import LinearMaker
from .common import DecoderConfig, LoraHook


class DecoderWrap(nn.Module):
    def __init__(self, config: DecoderConfig, linear_maker: LinearMaker, rotary_emb: RotaryEmbedding) -> None:
        nn.Module.__init__(self)
        self.cfg = config
        self.self_attn = AttentionData(config=config.attention_config, linear_maker=linear_maker)
        self.rotary_emb = rotary_emb
        self.mlp = MLP(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, linear_maker=linear_maker)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.device = torch.zeros((0,)).device

    async def finetuning_forward_before_stage(
        self,
        container: Container,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        lora_hook: Optional[LoraHook],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        num_tokens = hidden_states.size(0)
        hidden_states = await container.step("input_layernorm.forward", self.input_layernorm.forward, hidden_states=hidden_states)
        # Attention

        def qkv(hidden_states: torch.Tensor):
            qkv_proj = getattr(self.self_attn, "qkv_proj", None)
            if qkv_proj is None:
                query_states = self.self_attn.q_proj(hidden_states)
                key_states = self.self_attn.k_proj(hidden_states)
                value_states = self.self_attn.v_proj(hidden_states)
            else:
                query_states, key_states, value_states = qkv_proj(
                    hidden_states
                ).split(
                    (
                        self.cfg.num_query_heads * self.cfg.head_dim,
                        self.cfg.num_key_value_heads * self.cfg.head_dim,
                        self.cfg.num_key_value_heads * self.cfg.head_dim,
                    ),
                    dim=-1,
                )
            if lora_hook:
                hidden_states = hidden_states.view(num_tokens, self.cfg.num_query_heads*self.cfg.head_dim)
                lq, lk, lv = lora_hook(hidden_states, query_states, key_states, value_states, None)
                if lq is not None:
                    query_states = lq
                if lk is not None:
                    key_states = lk
                if lv is not None:
                    value_states = lv
            query_states = query_states.view(num_tokens, self.cfg.num_query_heads, self.cfg.head_dim)
            key_states = key_states.view(num_tokens, self.cfg.num_key_value_heads, self.cfg.head_dim)
            value_states = value_states.view(num_tokens, self.cfg.num_key_value_heads, self.cfg.head_dim)
            return query_states, key_states, value_states
        query_states, key_states, value_states = await container.step("qkv", qkv, hidden_states=hidden_states)
        query_states, key_states = await container.step("rotary_emb.forward", self.rotary_emb.forward, q=query_states, k=key_states, position_ids=position_ids)
        return query_states, key_states, value_states, residual

    async def finetuning_forward_after_stage(self, container: Container,  hidden_states: torch.Tensor, residual: torch.Tensor):
        hidden_states = hidden_states.view_as(residual)

        def o_proj(hidden_states: torch.Tensor, residual: torch.Tensor):
            hidden_states = self.self_attn.o_proj(hidden_states)
            # Attention end
            hidden_states = residual + hidden_states
            # Fully Connected
            residual = hidden_states
            return hidden_states, residual
        hidden_states, residual = await container.step("o_proj", o_proj, hidden_states=hidden_states, residual=residual)
        hidden_states = await container.step("post_attention_layernorm", self.post_attention_layernorm.forward, hidden_states=hidden_states)

        def mlp(hidden_states: torch.Tensor, residual: torch.Tensor):
            hidden_states = self.mlp.forward(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states
        hidden_states = await container.step("mlp", mlp, hidden_states=hidden_states, residual=residual)
        return hidden_states

    def inference_forward_before_stage(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        lora_hook: Optional[LoraHook],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        num_tokens = hidden_states.size(0)
        hidden_states = self.input_layernorm.forward(hidden_states)
        # Attention
        qkv_proj = getattr(self.self_attn, "qkv_proj", None)
        if qkv_proj is None:
            query_states = self.self_attn.q_proj(hidden_states)
            key_states = self.self_attn.k_proj(hidden_states)
            value_states = self.self_attn.v_proj(hidden_states)
        else:
            query_states, key_states, value_states = qkv_proj(
                hidden_states
            ).split(
                (
                    self.cfg.num_query_heads * self.cfg.head_dim,
                    self.cfg.num_key_value_heads * self.cfg.head_dim,
                    self.cfg.num_key_value_heads * self.cfg.head_dim,
                ),
                dim=-1,
            )

        if lora_hook is not None:
            hidden_states = hidden_states.view(num_tokens, self.cfg.num_query_heads*self.cfg.head_dim)
            lq, lk, lv = lora_hook(hidden_states, query_states, key_states, value_states, None)
            if lq is not None:
                query_states = lq
            if lk is not None:
                key_states = lk
            if lv is not None:
                value_states = lv

        query_states = query_states.view(num_tokens, self.cfg.num_query_heads, self.cfg.head_dim)
        key_states = key_states.view(num_tokens, self.cfg.num_key_value_heads, self.cfg.head_dim)
        value_states = value_states.view(num_tokens, self.cfg.num_key_value_heads, self.cfg.head_dim)
        query_states, key_states = self.rotary_emb.forward(q=query_states, k=key_states, position_ids=position_ids)
        return query_states, key_states, value_states, residual

    def inference_forward_after_stage(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        hidden_states = hidden_states.view_as(residual)
        hidden_states = self.self_attn.o_proj(hidden_states)
        # Attention end
        hidden_states, residual = self.post_attention_layernorm.forward_add(
            hidden_states,
            residual,
        )
        # Fully Connected
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
