import torch
from flash_attn import flash_attn_func  # type: ignore
from .common import AttentionConfig


class FlashAttention:
    def __init__(self, config: AttentionConfig):
        self.cfg = config

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        multi_seq: bool,
    ) -> torch.Tensor:
        """
        multi_seq=True:
            query_states: [bsz,query_seq_len,num_query_heads,head_dim]
            key_states: [bsz,context_len,num_kv_heads,head_dim]
            value_states: [bsz,context_len,num_kv_heads,head_dim]
            out: [bsz,query_seq_len,num_query_heads,head_dim]
        seq:
            query_states: [query_seq_len,num_query_heads,head_dim]
            key_states: [context_len,num_kv_heads,head_dim]
            value_states: [context_len,num_kv_heads,head_dim]
            out: [query_seq_len,num_query_heads,head_dim]
        """
        if not multi_seq:
            query_states = query_states.unsqueeze(0)
            key_states = key_states.unsqueeze(0)
            value_states = value_states.unsqueeze(0)
        _, _, num_query_heads, head_dim = query_states.shape
        assert num_query_heads == self.cfg.num_query_heads
        assert head_dim == self.cfg.head_dim
        assert len(key_states.shape) == 4
        assert len(value_states.shape) == 4
        attn_output: torch.Tensor = flash_attn_func(q=query_states, k=key_states, v=value_states, causal=True)  # type: ignore
        if multi_seq:
            return attn_output
        else:
            return attn_output.squeeze(0)

    # def decoding_forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     position_ids: torch.Tensor,
    #     past_key_values: Tuple[torch.Tensor, torch.Tensor],
    #     past_key_value_lens: Union[torch.Tensor, int]
    # ) -> torch.Tensor:
    #     bsz, q_len, _ = hidden_states.size()

    #     query_states = self.q_proj(hidden_states)
    #     key_states = self.k_proj(hidden_states)
    #     value_states = self.v_proj(hidden_states)

    #     query_states = query_states.view(bsz, q_len, self.num_query_heads, self.head_dim)
    #     key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    #     value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    #     # don't use past_key_value_lens/fused flash decoding here because in ptuning, position_ids start after perfix
    #     query_states, key_states = self.rotary_emb(query_states, key_states,  position_ids, head_dim=2)
    #     attn_output = flash_attn_with_kvcache(q=query_states,
    #                                           k_cache=past_key_values[0],
    #                                           v_cache=past_key_values[1],
    #                                           k=key_states,
    #                                           v=value_states,
    #                                           cache_seqlens=past_key_value_lens,
    #                                           causal=True
    #                                           )

    #     # attn_output = attn_output.transpose(1, 2).contiguous()
    #     attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    #     attn_output = self.o_proj(attn_output)
    #     return attn_output
