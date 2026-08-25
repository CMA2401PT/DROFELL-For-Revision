import torch
from typing import Optional
from cuda_extends import vllm_attention as attention_ops
from operators import flags
from .common import AttentionConfig
from ..block.real import LayerBlocksHolder
from ..block.common import BlockInfo


# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


class PagedAttention:
    def __init__(self, config: AttentionConfig) -> None:
        self.cfg = config
        self.scale = self.cfg.head_dim ** -0.5
        self.head_mapping = torch.repeat_interleave(
            torch.arange(self.cfg.num_key_value_heads, dtype=torch.int32),
            self.cfg.num_key_value_groups
        )

    def forward(
        self,
        query: torch.Tensor,
        context_len_max: int,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        kv_cacher: LayerBlocksHolder,
        alibi_slopes: Optional[torch.Tensor] = None,
    ):
        """
        query=[bsz,num_query_head,head_dim]  
        context_lens=[seq_len]  
        block_tables=[seq_len,context_len_max]  
        out=[bsz,num_query_head,head_dim]  
        """
        output = torch.empty_like(query)
        if not kv_cacher.is_contigious:
            return self.ptr_table_paged_attention_forward(output=output, query=query, cacher=kv_cacher, context_len_max=context_len_max, context_lens=context_lens, block_tables=block_tables, alibi_slopes=alibi_slopes)
        else:
            k, v = kv_cacher.contigious_kv_tenosr
            return self.vllm_paged_attention_forward(output=output, query=query, block_info=kv_cacher.block_info, key_cache=k, val_cache=v, context_len_max=context_len_max, context_lens=context_lens, block_tables=block_tables, alibi_slopes=alibi_slopes)

    def ptr_table_paged_attention_forward(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        cacher: LayerBlocksHolder,
        context_len_max: int,  # max kv cache len
        context_lens: torch.Tensor,  # kv cache len of each seq
        block_tables: torch.Tensor,  # kv block table
        alibi_slopes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        query =[bsz,num_query_heads,head_dim]
        alibi_slopes: shape = [num_heads]
        """
        block_size = flags.BLOCK_SIZE
        num_seqs, num_heads, head_dim = query.shape
        assert self.cfg.num_query_heads == num_heads
        assert self.cfg.head_dim == head_dim

        max_num_partitions = ((context_len_max + _PARTITION_SIZE - 1) // _PARTITION_SIZE)
        use_v1 = context_len_max <= 8192 and (max_num_partitions == 1 or num_seqs * num_heads > 512)

        if use_v1:
            # Run PagedAttention V1.
            attention_ops.paged_attention_v1(
                output,
                query,
                cacher.layer_key_ptr_table_device,
                cacher.layer_val_ptr_table_device,
                self.head_mapping,
                self.scale,
                block_tables,
                context_lens,
                block_size,
                context_len_max,
                alibi_slopes,
                cacher.block_info.head_stride
            )
        else:
            # Run PagedAttention V2.
            assert _PARTITION_SIZE % block_size == 0
            assert _PARTITION_SIZE % block_size == 0
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_dim),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            attention_ops.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                cacher.layer_key_ptr_table_device,
                cacher.layer_val_ptr_table_device,
                self.head_mapping,
                self.scale,
                block_tables,
                context_lens,
                block_size,
                context_len_max,
                alibi_slopes,
                cacher.block_info.head_stride
            )

        return output

    def vllm_paged_attention_forward(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        block_info: BlockInfo,
        key_cache: torch.Tensor,
        val_cache: torch.Tensor,
        context_len_max: int,  # max kv cache len
        context_lens: torch.Tensor,  # kv cache len of each seq
        block_tables: torch.Tensor,  # kv block table
        alibi_slopes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        alibi_slopes: shape = [num_heads]
        key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
            block_size, x]
        value_cache: shape = [num_blocks, num_kv_heads, head_size,
            block_size]
        """
        block_size = flags.BLOCK_SIZE
        num_seqs, num_heads, head_dim = query.shape

        max_num_partitions = ((context_len_max + _PARTITION_SIZE - 1) // _PARTITION_SIZE)
        use_v1 = context_len_max <= 8192 and (max_num_partitions == 1 or num_seqs * num_heads > 512)

        if use_v1:
            # Run PagedAttention V1.
            attention_ops.paged_attention_v1_orig(
                output,
                query,
                key_cache,
                val_cache,
                self.head_mapping,
                self.scale,
                block_tables,
                context_lens,
                block_size,
                context_len_max,
                alibi_slopes,
            )
        else:
            # Run PagedAttention V2.
            assert _PARTITION_SIZE % block_size == 0
            assert _PARTITION_SIZE % block_size == 0
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_dim),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            attention_ops.paged_attention_v2_orig(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                val_cache,
                self.head_mapping,
                self.scale,
                block_tables,
                context_lens,
                block_size,
                context_len_max,
                alibi_slopes
            )
        return output
