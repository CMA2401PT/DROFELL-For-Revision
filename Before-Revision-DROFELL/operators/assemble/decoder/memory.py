from typing import Optional

from cuda_extends.exllama import TuningParams
from ...core.memory.static import MemoryTensor
from ...core.memory.env import Env, MemoryHolder
from ...core.attention.memory import PagedAttentionMemoryModel
from ...core.rope.memory import RotaryEmbeddingMemoryModel
from ...core.decoder_wrap.memory import CacheV2MemoryModel, DecoderWrapFuseMemoryModel, DecoderWrapMemoryModel, FuseCacheMemoryModel
from ...core.linear.memory import LinearMakerMemoryModel
from ...core.decoder_wrap.common import DecoderConfig
from ...core.decoder_wrap.memory import DecoderWrapMemoryModel


class DecoderMemoryModel(DecoderWrapMemoryModel):
    def __init__(self, config: DecoderConfig, linear_maker: LinearMakerMemoryModel, rotary_emb: RotaryEmbeddingMemoryModel):
        DecoderWrapMemoryModel.__init__(self=self, config=config, linear_maker=linear_maker, rotary_emb=rotary_emb)
        self.fuse_wrap: Optional[DecoderWrapFuseMemoryModel] = None
        self.paged_attention = PagedAttentionMemoryModel(config=config.attention_config)
        self.config = config

    def fuse(self, fuse_cache: FuseCacheMemoryModel, quant_linear_cache: CacheV2MemoryModel, tuning_params: TuningParams):
        self.fuse_wrap = DecoderWrapFuseMemoryModel(self, fuse_cache=fuse_cache, quant_linear_cache=quant_linear_cache, tuning_params=tuning_params)

    def multi_seqs_finetuning_forward(self, env: Env, hidden_states: MemoryHolder[MemoryTensor], position_ids: MemoryHolder[MemoryTensor], input_lens: list[int]):
        t = hidden_states.item
        assert not t.is_cpu
        assert not position_ids.item.is_cpu
        return env.trace(MemoryTensor(t.size, t.dtype, False))
