from cuda_extends.exllama import TuningParams
from ..rope.memory import RotaryEmbeddingMemoryModel
from ..rms_norm.memory import RMSNormMemoryModel
from ..mlp.memory import MLPMemoryModel
from ..attention.memory import AttentionDataMemoryModel
from ..linear.memory import CacheV2MemoryModel, LinearMakerMemoryModel
from .common import DecoderConfig, FuseConfig
from ..memory import MemoryModule


class DecoderWrapMemoryModel(MemoryModule):
    def __init__(self, config: DecoderConfig, linear_maker: LinearMakerMemoryModel, rotary_emb: RotaryEmbeddingMemoryModel):
        super().__init__()
        self.cfg = config
        self.self_attn = AttentionDataMemoryModel(config=config.attention_config, linear_maker=linear_maker)
        self.rotary_emb = rotary_emb
        self.mlp = MLPMemoryModel(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, linear_maker=linear_maker)
        self.input_layernorm = RMSNormMemoryModel(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormMemoryModel(config.hidden_size, eps=config.rms_norm_eps)


class FuseCacheMemoryModel(MemoryModule):
    def __init__(self, config: FuseConfig) -> None:
        super().__init__()


class DecoderWrapFuseMemoryModel(MemoryModule):
    def __init__(self, base: DecoderWrapMemoryModel, fuse_cache: FuseCacheMemoryModel, quant_linear_cache: CacheV2MemoryModel, tuning_params: TuningParams) -> None:
        super().__init__()
        self.fuse_input_layer_norm = base.input_layernorm.weight.data.half()
        self.fuse_input_norm_epslion = base.input_layernorm.variance_epsilon
        self.fuse_mlp_layer_norm = base.post_attention_layernorm.weight.data.half()
        self.fuse_mlp_norm_epslion = base.post_attention_layernorm.variance_epsilon
        self.fuse_cache = fuse_cache
        self.quant_linear_cache = quant_linear_cache
        self.tuning_params: TuningParams = tuning_params
        self.base = base
