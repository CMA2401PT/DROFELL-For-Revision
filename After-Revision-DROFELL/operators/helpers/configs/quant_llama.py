# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" LLaMA model configuration"""
from typing import Dict, Optional, Any

from ... import flags

from ...core.block.common import BlockInfo
from ...core.tail.common import ModelTailConfig

from ...core.head.common import ModelHeadConfig
from ...core.decoder_wrap.common import DecoderConfig
from ...core.rope.common import RopeConfig
from ...core.tokenizer.common import TokenizerSpecialTokens
from ...core.linear.common import QuantizationConfig
from ...assemble.decoders.common import QuantDecodersConfig


class LlamaConfig:
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        pretraining_tp (`int`, *optional*, defaults to `1`):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be an float greater than 1. The expected format
            is `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
    ```"""

    def __init__(self,
                 vocab_size: int = 32000,
                 hidden_size: int = 4096,
                 intermediate_size: int = 11008,
                 num_hidden_layers: int = 32,
                 num_attention_heads: int = 32,
                 num_key_value_heads: Optional[int] = None,
                 hidden_act: str = "silu",
                 max_position_embeddings: int = 2048,
                 initializer_range: float = 0.02,
                 rms_norm_eps: float = 1e-6,
                 use_cache: bool = True,
                 pad_token_id: int = 0,
                 bos_token_id: int = 1,
                 eos_token_id: int = 2,
                 pretraining_tp: int = 1,
                 tie_word_embeddings: bool = False,
                 rope_scaling: Optional[Dict[str, Any]] = None,
                 rope_theta: int = 10000,
                 quantization_config: Optional[Dict[str, Any]] = None,
                 **kwargs: Any):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_decoders = num_hidden_layers

        self.num_query_heads = num_attention_heads
        if flags.MAX_TOKEN_LEN_OVERRIDE != 0:
            max_position_embeddings = flags.MAX_TOKEN_LEN_OVERRIDE
        self.max_seq_len = max_position_embeddings
        if num_key_value_heads is None:
            self.num_key_value_heads = num_attention_heads
        else:
            self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_query_heads//self.num_key_value_heads
        self.head_dim = hidden_size // self.num_query_heads
        assert pretraining_tp == 1

        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.quantization_dict_config = quantization_config
        self.tie_word_embeddings = tie_word_embeddings

    @property
    def decoder_config(self):
        return DecoderConfig(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            num_key_value_groups=self.num_key_value_groups,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            rms_norm_eps=self.rms_norm_eps
        )

    @property
    def rope_config(self):
        return RopeConfig(
            rope_theta=self.rope_theta,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            rope_scaling_type=None if self.rope_scaling is None else self.rope_scaling["type"],
            rope_scaling_factor=None if self.rope_scaling is None else self.rope_scaling["factor"],
        )

    @property
    def tokenizer_special_tokens(self):
        return TokenizerSpecialTokens(
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id
        )

    @property
    def quantization_config(self):
        assert self.quantization_dict_config is not None
        return QuantizationConfig(
            bits=self.quantization_dict_config["bits"],
            group_size=self.quantization_dict_config["group_size"],
            desc_act=self.quantization_dict_config["desc_act"],
            quant_method=self.quantization_dict_config["quant_method"],
        )

    @property
    def model_head_config(self):
        return ModelHeadConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            pad_token_id=self.pad_token_id
        )

    @property
    def quant_decoder_config(self):
        return QuantDecodersConfig(
            decoder_config=self.decoder_config,
            rope_config=self.rope_config,
            quant_config=self.quantization_config,
        )

    @property
    def model_tail_config(self):
        return ModelTailConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            rms_norm_eps=self.rms_norm_eps
        )

    @property
    def block_info(self):
        return BlockInfo(
            head_dim=self.decoder_config.head_dim,
            num_key_value_heads=self.decoder_config.num_key_value_heads,
        )
