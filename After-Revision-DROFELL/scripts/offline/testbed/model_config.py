from dataclasses import dataclass
import copy
from typing import Callable, List

from operators.helpers.model_lib import ModelsLib
from operators.assemble.decoders.real import DecodersImplement
import split_qllm_models


@dataclass
class ModelConfig:
    model_name: str = "teared-Llama-2-7B-GPTQ"
    decoder_backend: DecodersImplement = "marlin"
    enable_cuda_graph: bool = False
    max_seq_len: int = -1
    max_layers_to_load: int = -1
    layers: int = 0
    group_fn: Callable[[int], List[int]] = lambda layers: [
        min(8, layers-start) for start in range(0, layers, 8)
    ]

    def __post_init__(self):
        # process model config
        models = ModelsLib(split_qllm_models.lib_path)
        self.model_info = models.models_storage[self.model_name]
        self.base_model_config = copy.copy(self.model_info.config)
        if self.max_seq_len == -1:
            self.max_seq_len = self.base_model_config.max_seq_len
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.block_info = self.base_model_config.block_info
        self.base_model_config.max_seq_len = self.max_seq_len
        if self.max_layers_to_load > 0:
            assert self.max_layers_to_load <= self.base_model_config.num_decoders
            self.base_model_config.num_decoders = self.max_layers_to_load
            self.layers = self.base_model_config.num_decoders
            print(f"load only {self.layers} layers")
        else:
            self.layers = self.base_model_config.num_decoders
        self.tokenizer_path = self.model_info.tokenizer_path
