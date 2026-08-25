from dataclasses import dataclass
from typing import Callable, List

from operators.helpers.model_lib import ModelsLib
import split_qllm_models


@dataclass
class ModelConfig:
    model_name: str = "teared-Llama-2-7B-GPTQ"
    max_layers_to_load: int = -1
    max_allowed_len: int = 128*1024
    layers: int = 0
    group_fn: Callable[[int], List[int]] = lambda layers: [8 for _ in range(layers//8)]

    def __post_init__(self):
        # process model config
        models = ModelsLib(split_qllm_models.lib_path)
        self.model_info = models.models_storage[self.model_name]
        self.base_model_config = self.model_info.config
        self.block_info = self.base_model_config.block_info
        self.base_model_config.max_seq_len = self.max_allowed_len
        if self.max_layers_to_load > 0:
            assert self.max_layers_to_load <= self.base_model_config.num_decoders
            self.base_model_config.num_decoders = self.max_layers_to_load
            self.layers = self.base_model_config.num_decoders
            print(f"load only {self.layers} layers")
        else:
            self.layers = self.base_model_config.num_decoders
        self.tokenizer_path = self.model_info.tokenizer_path
