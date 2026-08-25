from dataclasses import dataclass
import json
import os
from typing import Dict
from ..configs import LlamaConfig


@dataclass
class ModelInfo:
    base_model_path: str
    config: LlamaConfig

    @property
    def model_config_path(self):
        return os.path.join(self.base_model_path, "config.json")

    @property
    def has_model_config_file(self):
        return os.path.exists(self.model_config_path)

    @property
    def tokenizer_path(self):
        return os.path.join(self.base_model_path, "tokenizer")

    @property
    def model_head_path(self):
        return os.path.join(self.base_model_path, "split_model")

    @property
    def model_tail_path(self):
        return os.path.join(self.base_model_path, "split_model")

    def decoder_path(self, layer_i: int):
        return os.path.join(self.base_model_path, "split_model", f"layer_{layer_i}.safetensors")


class ModelsLib:
    def __init__(self, lib_path: str) -> None:
        self.models_dir = lib_path
        self.models_storage: Dict[str, ModelInfo] = {}
        self._load_all()

    def _load_all(self):
        for model_name in os.listdir(self.models_dir):
            base_model_dir = os.path.join(self.models_dir, model_name)
            config_file = os.path.join(base_model_dir, "config.json")
            if not os.path.exists(config_file):
                continue
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            config = LlamaConfig(**config_dict)
            self.models_storage[model_name] = ModelInfo(base_model_dir, config)
