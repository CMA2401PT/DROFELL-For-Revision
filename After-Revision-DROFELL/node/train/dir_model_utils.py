
from typing import List

from dealers.common.base_types import ModelDescribe


class DirModel:
    def __init__(self, models: List[str]) -> None:
        self._models = models
        self._dir_models: List[str] = []
        self._has_tail: bool = False
        for m in models:
            desc = ModelDescribe.from_str(m)
            if desc.is_decoders:
                self._dir_models.append(m+",forward")
            else:
                self._dir_models.append(m)
            self._has_tail = desc.is_tail or self._has_tail
        self._forward_models_count = len(models)
        self._backward_start = len(self._dir_models)
        self._backward_models_count = 0
        for m in models[::-1]:
            desc = ModelDescribe.from_str(m)
            if desc.is_decoders:
                self._dir_models.append(m+",backward")
                self._backward_models_count += 1

    @property
    def models(self):
        return self._models

    @property
    def dir_models(self):
        return self._dir_models

    @property
    def has_tail(self):
        return self._has_tail

    @property
    def forward_only_models_count(self):
        return self._forward_models_count

    @property
    def backward_only_models_count(self):
        return self._backward_models_count

    @property
    def model_seq_count(self):
        if self._has_tail:
            return self._forward_models_count+self._backward_models_count, None
        return self._forward_models_count, self._backward_models_count

    def __getitem__(self, dir_model_i: int):
        return self._dir_models[dir_model_i]

    def __len__(self):
        return len(self._dir_models)

    def index_dir(self, dir_model: str):
        return self._dir_models.index(dir_model)
