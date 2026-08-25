
from typing import Dict, List, Optional, Tuple
import torch

from operators import flags
from ..memory.env import Env, MemoryDict, MemoryHolder
from ..lora.render import LoraRenderConfig

from ...assemble.fwd_ctx.trainable import TrainableUserMemoryModel
from .common import PtuningConfig
from ..memory.static import MemoryDType, MemoryTensor


class PtuningMemoryModel(TrainableUserMemoryModel, MemoryDict):
    def __init__(self, config: PtuningConfig, env: Env):
        TrainableUserMemoryModel.__init__(self)
        MemoryDict.__init__(self, env)
        self.cfg = config
        self.embedding = MemoryTensor(size=(self.cfg.prefix_len, self.cfg.combined_output_shape))
        self.output_shape = (1, self.cfg.prefix_len, self.cfg.combined_output_shape)
        self.prefix_tokens = MemoryTensor((1, self.cfg.prefix_len), torch.long)

    @property
    def trainable_parameters(self) -> Dict[str, MemoryTensor]:
        return {"embedding": self.embedding}

    def train_forward(self, env: Env, dtype: MemoryDType) -> Tuple[MemoryDict, List[MemoryHolder[MemoryTensor]]]:
        if flags.USER_MODEL_ENABLE_CHECKPOTIN:
            return self._train_forward_no_grad(env, dtype)
        else:
            return self._train_forward_with_grad(env, dtype)

    def ensure_backward(self, env: Env):
        if "grad" not in self.childrens:
            self.set("grad", env.trace(MemoryTensor.like(self.embedding)))

    def train_backward(self, env: Env, ctx: MemoryDict, grad: MemoryHolder[MemoryTensor]) -> MemoryHolder[MemoryTensor]:
        self.ensure_backward(env)
        if "intermediate-emb-out" in self.childrens:
            self.remove("intermediate-emb-out")
        return env.trace(MemoryTensor(grad.item.size, grad.item.dtype, grad.item.is_cpu))

    @property
    def mode(self) -> Optional[LoraRenderConfig]:
        # if is ptuning mode, return None
        # is is lora mode, return lora config
        return None

    def _train_forward_no_grad(self, env: Env, dtype: MemoryDType) -> Tuple[MemoryDict, list[MemoryHolder[MemoryTensor]]]:
        per_layer_input: list[MemoryHolder[MemoryTensor]] = [
            MemoryHolder(MemoryTensor((2, self.cfg.prefix_len, self.cfg.num_key_value_heads, self.cfg.head_dim), dtype=dtype), env)
            for _ in range(self.cfg.num_layers)
        ]
        return MemoryDict(env), per_layer_input

    def _train_forward_with_grad(self, env: Env, dtype: MemoryDType) -> Tuple[MemoryDict, list[MemoryHolder[MemoryTensor]]]:
        per_layer_input: list[MemoryHolder[MemoryTensor]] = [
            MemoryHolder(MemoryTensor((2, self.cfg.prefix_len, self.cfg.num_key_value_heads, self.cfg.head_dim), dtype=dtype), env)
            for _ in range(self.cfg.num_layers)
        ]
        self.set("intermediate-emb-out", env.trace(MemoryTensor.like(self.embedding, dtype=torch.bool)))
        return MemoryDict(env), per_layer_input
