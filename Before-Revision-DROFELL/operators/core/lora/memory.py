from typing import Dict, Optional, Tuple

from .render import LoraRenderConfig
from ..memory.env import Env, MemoryDict, MemoryHolder
from ...assemble.fwd_ctx.trainable import TrainableUserMemoryModel
from .common import LoraConfig
from ..memory.static import MemoryDType, MemoryTensor


class LoraMemoryModel(TrainableUserMemoryModel, MemoryDict):
    def __init__(self, config: LoraConfig, env: Env):
        TrainableUserMemoryModel.__init__(self)
        MemoryDict.__init__(self, env)
        self.cfg = config
        self.lora_block = MemoryTensor(size=(self.cfg.num_layers, *self.cfg.lora_block_shape))

    @property
    def trainable_parameters(self) -> Dict[str, MemoryTensor]:
        return {"lora_block": self.lora_block}

    def ensure_backward(self, env: Env):
        if "grad" not in self.childrens:
            self.set("grad", env.trace(MemoryTensor.like(self.lora_block)))

    def train_backward(self, env: Env, ctx: MemoryDict, grad: MemoryHolder[MemoryTensor]) -> MemoryHolder[MemoryTensor]:
        self.ensure_backward(env)
        return env.trace(MemoryTensor(grad.item.size, grad.item.dtype, grad.item.is_cpu))

    @property
    def mode(self) -> Optional[LoraRenderConfig]:
        # if is ptuning mode, return None
        # is is lora mode, return lora config
        return self.cfg.render_config

    def train_forward(self, env: Env, dtype: MemoryDType) -> Tuple[MemoryDict, list[MemoryHolder[MemoryTensor]]]:
        per_layer_input: list[MemoryHolder[MemoryTensor]] = [
            MemoryHolder(MemoryTensor(self.cfg.lora_block_shape, dtype=dtype), env)
            for _ in range(self.cfg.num_layers)
        ]
        return MemoryDict(env), per_layer_input
