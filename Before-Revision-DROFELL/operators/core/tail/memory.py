from ..memory.static import MemoryTensor
from ..memory.env import Env, MemoryHolder
from ..memory.module import LinearMemoryModel
from ..rms_norm.memory import RMSNormMemoryModel
from ..memory import MemoryModule
from .common import ModelTailConfig


class ModelTailMemoryModel(MemoryModule):
    def __init__(self, config: ModelTailConfig) -> None:
        super().__init__()
        self.cfg = config
        self.norm = RMSNormMemoryModel(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = LinearMemoryModel(config.hidden_size, config.vocab_size, bias=False)

    def compute_loss_and_grad(
        self,
        hidden_states: MemoryHolder[MemoryTensor],
        samples_label_ids: MemoryHolder[MemoryTensor],
        env: Env,
    ):
        return env.trace(MemoryTensor(hidden_states.item.size, hidden_states.item.dtype, hidden_states.item.is_cpu))
