from ..memory.static import MemoryDType, MemoryFloat32, MemoryTensor
from ..memory.env import Env, MemoryHolder
from ..memory.module import LinearMemoryModel
from ..rms_norm.memory import RMSNormMemoryModel
from ..memory import MemoryModule
from .common import ModelTailConfig


class ModelTailMemoryModel(MemoryModule):
    def __init__(
        self,
        config: ModelTailConfig,
        dtype: MemoryDType = MemoryFloat32,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.norm = RMSNormMemoryModel(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.lm_head = LinearMemoryModel(
            config.hidden_size, config.vocab_size, bias=False, dtype=dtype
        )

    def compute_loss_and_grad(
        self,
        hidden_states: MemoryHolder[MemoryTensor],
        samples_label_ids: MemoryHolder[MemoryTensor],
        env: Env,
    ):
        return env.trace(MemoryTensor(hidden_states.item.size, hidden_states.item.dtype, hidden_states.item.is_cpu))
