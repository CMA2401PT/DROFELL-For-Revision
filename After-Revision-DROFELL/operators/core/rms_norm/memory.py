from ..memory import MemoryModule, MemoryTensor
from ..memory.static import MemoryDType, MemoryFloat32


class RMSNormMemoryModel(MemoryModule):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: MemoryDType = MemoryFloat32,
    ):
        super().__init__()
        self.weight = MemoryTensor((hidden_size,), dtype=dtype)
        self.variance_epsilon = eps
