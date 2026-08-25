from ..memory import MemoryModule, MemoryTensor


class RMSNormMemoryModel(MemoryModule):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = MemoryTensor((hidden_size,))
        self.variance_epsilon = eps
