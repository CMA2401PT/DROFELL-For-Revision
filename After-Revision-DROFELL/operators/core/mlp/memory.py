from ..linear.memory import LinearMakerMemoryModel
from ..memory import MemoryModule


class MLPMemoryModel(MemoryModule):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, linear_maker: LinearMakerMemoryModel):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = linear_maker.make_linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = linear_maker.make_linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = linear_maker.make_linear(self.intermediate_size, self.hidden_size, bias=False)
        # self.act_fn = SiLU()
