from dataclasses import dataclass


@dataclass
class QuantizationConfig:
    bits: int
    group_size: int
    desc_act: bool
    quant_method: str

    def __post_init__(self):
        assert self.quant_method == "gptq"
        assert self.bits == 4
