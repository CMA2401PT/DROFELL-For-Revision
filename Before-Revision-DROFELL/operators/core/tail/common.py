from dataclasses import dataclass


@dataclass
class ModelTailConfig:
    vocab_size: int
    hidden_size: int
    rms_norm_eps: float
