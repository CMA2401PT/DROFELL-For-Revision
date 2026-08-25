from dataclasses import dataclass


@dataclass
class ModelHeadConfig:
    vocab_size: int
    hidden_size: int
    pad_token_id: int
