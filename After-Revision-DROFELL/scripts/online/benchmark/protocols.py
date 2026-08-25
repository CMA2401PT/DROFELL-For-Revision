from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DecodeBenchmarkConfig:
    model_name: str
    prompt: str
    num_seqs: tuple[int, ...]
    runs: int
    warmup_requests: int
    warmup_decode_tokens: int
    decode_tokens: int
    max_model_len: int
    expected_dealers: Optional[int]
