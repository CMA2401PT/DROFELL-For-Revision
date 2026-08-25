from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any, Mapping, Optional

from operators import flags


DECODER_BACKENDS = frozenset({"marlin", "exllama", "exllama_fuse"})
API_DECODE_GRAPH_STATS = "telemetric-decode-graph"


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _env_positive_int(
    env: Mapping[str, str], name: str, default: int
) -> int:
    value = int(env.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class OnlineInferenceConfig:
    decoder_backend: str = "marlin"
    enable_cuda_graph: bool = False
    max_seq_len: int = 256
    max_concurrency: int = 8

    def __post_init__(self) -> None:
        if self.decoder_backend not in DECODER_BACKENDS:
            raise ValueError(
                "decoder_backend must be marlin, exllama, or exllama_fuse, "
                f"got {self.decoder_backend!r}"
            )
        if self.max_seq_len <= 0:
            raise ValueError("graph_max_context_len must be positive")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        if self.enable_cuda_graph and self.decoder_backend != "marlin":
            raise ValueError("online CUDA graph currently requires Marlin")

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None
    ) -> "OnlineInferenceConfig":
        source = os.environ if env is None else env
        backend = source.get("DECODER_BACKEND", "marlin").strip().lower()
        return cls(
            decoder_backend=backend,
            enable_cuda_graph=_env_bool(
                source, "ENABLE_CUDA_GRAPH", False
            ),
            max_seq_len=_env_positive_int(
                source, "CUDA_GRAPH_MAX_CONTEXT_LEN", flags.MAX_TOKEN_LEN_OVERRIDE
            ),
            max_concurrency=_env_positive_int(
                source, "MAX_CONCURRENCY", 8
            ),
        )

    def as_dict(self) -> Mapping[str, Any]:
        return asdict(self)
