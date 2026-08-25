"""Validated logical-network specification for online benchmarks.

The specification deliberately excludes physical placement.  Host, process,
thread and CUDA-device bindings are supplied to ``launch_network.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from dealers.common.base_types import group_model_with_split
from dealers.memory_schedule.restriction import (
    MemoryConfig,
    MemoryConfigPercentage,
    MemoryConfigStatic,
)
from node.aux_node.protocols import ExtraInfo, ExtraModelInfo
from node.dealer_node.dealer_node import ModelTargets
from operators.helpers.model_lib.common import ModelsLib
from scripts.online.runtime_config import DECODER_BACKENDS, OnlineInferenceConfig
import split_qllm_models


MIB = 2**20
SCHEMA_VERSION = 1


def _require_exact_keys(
    value: Mapping[str, Any],
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValueError(f"{context} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {sorted(unknown)}")


@dataclass(frozen=True)
class MemorySpec:
    policy: str = "static"
    major_reserved_mib: int = 8192
    major_percentage: float = 0.8
    unmanaged_reserved_mib: int = 32
    minor_min_mib: int = 0
    minor_max_mib: int = 4096
    operate_reserved_mib: int = 1024
    graph_budget_mib: int = 0

    def __post_init__(self) -> None:
        if self.policy not in {"static", "percentage"}:
            raise ValueError("memory policy must be static or percentage")
        integer_values = {
            "major_reserved_mib": self.major_reserved_mib,
            "unmanaged_reserved_mib": self.unmanaged_reserved_mib,
            "minor_min_mib": self.minor_min_mib,
            "minor_max_mib": self.minor_max_mib,
            "operate_reserved_mib": self.operate_reserved_mib,
            "graph_budget_mib": self.graph_budget_mib,
        }
        for name, value in integer_values.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not 0.0 < self.major_percentage <= 1.0:
            raise ValueError("major_percentage must be in (0, 1]")
        if self.minor_min_mib > self.minor_max_mib:
            raise ValueError("minor_min_mib cannot exceed minor_max_mib")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "policy": self.policy,
            "major_reserved_mib": self.major_reserved_mib,
            "major_percentage": self.major_percentage,
            "unmanaged_reserved_mib": self.unmanaged_reserved_mib,
            "minor_min_mib": self.minor_min_mib,
            "minor_max_mib": self.minor_max_mib,
            "operate_reserved_mib": self.operate_reserved_mib,
        }
        if self.graph_budget_mib:
            result["graph_budget_mib"] = self.graph_budget_mib
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemorySpec":
        fields = set(cls.__dataclass_fields__)
        _require_exact_keys(value, set(), fields, "dealer.memory")
        return cls(**value)

    def make_runtime_config(self) -> MemoryConfig:
        common = {
            "major_percentage": self.major_percentage,
            "unmanaged_reserved": self.unmanaged_reserved_mib * MIB,
            "minor_minimum": self.minor_min_mib * MIB,
            "minor_maximum": self.minor_max_mib * MIB,
            "operate_reserved": self.operate_reserved_mib * MIB,
            "graph_budget": self.graph_budget_mib * MIB,
        }
        if self.policy == "static":
            return MemoryConfigStatic(
                major_reserved=self.major_reserved_mib * MIB,
                **common,
            )
        return MemoryConfigPercentage(**common)


@dataclass(frozen=True)
class DealerSpec:
    name: str
    group_ids: tuple[int, ...]
    decoder_backend: str = "marlin"
    max_seq_len: int = 256
    max_concurrency: int = 8
    enable_cuda_graph: bool = False
    time_trace: bool = False
    time_trace_sync_cuda: bool = False
    failure_injection: bool = False
    memory: MemorySpec = field(default_factory=MemorySpec)

    def __post_init__(self) -> None:
        if not self.name or self.name == "aux":
            raise ValueError("dealer name must be non-empty and cannot be 'aux'")
        if not self.group_ids:
            raise ValueError(f"dealer {self.name!r} must contain decoder groups")
        if len(set(self.group_ids)) != len(self.group_ids):
            raise ValueError(f"dealer {self.name!r} contains duplicate group IDs")
        if tuple(sorted(self.group_ids)) != self.group_ids:
            raise ValueError(
                f"dealer {self.name!r} group IDs must be in ascending order"
            )
        if any(type(group_id) is not int or group_id < 0 for group_id in self.group_ids):
            raise ValueError(f"dealer {self.name!r} group IDs must be non-negative integers")
        if self.decoder_backend not in DECODER_BACKENDS:
            raise ValueError(f"invalid decoder backend {self.decoder_backend!r}")
        if self.max_seq_len <= 0 or self.max_concurrency <= 0:
            raise ValueError("max_seq_len and max_concurrency must be positive")
        if self.enable_cuda_graph and self.decoder_backend != "marlin":
            raise ValueError("online CUDA graph currently requires Marlin")
        if self.enable_cuda_graph and self.memory.graph_budget_mib <= 0:
            raise ValueError(
                "online CUDA graph requires a positive graph_budget_mib"
            )
        if self.time_trace_sync_cuda and not self.time_trace:
            raise ValueError("time_trace_sync_cuda requires time_trace")

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_ids": list(self.group_ids),
            "decoder_backend": self.decoder_backend,
            "max_seq_len": self.max_seq_len,
            "max_concurrency": self.max_concurrency,
            "enable_cuda_graph": self.enable_cuda_graph,
            "time_trace": self.time_trace,
            "time_trace_sync_cuda": self.time_trace_sync_cuda,
            "failure_injection": self.failure_injection,
            "memory": self.memory.to_dict(),
        }

    @classmethod
    def from_dict(cls, name: str, value: Mapping[str, Any]) -> "DealerSpec":
        required = {"group_ids"}
        optional = {
            "decoder_backend",
            "max_seq_len",
            "max_concurrency",
            "enable_cuda_graph",
            "time_trace",
            "time_trace_sync_cuda",
            "failure_injection",
            "memory",
        }
        _require_exact_keys(value, required, optional, f"dealer {name!r}")
        kwargs = dict(value)
        kwargs["group_ids"] = tuple(kwargs["group_ids"])
        if "memory" in kwargs:
            kwargs["memory"] = MemorySpec.from_dict(kwargs["memory"])
        return cls(name=name, **kwargs)

    def inference_config(self) -> OnlineInferenceConfig:
        return OnlineInferenceConfig(
            decoder_backend=self.decoder_backend,
            enable_cuda_graph=self.enable_cuda_graph,
            max_seq_len=self.max_seq_len,
            max_concurrency=self.max_concurrency,
        )


@dataclass(frozen=True)
class NetworkSpec:
    name: str
    model_name: str
    num_decoders: int
    decoder_group_sizes: tuple[int, ...]
    dealers: tuple[DealerSpec, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("network name must be non-empty")
        if not self.model_name:
            raise ValueError("model name must be non-empty")
        if self.num_decoders <= 0:
            raise ValueError("num_decoders must be positive")
        if not self.decoder_group_sizes:
            raise ValueError("decoder_group_sizes must not be empty")
        if any(type(size) is not int or size <= 0 for size in self.decoder_group_sizes):
            raise ValueError("decoder group sizes must be positive integers")
        if sum(self.decoder_group_sizes) != self.num_decoders:
            raise ValueError(
                "decoder group sizes must cover the model exactly: "
                f"sum={sum(self.decoder_group_sizes)}, num_decoders={self.num_decoders}"
            )
        if not self.dealers:
            raise ValueError("network must contain at least one dealer")
        dealer_names = [dealer.name for dealer in self.dealers]
        if len(set(dealer_names)) != len(dealer_names):
            raise ValueError("dealer names must be unique")
        group_count = len(self.decoder_group_sizes)
        for dealer in self.dealers:
            if any(group_id >= group_count for group_id in dealer.group_ids):
                raise ValueError(
                    f"dealer {dealer.name!r} references a group outside [0, {group_count})"
                )
        covered = {group_id for dealer in self.dealers for group_id in dealer.group_ids}
        expected = set(range(group_count))
        if covered != expected:
            raise ValueError(
                "dealer assignments must cover every decoder group; "
                f"missing={sorted(expected - covered)}"
            )

    @property
    def dealer_names(self) -> tuple[str, ...]:
        return tuple(dealer.name for dealer in self.dealers)

    def group_names(self) -> tuple[str, ...]:
        _, groups, _ = group_model_with_split(
            self.model_name,
            self.num_decoders,
            lambda _layers: list(self.decoder_group_sizes),
        )
        return tuple(groups)

    def model_targets(self, dealer: DealerSpec) -> ModelTargets:
        head, groups, tail = group_model_with_split(
            self.model_name,
            self.num_decoders,
            lambda _layers: list(self.decoder_group_sizes),
        )
        selected = [groups[group_id] for group_id in dealer.group_ids]
        bundle: dict[str, list[str]] = {}
        if 0 in dealer.group_ids:
            bundle.setdefault(groups[0], []).append(head)
        if len(groups) - 1 in dealer.group_ids:
            bundle.setdefault(groups[-1], []).append(tail)
        return ModelTargets.from_specific(decoders=selected, bundle=bundle)

    def extra_info(self) -> ExtraInfo:
        model_lib = ModelsLib(split_qllm_models.lib_path)
        try:
            model_info = model_lib.models_storage[self.model_name]
        except KeyError as error:
            raise ValueError(f"unknown model {self.model_name!r}") from error
        head, groups, tail = group_model_with_split(
            self.model_name,
            self.num_decoders,
            lambda _layers: list(self.decoder_group_sizes),
        )
        return ExtraInfo(
            models={
                self.model_name: ExtraModelInfo(
                    hidden_size=model_info.config.hidden_size,
                    num_decoders=self.num_decoders,
                    decoders_groups=groups,
                    head=head,
                    tail=tail,
                )
            },
            network_name=self.name,
            network_spec_hash=self.spec_hash(),
        )

    def _content_dict(self) -> dict[str, Any]:
        ranges = []
        start = 0
        for group_id, size in enumerate(self.decoder_group_sizes):
            ranges.append({"id": group_id, "range": [start, start + size]})
            start += size
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "model": {
                "name": self.model_name,
                "num_decoders": self.num_decoders,
                "decoder_group_sizes": list(self.decoder_group_sizes),
                "decoder_groups": ranges,
            },
            "dealers": {
                dealer.name: dealer.to_dict()
                for dealer in self.dealers
            },
        }

    def spec_hash(self) -> str:
        payload = json.dumps(
            self._content_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        result = self._content_dict()
        result["spec_sha256"] = self.spec_hash()
        return result

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NetworkSpec":
        _require_exact_keys(
            value,
            {"schema_version", "name", "model", "dealers", "spec_sha256"},
            set(),
            "network spec",
        )
        if value["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {value['schema_version']!r}"
            )
        model = value["model"]
        if not isinstance(model, Mapping):
            raise TypeError("model must be an object")
        _require_exact_keys(
            model,
            {"name", "num_decoders", "decoder_group_sizes", "decoder_groups"},
            set(),
            "model",
        )
        dealers_value = value["dealers"]
        if not isinstance(dealers_value, Mapping):
            raise TypeError("dealers must be an object")
        dealers = tuple(
            DealerSpec.from_dict(str(name), config)
            for name, config in dealers_value.items()
            if isinstance(config, Mapping)
        )
        if len(dealers) != len(dealers_value):
            raise TypeError("every dealer configuration must be an object")
        spec = cls(
            name=str(value["name"]),
            model_name=str(model["name"]),
            num_decoders=int(model["num_decoders"]),
            decoder_group_sizes=tuple(model["decoder_group_sizes"]),
            dealers=dealers,
        )
        expected_ranges = spec._content_dict()["model"]["decoder_groups"]
        if model["decoder_groups"] != expected_ranges:
            raise ValueError("model.decoder_groups does not match decoder_group_sizes")
        if value["spec_sha256"] != spec.spec_hash():
            raise ValueError("network spec hash does not match its content")
        return spec

    @classmethod
    def load(cls, path: Path) -> "NetworkSpec":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("network spec must contain a JSON object")
        return cls.from_dict(value)


def model_decoder_count(model_name: str) -> int:
    model_lib = ModelsLib(split_qllm_models.lib_path)
    try:
        return int(model_lib.models_storage[model_name].config.num_decoders)
    except KeyError as error:
        available = ", ".join(sorted(model_lib.models_storage))
        raise ValueError(
            f"unknown model {model_name!r}; available models: {available}"
        ) from error
