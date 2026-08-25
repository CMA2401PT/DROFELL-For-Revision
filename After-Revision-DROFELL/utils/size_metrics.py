"""Estimate retained Python payload and exact tensor storage bytes."""

from __future__ import annotations

import dataclasses
import sys
from typing import Any, Set

import torch


def retained_bytes(value: Any, seen: Set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (bool, int, float)):
        return sys.getsizeof(value)
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)
    if isinstance(value, torch.Tensor):
        return value.nelement() * value.element_size()
    tensor = getattr(value, "t", None)
    if isinstance(tensor, torch.Tensor):
        return sys.getsizeof(value) + retained_bytes(tensor, seen)
    if isinstance(value, (bytes, bytearray, memoryview, str)):
        return sys.getsizeof(value)
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(
            retained_bytes(key, seen) + retained_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return sys.getsizeof(value) + sum(retained_bytes(item, seen) for item in value)
    if dataclasses.is_dataclass(value):
        return sys.getsizeof(value) + sum(
            retained_bytes(getattr(value, field.name), seen)
            for field in dataclasses.fields(value)
        )
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return sys.getsizeof(value) + retained_bytes(attributes, seen)
    return sys.getsizeof(value)
