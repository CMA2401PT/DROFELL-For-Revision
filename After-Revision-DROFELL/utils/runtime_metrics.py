"""Low-overhead JSONL event recording for experiment instrumentation."""

from __future__ import annotations

import json
import os
import atexit
import threading
import time
from typing import Any


_lock = threading.Lock()
_descriptor: int | None = None
_descriptor_pid: int | None = None
_descriptor_path: str | None = None
_sample_counters: dict[tuple[int, str, str], int] = {}


def _close_descriptor() -> None:
    global _descriptor, _descriptor_pid, _descriptor_path
    with _lock:
        if _descriptor is not None:
            try:
                os.close(_descriptor)
            except OSError:
                pass
        _descriptor = None
        _descriptor_pid = None
        _descriptor_path = None


atexit.register(_close_descriptor)


def _get_descriptor(path: str) -> int:
    global _descriptor, _descriptor_pid, _descriptor_path
    pid = os.getpid()
    if (
        _descriptor is not None
        and _descriptor_pid == pid
        and _descriptor_path == path
    ):
        return _descriptor
    if _descriptor is not None:
        try:
            os.close(_descriptor)
        except OSError:
            pass
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    _descriptor = os.open(path, flags, 0o644)
    _descriptor_pid = pid
    _descriptor_path = path
    return _descriptor


def metrics_event_enabled(category: str) -> bool:
    raw = os.getenv("DROFELL_METRICS_EVENTS", "")
    enabled = {item.strip() for item in raw.split(",") if item.strip()}
    return "*" in enabled or "all" in enabled or category in enabled


def _sample_every(category: str) -> int:
    specific = os.getenv(
        f"DROFELL_METRICS_SAMPLE_EVERY_{category.upper()}"
    )
    raw = specific or os.getenv("DROFELL_METRICS_SAMPLE_EVERY", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def record_sampled_event(category: str, event: str, **fields: Any) -> None:
    """Record an opt-in event with deterministic per-process sampling."""
    if not metrics_event_enabled(category):
        return
    key = (os.getpid(), category, event)
    with _lock:
        count = _sample_counters.get(key, 0)
        _sample_counters[key] = count + 1
    every = _sample_every(category)
    if count % every != 0:
        return
    record_event(
        event,
        metrics_category=category,
        sample_ordinal=count,
        sample_every=every,
        **fields,
    )


def record_event(event: str, **fields: Any) -> None:
    path = os.getenv("DROFELL_METRICS_PATH")
    if not path:
        return
    record = {
        "schema_version": 1,
        "event": event,
        "time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        **fields,
    }
    payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    with _lock:
        descriptor = _get_descriptor(path)
        os.write(descriptor, payload)
