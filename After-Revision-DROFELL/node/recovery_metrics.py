"""Process-local recovery correlation built on the shared runtime event sink."""

from __future__ import annotations

from typing import Dict, Optional
from uuid import uuid4

from utils.runtime_metrics import record_event


_pending_by_control: Dict[str, str] = {}


def new_recovery_id() -> str:
    return uuid4().hex


def mark_pending(control_id: str, recovery_id: str) -> None:
    _pending_by_control[control_id] = recovery_id


def record_first_output(
    control_id: str,
    step: int,
    trace_invalid: bool,
    packet_recovery_id: Optional[str] = None,
    recovery_packet_id: Optional[str] = None,
    recovery_task_index: Optional[int] = None,
    recovery_role: Optional[str] = None,
) -> None:
    recovery_id = packet_recovery_id or _pending_by_control.get(control_id)
    if recovery_id is None:
        return
    if trace_invalid:
        record_event(
            "replay_output_observed",
            recovery_id=recovery_id,
            control_id=control_id,
            step=step,
            packet_id=recovery_packet_id,
            task_index=recovery_task_index,
            role=recovery_role,
        )
        return
    _pending_by_control.pop(control_id, None)
    record_event(
        "first_output_after_recovery",
        recovery_id=recovery_id,
        control_id=control_id,
        step=step,
        trace_invalid=trace_invalid,
    )


def record_intermediate_arrival(
    control_id: str,
    step: int,
    current_model_index: int,
    trace_invalid: bool,
) -> None:
    """Record a root-routed split activation without calling it an output."""
    recovery_id = _pending_by_control.get(control_id)
    if recovery_id is None:
        return
    record_event(
        "recovery_intermediate_arrival",
        recovery_id=recovery_id,
        control_id=control_id,
        step=step,
        current_model_index=current_model_index,
        trace_invalid=trace_invalid,
    )
