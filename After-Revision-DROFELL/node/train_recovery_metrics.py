"""Process-local correlation for training recovery replay completion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Set
from uuid import uuid4

from utils.runtime_metrics import record_event


@dataclass
class _PendingRecovery:
    recovery_id: str
    expected_minibatches: Set[str]
    observed_minibatches: Set[str] = field(default_factory=set)
    replay_complete: bool = False


_pending_by_control: Dict[str, _PendingRecovery] = {}


def new_train_recovery_id() -> str:
    return uuid4().hex


def mark_train_recovery(
    control_id: str,
    recovery_id: str,
    minibatch_ids: Iterable[str],
) -> None:
    expected = set(minibatch_ids)
    pending = _PendingRecovery(recovery_id, expected)
    _pending_by_control[control_id] = pending
    if not expected:
        pending.replay_complete = True
        record_event(
            "train_replay_complete",
            recovery_id=recovery_id,
            control_id=control_id,
            expected_minibatch_count=0,
            observed_minibatch_count=0,
        )


def record_train_output(
    control_id: str,
    minibatch_id: str,
    request_version: int,
    trace_invalid: bool,
) -> None:
    pending = _pending_by_control.get(control_id)
    if pending is None:
        return
    if trace_invalid:
        record_event(
            "train_replay_output_observed",
            recovery_id=pending.recovery_id,
            control_id=control_id,
            minibatch_id=minibatch_id,
            request_version=request_version,
        )
        if minibatch_id in pending.expected_minibatches:
            pending.observed_minibatches.add(minibatch_id)
        if (
            not pending.replay_complete
            and pending.observed_minibatches == pending.expected_minibatches
        ):
            pending.replay_complete = True
            record_event(
                "train_replay_complete",
                recovery_id=pending.recovery_id,
                control_id=control_id,
                expected_minibatch_count=len(pending.expected_minibatches),
                observed_minibatch_count=len(pending.observed_minibatches),
            )
        return
    event = (
        "first_train_output_after_recovery"
        if pending.replay_complete
        else "train_output_during_recovery"
    )
    record_event(
        event,
        recovery_id=pending.recovery_id,
        control_id=control_id,
        minibatch_id=minibatch_id,
        request_version=request_version,
        trace_invalid=False,
    )
    if pending.replay_complete:
        _pending_by_control.pop(control_id, None)
