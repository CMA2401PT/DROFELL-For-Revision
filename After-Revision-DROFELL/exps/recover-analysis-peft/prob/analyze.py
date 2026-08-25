#!/usr/bin/env python3
"""Locate the earliest exact-state divergence in determinism probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def by_event(rows: list[dict[str, Any]], event: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["event"] == event]


def compare_optimizer(left: Path, right: Path) -> dict[str, Any]:
    lhs = read_json(left / "fingerprints.json")
    rhs = read_json(right / "fingerprints.json")
    equal = lhs == rhs
    first_difference = None
    for index, (a, b) in enumerate(zip(lhs, rhs)):
        if a != b:
            first_difference = {
                "event_index": index,
                "left_event": a["event"],
                "right_event": b["event"],
            }
            break
    if first_difference is None and len(lhs) != len(rhs):
        first_difference = {"event_count": [len(lhs), len(rhs)]}
    return {"exactly_equal": equal, "first_difference": first_difference}


def update_rows(path: Path) -> dict[int, dict[str, Any]]:
    events = read_json(path / "fingerprints.json")
    rows = by_event(events, "optimizer_step")
    return {int(row["update_index"]): row for row in rows}


def train_comparison(left: Path, right: Path) -> dict[str, Any]:
    lhs_events = read_json(left / "fingerprints.json")
    rhs_events = read_json(right / "fingerprints.json")
    lhs_initial = by_event(lhs_events, "adapter_initialized")
    rhs_initial = by_event(rhs_events, "adapter_initialized")
    if len(lhs_initial) != 1 or len(rhs_initial) != 1:
        raise RuntimeError("each train probe must contain one initialization")
    initial_equal = (
        lhs_initial[0]["adapter"]["sha256"]
        == rhs_initial[0]["adapter"]["sha256"]
    )
    lhs_record = read_json(left / "record.json")
    rhs_record = read_json(right / "record.json")
    lhs_losses = [float(value) for value in lhs_record["history"]]
    rhs_losses = [float(value) for value in rhs_record["history"]]
    first_loss_difference = next(
        (
            {
                "minibatch_index": index,
                "left": a,
                "right": b,
                "abs_difference": abs(a - b),
            }
            for index, (a, b) in enumerate(zip(lhs_losses, rhs_losses))
            if a != b
        ),
        None,
    )
    lhs_updates = update_rows(left)
    rhs_updates = update_rows(right)
    if set(lhs_updates) != set(rhs_updates):
        raise RuntimeError("train probes have different update indices")
    boundaries = (
        ("parameters_before_step", ("before", "parameters", "sha256")),
        ("gradients_before_step", ("before", "gradients", "sha256")),
        ("optimizer_before_step", ("before", "optimizer", "sha256")),
        ("parameters_after_step", ("after", "parameters", "sha256")),
        ("optimizer_after_step", ("after", "optimizer", "sha256")),
    )

    def get(row: dict[str, Any], path: tuple[str, ...]) -> Any:
        value: Any = row
        for key in path:
            value = value[key]
        return value

    update_comparisons = []
    first_state_difference = None
    for update in sorted(lhs_updates):
        fields = {
            name: get(lhs_updates[update], path) == get(rhs_updates[update], path)
            for name, path in boundaries
        }
        update_comparisons.append({"update_index": update, **fields})
        if first_state_difference is None:
            for name, _ in boundaries:
                if not fields[name]:
                    first_state_difference = {
                        "update_index": update,
                        "boundary": name,
                    }
                    break
    stage_names = (
        "adapter_forward_output",
        "decoder_forward_output",
        "decoder_backward_input",
        "adapter_backward_input",
        "adapter_gradient_after_backward",
    )
    stage_comparisons = []
    stage_differences: list[tuple[int, int, str]] = []
    for stage_order, event in enumerate(stage_names):
        lhs_stages = {
            int(row["minibatch_index"]): row["fingerprint"]["sha256"]
            for row in by_event(lhs_events, event)
        }
        rhs_stages = {
            int(row["minibatch_index"]): row["fingerprint"]["sha256"]
            for row in by_event(rhs_events, event)
        }
        exact = lhs_stages == rhs_stages
        first_index = next(
            (
                index for index in sorted(set(lhs_stages) | set(rhs_stages))
                if lhs_stages.get(index) != rhs_stages.get(index)
            ),
            None,
        )
        stage_comparisons.append({
            "stage": event,
            "exactly_equal": exact,
            "first_difference_minibatch": first_index,
        })
        if first_index is not None:
            stage_differences.append((first_index, stage_order, event))
    first_stage_difference = None
    if stage_differences:
        first_index, _, first_event = min(stage_differences)
        first_stage_difference = {
            "stage": first_event,
            "minibatch_index": first_index,
        }
    return {
        "initial_adapter_exactly_equal": initial_equal,
        "losses_exactly_equal": lhs_losses == rhs_losses,
        "first_loss_difference": first_loss_difference,
        "first_stage_difference": first_stage_difference,
        "stages": stage_comparisons,
        "first_state_difference": first_state_difference,
        "updates": update_comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    root = args.result_root.resolve()
    manifest = read_json(root / "manifest.json")
    conditions = {item["name"]: root / item["relative_path"] for item in manifest["runs"]}
    required = {
        "optimizer-gpu0-r0", "optimizer-gpu0-r1",
        "train-gpu0-r0", "train-gpu0-r1",
    }
    if not required.issubset(conditions):
        raise RuntimeError(f"missing required probes: {sorted(required - conditions.keys())}")
    summary = {
        "schema_version": 1,
        "parameters": manifest["parameters"],
        "same_gpu_optimizer": compare_optimizer(
            conditions["optimizer-gpu0-r0"], conditions["optimizer-gpu0-r1"]
        ),
        "same_gpu_training": train_comparison(
            conditions["train-gpu0-r0"], conditions["train-gpu0-r1"]
        ),
    }
    if "optimizer-gpu1-r0" in conditions:
        summary["cross_gpu_optimizer"] = compare_optimizer(
            conditions["optimizer-gpu0-r0"], conditions["optimizer-gpu1-r0"]
        )
    if "train-gpu1-r0" in conditions:
        summary["cross_gpu_training"] = train_comparison(
            conditions["train-gpu0-r0"], conditions["train-gpu1-r0"]
        )
    same_optimizer = summary["same_gpu_optimizer"]["exactly_equal"]
    same_train = summary["same_gpu_training"]
    if not same_optimizer:
        conclusion = "AdamW diverged despite identical synthetic parameters and gradients."
    elif same_train["first_state_difference"] is None:
        conclusion = (
            "Every instrumented state in this short sequential monolithic training "
            "probe was exactly repeatable on the same GPU."
        )
    elif same_train["first_state_difference"]["boundary"] == "gradients_before_step":
        conclusion = (
            "The accumulated gradient diverged before AdamW.step; "
            f"the earliest instrumented divergence was {same_train['first_stage_difference']}."
        )
    else:
        conclusion = (
            "Training state diverged at "
            f"{same_train['first_state_difference']['boundary']}."
        )
    summary["conclusion"] = conclusion
    write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
