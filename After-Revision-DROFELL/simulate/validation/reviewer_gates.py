#!/usr/bin/env python3
"""Unified fail-closed paper-eligibility gate for DROSIM artifacts.

Every required evidence file must explicitly pass. Missing evidence is a failed
gate, never an implicit skip. This runner does not make a simulation valid; it
only prevents partial checks from being presented as full validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def evidence(path: Optional[Path], gate: str) -> dict[str, Any]:
    if path is None:
        return {"passed": False, "reason": f"{gate} evidence not supplied"}
    if not path.exists():
        return {"passed": False, "reason": f"not found: {path}"}
    data = json.loads(path.read_text())
    passed = bool(data.get("passed", data.get("paper_eligible", False)))
    return {
        "passed": passed,
        "path": str(path),
        "reason": data.get("reason") or data.get("limitation"),
        "evidence_id": data.get("evidence_id") or data.get("dataset_id"),
    }


def gate_g0(result: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "schema_v2": int(result.get("schema_version", 0)) >= 2,
        "timing_v2": int(result.get("timing", {}).get("schema_version", 0)) >= 2,
        "has_itl_intervals": (
            int(result.get("timing", {}).get("itl_ms", {}).get("count", 0)) > 0
        ),
        "single_hardware_cost_profile": (
            bool(result.get("cost_model", {}).get("hardware_id"))
            and "mixed" not in str(
                result.get("cost_model", {}).get("hardware_id", "")
            ).lower()
        ),
        "cost_profile_gate": bool(
            result.get("cost_model", {}).get("paper_eligible", False)
        ),
        "actual_link_policy_reported": bool(
            result.get("fabric", {}).get("link_policy")
        ),
        "shared_host_interfaces": bool(
            result.get("fabric", {}).get("shared_host_interfaces", False)
        ),
        "clean_teardown": not result.get("notes"),
        "no_cost_extrapolation": not result.get("cost_model", {}).get(
            "extrapolations"
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "reason": None if not failed else "failed: " + ", ".join(failed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--g1", type=Path)
    parser.add_argument("--g2", type=Path)
    parser.add_argument("--g3", type=Path)
    parser.add_argument("--g4", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    gates = {
        "G0_implementation_manifest": gate_g0(result),
        "G1_structural_equivalence": evidence(args.g1, "G1"),
        "G2_single_host_holdout_calibration": evidence(args.g2, "G2"),
        "G3_numeric_trend_validation": evidence(args.g3, "G3"),
        "G4_shadow_recovery_validation": evidence(args.g4, "G4"),
    }
    paper_eligible = all(gate["passed"] for gate in gates.values())
    report = {
        "schema_version": 1,
        "evidence_id": "drosim-reviewer-gates-v1",
        "result": str(args.result),
        "paper_eligible": paper_eligible,
        "passed": paper_eligible,
        "gates": gates,
        "reason": (
            None if paper_eligible else
            "one or more mandatory reviewer-grade gates failed or are missing"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"paper_eligible={str(paper_eligible).lower()}")
    for name, gate in gates.items():
        print(f"  {name}: {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"wrote {args.out}")
    raise SystemExit(0 if paper_eligible else 2)


if __name__ == "__main__":
    main()

