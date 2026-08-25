#!/usr/bin/env python3
"""End-to-end tests: the production control plane really runs, and recovers.

`test_drosim_core.py` checks the three replaced boundaries in isolation. This
file checks the thing that actually matters: that the UNMODIFIED
AuxNode / AgentNode / DealerNode / InferNode / InferControl / Provider / Shim /
STM / shadow-state stack completes a request on top of them, and recovers from a
node failure while doing so.

Kept separate from the core tests because each scenario builds a whole deployment
(seconds, not milliseconds).

Run from the repository root:
    python simulate/tests/test_drosim_e2e.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from simulate.drosim.run import SimConfig, run
    from simulate.drosim.topology import Placement

    # ---- 1. one request, four logical nodes -----------------------------
    print("\n1. a complete request through the real control plane")
    started = time.monotonic()
    result = run(SimConfig(segments=4, clients=1, tokens=8, log_level="error"))
    wall = time.monotonic() - started
    check("all requested tokens were produced",
          result.tokens_produced == 8, f"{result.tokens_produced}/8")
    check("virtual time advanced", result.virtual_seconds > 0,
          f"{result.virtual_seconds:.4f}s virtual in {wall:.2f}s wall")
    check("timing schema separates setup/workload/teardown",
          result.timing["schema_version"] == 2
          and result.timing["phases_ms"]["setup"] > 0
          and result.timing["phases_ms"]["teardown_drain"] > 0,
          str(result.timing["phases_ms"]))
    check("ITL is built from adjacent output timestamps",
          result.timing["itl_ms"]["count"] == 7
          and abs(
              result.per_client_inter_token_ms
              - result.timing["itl_ms"]["mean"]
          ) < 1e-12,
          f"count={result.timing['itl_ms']['count']} "
          f"mean={result.timing['itl_ms']['mean']}")
    old_total_div_tokens = 1000 * result.virtual_seconds / result.tokens_produced
    check("reported ITL excludes setup and teardown",
          abs(result.per_client_inter_token_ms - old_total_div_tokens) > 0.1,
          f"ITL={result.per_client_inter_token_ms:.2f}ms vs legacy "
          f"{old_total_div_tokens:.2f}ms")
    # NOT asserted on this run: an 8-token scenario is dominated by fixed setup
    # (building the twin, loading four shard models, the announce handshake), so
    # its virtual/wall ratio measures startup, not simulation speed. Measured on
    # a longer run below, where the ratio means what it says.
    check("packets actually crossed the fabric",
          result.fabric["packets"] > 0,
          f"{result.fabric['packets']} packets / {result.fabric['bytes_total']} bytes")
    check("the memory twin accounted real bytes",
          all(b["twin_total_bytes"] > 0 for b in result.backends),
          f"{result.backends[0]['twin_total_bytes'] / 2**30:.3f} GiB per node")
    check("no unexplained notes", not result.notes, f"{result.notes}")
    # A run that never merged anything would not be exercising the production
    # merge policy, so this is a check that the seam is live rather than bypassed.
    stages = {
        stage for backend in result.backends
        for stage in backend["operations_by_stage"]
    }
    check("every infer stage was exercised",
          {"decoder_prefill", "decoder_generating", "infer_head", "infer_tail"}
          <= stages,
          f"{sorted(stages)}")
    prefill_term = result.cost_model["terms"]["decoder_prefill"]
    expected_prefill_ns = round((
        prefill_term["alpha_us"]
        + prefill_term["beta_us_per_item"] * 4
        + prefill_term["gamma_us_per_layer"] * 8
        + prefill_term["delta_us_per_item_layer"] * 4 * 8
    ) * 1000)
    check(
        "prefill service uses the real 8-layer shard",
        all(
            backend["service_ns_by_stage"].get("decoder_prefill")
            == expected_prefill_ns
            for backend in result.backends
        ),
        f"expected {expected_prefill_ns}ns",
    )

    print("\n1b. simulation speed, measured where setup is amortised")
    started = time.monotonic()
    longer = run(SimConfig(segments=4, clients=8, tokens=64, log_level="error"))
    long_wall = time.monotonic() - started
    ratio = longer.virtual_seconds / max(long_wall, 1e-9)
    check("a longer run beats real time",
          ratio > 1.0,
          f"{longer.virtual_seconds:.3f}s virtual in {long_wall:.2f}s wall "
          f"= {ratio:.1f}x")
    check("and it produced every token",
          longer.tokens_produced == 8 * 64,
          f"{longer.tokens_produced}/{8 * 64}")

    # ---- 2. scale monotonicity ------------------------------------------
    print("\n2. more segments cost more; more holders do not")
    by_segments = {}
    for segments in (1, 2, 4, 8):
        scaled = run(
            SimConfig(segments=segments, clients=1, tokens=4, log_level="error")
        )
        by_segments[segments] = scaled.per_client_inter_token_ms
    rising = all(
        by_segments[a] < by_segments[b]
        for a, b in zip((1, 2, 4), (2, 4, 8))
    )
    check("per-token latency rises with segment count", rising,
          ", ".join(f"L={k}:{v:.1f}ms" for k, v in by_segments.items()))

    by_redundancy = {}
    for redundancy in (1, 2, 4):
        scaled = run(
            SimConfig(
                segments=4, redundancy=redundancy, clients=1, tokens=4,
                log_level="error",
            )
        )
        by_redundancy[redundancy] = scaled.per_client_inter_token_ms
    latencies = list(by_redundancy.values())
    flat = (max(latencies) - min(latencies)) / min(latencies) < 0.05
    check("per-token latency is flat in holder count (<5%)", flat,
          ", ".join(f"r={k}:{v:.1f}ms" for k, v in by_redundancy.items()))

    unraisable: list[str] = []
    old_unraisablehook = sys.unraisablehook
    sys.unraisablehook = lambda event: unraisable.append(
        repr(event.exc_value)
    )
    try:
        lifecycle = run(
            SimConfig(
                segments=4, redundancy=8, clients=1, tokens=4,
                log_level="error",
            )
        )
        gc.collect()
    finally:
        sys.unraisablehook = old_unraisablehook
    check(
        "late shim collection does not double-release decoder bindings",
        lifecycle.tokens_produced == 4 and not unraisable,
        str(unraisable),
    )

    # ---- 3. concurrency uses the production merge policy ----------------
    print("\n3. concurrency scales throughput via the real merge policy")
    single = run(SimConfig(segments=1, clients=1, tokens=6, log_level="error"))
    eight = run(SimConfig(segments=1, clients=8, tokens=6, log_level="error"))
    speedup = eight.aggregate_tokens_per_second / single.aggregate_tokens_per_second
    check("8 concurrent clients give a large speedup", speedup > 4.0,
          f"{speedup:.2f}x")
    merged = max(
        backend["mean_merged_batch"].get("decoder_generating", 0.0)
        for backend in eight.backends
    )
    check("operations really merged", merged > 1.0,
          f"mean merged batch {merged:.2f}")

    persistent = run(
        SimConfig(
            segments=1, clients=1, tokens=6,
            deployment="persistent-agent", log_level="error",
        )
    )
    check("persistent Agent uses the production remote RequestNode path",
          persistent.tokens_produced == 6
          and persistent.config["deployment"] == "persistent-agent"
          and persistent.fabric["packets"] > single.fabric["packets"],
          f"combined={single.fabric['packets']} packets, "
          f"persistent={persistent.fabric['packets']} packets")

    # ---- 4. recovery ----------------------------------------------------
    print("\n4. a killed node is recovered and the request still completes")
    handle = tempfile.NamedTemporaryFile(
        prefix="drosim-e2e-", suffix=".jsonl", delete=False
    )
    handle.close()
    metrics_path = Path(handle.name)
    os.environ["DROFELL_METRICS_PATH"] = str(metrics_path)
    try:
        recovered = run(
            SimConfig(
                segments=4, redundancy=2, clients=1, tokens=16,
                log_level="error", deterministic_drop_steps=(6,),
            )
        )
        events = [
            json.loads(line)
            for line in metrics_path.read_text().splitlines()
            if line.strip()
        ]
    finally:
        metrics_path.unlink(missing_ok=True)
        os.environ.pop("DROFELL_METRICS_PATH", None)

    seen = {record["event"] for record in events}
    check("a failure was injected",
          recovered.injection["drops_triggered"] > 0,
          f"{recovered.injection['drops_triggered']} drops")
    check("the request still completed",
          recovered.tokens_produced == 16, f"{recovered.tokens_produced}/16")
    for event in (
        "failure_detected", "replacement_lookup_start",
        "replacement_graph_installed", "recovery_packet_send",
        "first_recovery_task_complete", "frontier_materialized",
    ):
        check(f"recovery emitted {event}", event in seen)
    # The duplicate guard is what makes redundancy levels comparable; without it
    # the effective failure rate would scale with redundancy (drosim/failure.py).
    check("the duplicate guard suppressed repeat arrivals",
          recovered.injection["duplicates_skipped"] > 0,
          f"{recovered.injection['duplicates_skipped']} duplicates skipped")

    # ---- 5. shared devices ---------------------------------------------
    print("\n5. shared-device placement is expressible and reported")
    shared = run(
        SimConfig(
            segments=8, redundancy=1, clients=2, tokens=4,
            placement=Placement.SHARED, device_count=2, log_level="error",
        )
    )
    check("run completed on a shared device pool",
          shared.tokens_produced == 8, f"{shared.tokens_produced}/8")
    check("the result states its placement",
          shared.topology["placement"] == "shared",
          shared.topology["placement"])
    device_ids = {backend["device_id"] for backend in shared.backends}
    check("8 logical nodes really shared 2 devices",
          len(device_ids) == 2, f"{sorted(device_ids)}")

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {FAILURES}")
        return 1
    print("all end-to-end checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
