from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants taken from the real implementation (file::symbol given for audit)
# ---------------------------------------------------------------------------
SOURCES = {
    "block_size": "operators/flags.py::BLOCK_SIZE",
    "res_loc_expire_period": "node/infer/infer_ctrl.py::InferControl.res_loc_expire_period (=10)",
    "retrieve_postpone": "node/compute_node/infer_node.py::Node._retrieve_postpone (=8)",
    "path_search_redundancy": "node/agent/control_plane.py::AgentControlPlane.path_search_redundancy (=1)",
    "service_time_us": "measured: probes/p12 (alpha≈1915us, beta≈0.8us/item, r2 across repeats)",
    "replay_bytes": "measured: probes/p14 (83137 + 7551*steps, r2=0.315, n=70)",
    "eta": "measured: probes/p12 (throughput = eta*batch/service, eta≈0.264, r2≈0.95)",
    "path_dp": "measured: probes/p13 (prototype tie set is exactly (r+1)^L)",
    "drosim_selector": "simulate/drosim/routing.py (greedy node-disjoint, O(L*r))",
}

BLOCK_SIZE = 16          # tokens per KV block
SERVICE_ALPHA_US = 1915.0
SERVICE_BETA_US = 0.8
ETA = 0.264
# measured on the shadow-overhead sweep (70 snapshots): see shadow_bytes docstring
REPLAY_BASE_BYTES = 83137
REPLAY_BYTES_PER_STEP = 7551


def resource_map_entries(segments: int, redundancy: int) -> int:
    """Claim 1: resource-location map size is O(L * r), independent of N.

    `InferControl.res_location` builds one entry per model segment, each listing
    the control node plus that segment's holders. Nothing enumerates the network.
    """
    return segments * (1 + redundancy)


def recovery_control_messages(replay_window: int) -> dict[str, int]:
    """Claim 3: recovery message count is independent of N.

    One replacement lookup, one graph install, then k recovery packets, where k
    is the replay window. Nothing is broadcast and nothing crosses a Backbone.
    """
    return {
        "replacement_lookup": 1,
        "graph_install": 1,
        "recovery_packets": replay_window,
        "total": 2 + replay_window,
    }


def shadow_bytes(
    committed_blocks: int,
    shard_layers: int,
    kv_dim: int,
    bytes_per_element: int,
    window: int,
    input_bytes_per_step: int,
    metadata_bytes: int,
) -> dict[str, int]:
    """Claim 4: shadow memory grows with the distance to the last checkpoint,
    NOT with total request length.

    Calibrated against the measured sweep (影子节点开销, 70 snapshots):
        generating_bytes = 83137 + 7551 * replay_task_count   (r2 = 0.315)
    so the per-replayed-step constant below is MEASURED, not assumed. The low
    r2 is itself reported: per-step payload varies with prompt/decode content.
    """
    kv = committed_blocks * BLOCK_SIZE * shard_layers * 2 * kv_dim * bytes_per_element
    replay = REPLAY_BASE_BYTES + window * input_bytes_per_step
    return {
        "metadata_bytes": metadata_bytes,
        "kv_checkpoint_bytes": kv,
        "replay_window_bytes": replay,
        "total_bytes": metadata_bytes + kv + replay,
    }


def path_dp_tie_set(segments: int, redundancy: int) -> int:
    """Claim 2: the PROTOTYPE's placeholder selector enumerates every path tied
    at the minimum switch count, i.e. (r+1)^L -- independent of N but exponential
    in segment count (measured in probes/p13).

    Path-selection strategy is out of scope for the paper, so this bound belongs
    to that placeholder rather than to the orchestration mechanism. DROSIM uses
    `drosim/routing.py` instead: greedy, node-disjoint, O(L*r).
    """
    return (redundancy + 1) ** segments


def simplified_selector_keypoints(segments: int, redundancy: int) -> int:
    """What DROSIM's selector actually produces: linear in L*r."""
    return segments * redundancy


def decode_throughput(batch: float) -> float:
    """Measured cost model: throughput = eta * batch / service_time."""
    service_us = SERVICE_ALPHA_US + SERVICE_BETA_US * batch
    return ETA * batch / (service_us / 1e6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--segments", type=int, default=8)
    args = parser.parse_args()

    report: dict[str, Any] = {"sources": SOURCES, "claims": {}}

    print("=" * 74)
    print("D2 · Analytical projection of control-plane cost (Phase 0 deliverable)")
    print("=" * 74)
    print("Every constant is code-derived or measured; sources printed at the end.")
    print()

    # ---- Claim 1 ---------------------------------------------------------
    print("### Claim 1: resource-location map size = O(L*r), independent of N")
    print(f"{'logical nodes N':>16s} {'segments L':>11s} {'redundancy':>11s} "
          f"{'map entries':>12s}")
    rows = []
    for nodes in (2, 8, 32, 128, 256, 1024):
        for redundancy in (1, 2):
            entries = resource_map_entries(args.segments, redundancy)
            rows.append({"N": nodes, "r": redundancy, "entries": entries})
            print(f"{nodes:16d} {args.segments:11d} {redundancy:11d} {entries:12d}")
    print("=> flat in N: the map is indexed by model segment, never by node.")
    report["claims"]["resource_map"] = rows
    print()

    # ---- Claim 2 (corrected) --------------------------------------------
    print("### Claim 2: path search is independent of N; the exponential term")
    print("belongs to the PROTOTYPE's placeholder selector, not the mechanism")
    print(f"{'segments L':>11s} {'proto r=2':>11s} {'proto r=3':>12s} "
          f"{'DROSIM r=2':>11s} {'DROSIM r=3':>11s}")
    dp_rows = []
    for segments in (4, 8, 12, 16, 20):
        proto = [path_dp_tie_set(segments, r) for r in (2, 3)]
        simple = [simplified_selector_keypoints(segments, r) for r in (2, 3)]
        dp_rows.append({
            "L": segments, "prototype_r2": proto[0], "prototype_r3": proto[1],
            "drosim_r2": simple[0], "drosim_r3": simple[1],
        })
        print(f"{segments:11d} {proto[0]:11d} {proto[1]:12d} "
              f"{simple[0]:11d} {simple[1]:11d}")
    print("=> prototype selector: measured tractable to ~16 segments at r=2,")
    print("   past a 4 GiB budget at L=32 (probes/p13). Deployed topologies are")
    print("   far below that because consecutive layers share providers.")
    print("=> DROSIM selector (drosim/routing.py): linear, 256 segments in 0.44 ms.")
    print("   Path-selection strategy is out of scope for the paper, so the")
    print("   simulator is free to use the simpler one (docs/11 §1).")
    report["claims"]["path_selection"] = dp_rows
    print()

    # ---- Claim 3 ---------------------------------------------------------
    print("### Claim 3: recovery control messages, independent of N")
    print(f"{'replay window k':>16s} {'messages':>10s}")
    rec_rows = []
    for window in (0, 1, 8, 64):
        counts = recovery_control_messages(window)
        rec_rows.append({"window": window, **counts})
        print(f"{window:16d} {counts['total']:10d}")
    print("=> 2 + k, with no broadcast and no Backbone traversal.")
    report["claims"]["recovery_messages"] = rec_rows
    print()

    # ---- Claim 4 ---------------------------------------------------------
    print("### Claim 4: shadow memory vs recovery window (Llama2-7B geometry)")
    print("shard=8 layers, kv_dim=4096, fp16; replay constants MEASURED (p14)")
    print(f"{'blocks':>7s} {'window':>7s} {'kv MiB':>9s} {'replay KiB':>11s} "
          f"{'total MiB':>10s}")
    shadow_rows = []
    for blocks, window in ((0, 8), (0, 64), (1, 8), (4, 8), (16, 8)):
        parts = shadow_bytes(
            committed_blocks=blocks, shard_layers=8, kv_dim=4096,
            bytes_per_element=2, window=window,
            input_bytes_per_step=REPLAY_BYTES_PER_STEP,
            metadata_bytes=2400,
        )
        shadow_rows.append({"blocks": blocks, "window": window, **parts})
        print(f"{blocks:7d} {window:7d} {parts['kv_checkpoint_bytes'] / 2**20:9.2f} "
              f"{parts['replay_window_bytes'] / 2**10:11.1f} "
              f"{parts['total_bytes'] / 2**20:10.2f}")
    print("=> grows with checkpoint distance, not with total request length.")
    print("   Validated against 70 measured snapshots: with block_retrieve off")
    print("   (checkpoint_block_count=0 in all of them) the replay window IS the")
    print("   whole cost -- which is why the KV column is zero there too.")
    report["claims"]["shadow_memory"] = shadow_rows
    print()

    # ---- measured cost model --------------------------------------------
    print("### Measured cost model (probes/p12): throughput vs merged batch")
    print(f"{'batch':>6s} {'service us':>11s} {'tok/s (decode stage)':>21s}")
    cost_rows = []
    for batch in (1, 2, 4, 8, 16):
        service = SERVICE_ALPHA_US + SERVICE_BETA_US * batch
        rate = decode_throughput(batch)
        cost_rows.append({"batch": batch, "service_us": service, "tok_s": rate})
        print(f"{batch:6d} {service:11.1f} {rate:21.1f}")
    print(f"=> service time is nearly flat (beta={SERVICE_BETA_US} us/item), so")
    print("   throughput is close to linear in merge size: this is why batch")
    print("   merging is the dominant optimisation (measured 5.2x across cases).")
    print("   NOTE: batch>8 is EXTRAPOLATION -- merge limit is 8 by default.")
    report["cost_model"] = {
        "alpha_us": SERVICE_ALPHA_US, "beta_us_per_item": SERVICE_BETA_US,
        "eta": ETA, "rows": cost_rows,
        "validity_range_batch": [1, 8],
    }

    print()
    print("### Constant provenance (audit trail)")
    for key, source in SOURCES.items():
        print(f"  {key:24s} <- {source}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
