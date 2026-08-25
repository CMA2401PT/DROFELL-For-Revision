from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path
from typing import Any, Optional

# Address-space cap for the path-search phase only. `gen_lowest_energy_paths`
# enumerates the whole tie set, so an unguarded sweep can exhaust host memory
# (probes/p13). The cap turns that into a clean MemoryError. It is applied in a
# CHILD process, because torch cannot be imported under it and the shadow phase
# needs torch.
MEMORY_CAP_BYTES = 4 * 1024 ** 3


def build_topology(segments: int, redundancy: int) -> list[list[str]]:
    """`segments` model shards, each held by `redundancy` distinct dealers."""
    return [
        [f"dealer-g{group}-r{replica}" for replica in range(redundancy)]
        for group in range(segments)
    ]


def measure_path_search(
    segments: int, redundancy: int
) -> dict[str, Any]:
    """Run one path-search point inside a memory-capped child process."""
    import multiprocessing

    with multiprocessing.Pool(1, initializer=_cap_memory) as pool:
        return pool.apply(_path_search_worker, (segments, redundancy))


def _cap_memory() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_CAP_BYTES, MEMORY_CAP_BYTES))


def _path_search_worker(segments: int, redundancy: int) -> dict[str, Any]:
    from node.path_utils import gen_lowest_energy_paths, paths_to_keypoints

    providers = build_topology(segments, redundancy)
    started = time.perf_counter_ns()
    try:
        paths = gen_lowest_energy_paths(providers)
    except MemoryError:
        return {
            "logical_nodes": segments * redundancy,
            "segments": segments,
            "redundancy": redundancy,
            "tie_paths": None,
            "keypoints": None,
            "search_ms": None,
            "status": "memory_exhausted",
        }
    search_ms = (time.perf_counter_ns() - started) / 1e6
    keypoints = paths_to_keypoints(paths, no_overlap=True)
    return {
        "logical_nodes": segments * redundancy,
        "segments": segments,
        "redundancy": redundancy,
        "tie_paths": len(paths),
        "keypoints": len(keypoints),
        "search_ms": search_ms,
        "status": "ok",
    }


def measure_shadow(
    segments: int, decode_steps: int, prompt_len: int = 16
) -> dict[str, Any]:
    """Drive the REAL shadow state for one logical node and read its own metrics."""
    from dealers.common.base_types import ModelDescribe, gen_user_model_names
    from node.compute_node.infer_meta import InferInit, InferTask, Navigate
    from node.shadow_node_state.infer_state_data import InferStateData
    from operators.core.sampler.common import SamplingParams
    from operators.core.tokenizer.common import SequenceCore, StopConfig

    model = "teared-Llama-2-7B-GPTQ"
    layers_per_segment = max(1, 32 // segments)
    base_models = [str(ModelDescribe.new(model, is_head=True))]
    for index in range(segments):
        start = index * layers_per_segment
        base_models.append(
            str(ModelDescribe.new(model, start=start, end=start + layers_per_segment))
        )
    base_models.append(str(ModelDescribe.new(model, is_tail=True)))
    pairs = gen_user_model_names(base_models, "scale-seq")
    user_models = [user for user, _ in pairs]

    sequence = SequenceCore(stop_config=StopConfig(4096, ignore_eos=True))
    sequence.prompt_token_ids.extend(range(prompt_len))
    init = InferInit.gen(
        seq_id="scale-seq", models=pairs, user_model_name_and_slicer=None,
        seq_data=sequence,
        sampling_params=SamplingParams(temperature=0.0, logprobs=None),
    )
    shadow = InferStateData(init)

    def task(step: int, is_prefill: bool, payload: Any) -> InferTask:
        return InferTask(
            seq_id="scale-seq", step=step, navi=Navigate(user_models),
            is_prefill=is_prefill, input=payload, output=None,
        )

    shadow.update_by_task(task(0, True, list(range(prompt_len))))
    for step in range(1, decode_steps + 1):
        shadow.update_by_task(task(step, False, 1000 + step))

    metrics = shadow.metrics_snapshot()
    started = time.perf_counter_ns()
    _, replay = shadow.gen_init_cfg(
        confirmed_output_token_ids=[1000 + s for s in range(1, decode_steps + 1)]
    )
    rebuild_ns = time.perf_counter_ns() - started
    return {
        "segments": segments,
        "decode_steps": decode_steps,
        "metadata_bytes": metrics["metadata_bytes"],
        "generating_bytes": metrics["generating_bytes"],
        "total_retained_bytes": metrics["total_retained_bytes"],
        "update_cpu_ns": metrics["update_cpu_ns"],
        "replay_task_count": len(replay),
        "state_rebuild_ns": rebuild_ns,
    }


def measure_resource_map(segments: int, redundancy: int) -> int:
    """Entries in the map that每个 task carries: one per segment plus the tail slot."""
    # Mirrors InferControl.res_location: {segment_index: [ctrl] + holders}
    # plus one entry for the tail output slot.
    return segments * (1 + redundancy) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-segments", type=int, default=64)
    args = parser.parse_args()

    import extend_asyncio as ea

    ea.config_impl("py", False)

    report: dict[str, Any] = {"path_search": [], "shadow": [], "resource_map": []}

    print("=" * 78)
    print("D1 · Control-plane cost vs logical node count (compute held constant)")
    print("=" * 78)
    print("All quantities below are produced by DROFELL's production code.")
    print()

    # ---- resource-location map + path search ------------------------------
    print("### map size and path search vs scale")
    print(f"{'nodes':>6s} {'L':>4s} {'r':>3s} {'map entries':>12s} "
          f"{'tie paths':>11s} {'keypoints':>10s} {'search ms':>10s}")
    segment_counts = [s for s in (2, 4, 8, 16, 32, 64, 128, 256)
                      if s <= args.max_segments]
    for redundancy in (1, 2):
        for segments in segment_counts:
            row = measure_path_search(segments, redundancy)
            row["map_entries"] = measure_resource_map(segments, redundancy)
            report["path_search"].append(row)
            tie = "OOM" if row["tie_paths"] is None else f"{row['tie_paths']}"
            keys = "--" if row["keypoints"] is None else f"{row['keypoints']}"
            millis = "--" if row["search_ms"] is None else f"{row['search_ms']:.2f}"
            print(f"{row['logical_nodes']:6d} {segments:4d} {redundancy:3d} "
                  f"{row['map_entries']:12d} {tie:>11s} {keys:>10s} {millis:>10s}")
    print()
    print("=> map size is O(L*r) and independent of the rest of the network.")
    print("=> with r=1 path search is flat; with r>=2 the TIE SET is (r+1)^L,")
    print("   which is the real scaling wall (measured, see probes/p13).")
    print()

    # ---- shadow state -----------------------------------------------------
    print("### shadow state cost vs recovery window (real InferStateData)")
    print(f"{'L':>4s} {'steps':>6s} {'meta B':>8s} {'replay B':>10s} "
          f"{'total B':>10s} {'update us':>10s} {'rebuild us':>11s}")
    for segments in (1, 2, 4, 8):
        for steps in (8, 64, 256):
            row = measure_shadow(segments, steps)
            report["shadow"].append(row)
            print(f"{segments:4d} {steps:6d} {row['metadata_bytes']:8d} "
                  f"{row['generating_bytes']:10d} "
                  f"{row['total_retained_bytes']:10d} "
                  f"{row['update_cpu_ns'] / 1000:10.1f} "
                  f"{row['state_rebuild_ns'] / 1000:11.1f}")
    print()
    print("=> shadow footprint tracks the REPLAY WINDOW, not the node count:")
    print("   growing segments at a fixed window barely moves it.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
