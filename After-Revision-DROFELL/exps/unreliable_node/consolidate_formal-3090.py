#!/usr/bin/env python3
"""Build one strict formal matrix from explicitly selected run artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Any

from run import (
    EXP_ROOT,
    PAPER_PROBABILITIES,
    TOPOLOGIES,
    ExperimentError,
    graph_summary,
    legacy_throughput,
    probability_label,
    validate_keypoints,
    validate_topology,
    write_json,
)


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise ExperimentError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(value: Any) -> Any:
    return json.loads(json.dumps(value))


def network_spec_hash(path: Path) -> str:
    spec = load_json(path)
    declared_hash = spec.pop("spec_sha256", None)
    payload = json.dumps(
        spec, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    computed_hash = hashlib.sha256(payload).hexdigest()
    if declared_hash != computed_hash:
        raise ExperimentError(f"invalid declared network hash: {path}")
    return computed_hash


def has_uclamp_min_1024(record: dict[str, Any]) -> bool:
    return re.search(r"\bmin:\s*1024\b", record.get("uclamp", "")) is not None


def validate_source_provenance(
    environment: dict[str, Any], run_id: str
) -> dict[str, Any]:
    source_fingerprint_start = environment.get("source_fingerprint_start")
    source_fingerprint_end = environment.get("source_fingerprint_end")
    if source_fingerprint_start is None and source_fingerprint_end is None:
        return {
            "source_provenance": "legacy-unfingerprinted",
            "source_fingerprint_sha256": None,
        }
    if not isinstance(source_fingerprint_start, dict) or not isinstance(
        source_fingerprint_end, dict
    ):
        raise ExperimentError(f"incomplete source fingerprint for {run_id}")
    if source_fingerprint_start != source_fingerprint_end:
        raise ExperimentError(f"runtime source changed during {run_id}")
    source_fingerprint_sha256 = source_fingerprint_start.get(
        "aggregate_sha256"
    )
    if not re.fullmatch(r"[0-9a-f]{64}", str(source_fingerprint_sha256)):
        raise ExperimentError(f"invalid source fingerprint for {run_id}")
    return {
        "source_provenance": "fingerprinted",
        "source_fingerprint_sha256": source_fingerprint_sha256,
    }


def validate_source(
    run_id: str,
    topology: Any,
    deployment: str,
    probability: float,
) -> dict[str, Any]:
    environment_path = EXP_ROOT / "environment" / f"{run_id}.json"
    environment = load_json(environment_path)
    allowed_python_paths = {
        "/root/miniconda3/bin/python",
        "/home/dai/projects/LLM/dflash_related/.venv/bin/python",
    }
    if environment.get("python") not in allowed_python_paths:
        raise ExperimentError(
            f"unexpected Python for {run_id}: {environment.get('python')!r}"
        )
    if environment.get("drofell_metrics_path_present"):
        raise ExperimentError(f"metrics instrumentation polluted {run_id}")
    if environment.get("capture_metrics"):
        raise ExperimentError(f"metrics capture enabled for {run_id}")
    source_provenance = validate_source_provenance(environment, run_id)

    parameters = environment.get("parameters", {})
    expected_parameters = {
        "model": "teared-Llama-2-7B-GPTQ",
        "decoder_backend": "marlin",
        "cuda_graph": True,
        "failure_injection": True,
        "max_seq_len": 2048,
        "max_concurrency": 1,
        "formal_max_tokens": 1028,
        "warmup_tokens": 32,
        "sequences": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "logprobs": 0,
        "retrieve": False,
        "detail_trace": False,
        "disable_gc": True,
        "async_backend": "golang",
        "busy_wait": True,
        "uclamp_min": 1024,
        "drofell_metrics_path": None,
    }
    actual_parameters = {
        key: parameters.get(key) for key in expected_parameters
    }
    if actual_parameters != expected_parameters:
        raise ExperimentError(
            f"protocol mismatch in {environment_path}: {actual_parameters}"
        )
    topology_records = {
        item["name"]: item for item in parameters.get("topologies", [])
    }
    topology_record = topology_records.get(topology.name)
    expected_topology = normalized(asdict(topology))
    if topology_record is None:
        raise ExperimentError(
            f"topology snapshot mismatch in {environment_path}: {topology.name}"
        )
    actual_probabilities = topology_record.pop("probabilities", [])
    expected_topology.pop("probabilities", None)
    if topology_record != expected_topology or probability not in actual_probabilities:
        raise ExperimentError(
            f"topology snapshot mismatch in {environment_path}: {topology.name}"
        )

    probability_text = str(probability)
    purpose = f"formal:{deployment}:{topology.name}:p={probability_text}"
    benchmark_commands = [
        command for command in environment.get("commands", [])
        if command.get("purpose") == purpose
    ]
    if len(benchmark_commands) != 1:
        raise ExperimentError(
            f"expected one benchmark command {purpose!r} in {environment_path}"
        )
    benchmark_command = benchmark_commands[0]
    if benchmark_command.get("returncode") != 0:
        raise ExperimentError(
            f"selected benchmark command did not exit cleanly: {run_id}/{purpose}"
        )
    if not has_uclamp_min_1024(benchmark_command):
        raise ExperimentError(
            f"benchmark uclamp evidence missing: {run_id}/{purpose}"
        )
    network_purpose = f"network:formal-{deployment}-{topology.name}"
    network_commands = [
        command for command in environment.get("commands", [])
        if command.get("purpose") == network_purpose
    ]
    if len(network_commands) != 1 or not has_uclamp_min_1024(
        network_commands[0]
    ):
        raise ExperimentError(
            f"network uclamp evidence missing: {run_id}/{network_purpose}"
        )
    return {
        "environment": str(environment_path.relative_to(EXP_ROOT)),
        "environment_sha256": sha256_file(environment_path),
        **source_provenance,
        "benchmark_command": benchmark_command["command"],
        "network_command": network_commands[0]["command"],
    }


def validate_network_spec(
    run_id: str,
    topology: Any,
    phases: dict[str, Any],
) -> tuple[str, str]:
    preferred_spec_path = (
        EXP_ROOT / "results" / run_id / "network_specs"
        / f"{topology.name}-graph-failure.json"
    )
    expected_hash = phases.get("network_spec_hash")
    candidates = [preferred_spec_path]
    if not preferred_spec_path.is_file():
        candidates.extend(sorted(
            (EXP_ROOT / "results").glob(
                f"*/network_specs/{topology.name}-graph-failure.json"
            )
        ))
    spec_path = next(
        (
            path for path in candidates
            if path.is_file() and network_spec_hash(path) == expected_hash
        ),
        None,
    )
    if spec_path is None:
        raise ExperimentError(
            f"no immutable network spec matches {run_id}/{topology.name} "
            f"hash {expected_hash!r}"
        )
    spec = load_json(spec_path)
    spec_hash = network_spec_hash(spec_path)
    if expected_hash != spec_hash:
        raise ExperimentError(f"network hash mismatch: {spec_path}")
    model = spec.get("model", {})
    if model.get("num_decoders") != 32:
        raise ExperimentError(f"wrong decoder count: {spec_path}")
    if model.get("decoder_group_sizes") != list(topology.group_sizes):
        raise ExperimentError(f"wrong decoder groups: {spec_path}")
    actual_dealers = {
        dealer: tuple(record.get("group_ids", []))
        for dealer, record in spec.get("dealers", {}).items()
    }
    if actual_dealers != dict(topology.dealers):
        raise ExperimentError(f"wrong dealer mapping: {spec_path}")
    return str(spec_path.relative_to(EXP_ROOT)), spec_hash


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["keypoint_coverage"] = json.dumps(
                row["keypoint_coverage"]
            )
            encoded["source_runs"] = json.dumps(row["source_runs"])
            writer.writerow(encoded)


def case_rows(
    *,
    topology: Any,
    deployment: str,
    probability: float,
    sources: list[Any],
) -> list[dict[str, Any]]:
    rows = []
    label = probability_label(probability)
    for source in sources:
        if isinstance(source, str):
            run_id = source
            selected_repeats = None
        else:
            run_id = str(source["run_id"])
            selected_repeats = {int(value) for value in source["repeats"]}
        source_evidence = validate_source(
            run_id, topology, deployment, probability
        )
        cell_root = (
            EXP_ROOT / "results" / run_id / deployment / topology.name
            / f"drop-{label}"
        )
        if not cell_root.is_dir():
            raise ExperimentError(f"missing selected cell: {cell_root}")
        repeat_dirs = sorted(
            cell_root.glob("repeat-*"),
            key=lambda path: int(path.name.removeprefix("repeat-")),
        )
        if selected_repeats is not None:
            repeat_dirs = [
                path for path in repeat_dirs
                if int(path.name.removeprefix("repeat-")) in selected_repeats
            ]
            found = {
                int(path.name.removeprefix("repeat-")) for path in repeat_dirs
            }
            if found != selected_repeats:
                raise ExperimentError(
                    f"selected repeats {sorted(selected_repeats)} not found "
                    f"in {cell_root}"
                )
        for repeat_dir in repeat_dirs:
            result_path = repeat_dir / "result.json"
            tokens_path = repeat_dir / "tokens.json"
            phases_path = repeat_dir / "phases.json"
            graph_path = repeat_dir / "decode_graph.json"
            record = load_json(result_path)
            tokens = load_json(tokens_path)
            phases = load_json(phases_path)
            graph_raw = load_json(graph_path)
            graph = graph_summary(graph_raw)
            if phases.get("deployment") != deployment:
                raise ExperimentError(
                    f"deployment mismatch in {repeat_dir / 'phases.json'}"
                )
            coverage = validate_keypoints(phases["keypoints"], topology)
            if len(record) != 1027:
                raise ExperimentError(
                    f"expected 1027 timestamps in {repeat_dir}, got {len(record)}"
                )
            if any(
                not isinstance(item, list) or len(item) != 2 or item[0] != 0
                for item in record
            ):
                raise ExperimentError(f"invalid timestamp ledger in {result_path}")
            timestamps = [int(item[1]) for item in record]
            if any(
                current <= previous
                for previous, current in zip(timestamps, timestamps[1:])
            ):
                raise ExperimentError(f"non-monotonic timestamps in {result_path}")
            if len(tokens) != 1 or len(tokens[0]) != 1028:
                raise ExperimentError(f"invalid token ledger in {tokens_path}")
            expected_request_parameters = {
                "model_name": "teared-Llama-2-7B-GPTQ",
                "prompt": "once upon a time",
                "max_tokens": 1028,
                "warmup_tokens": 32,
                "parallel_seqs": 1,
                "num_paths": topology.num_paths,
                "allow_provider_reuse": False,
                "retrieve": False,
                "detail_trace": False,
                "logprobs": 0,
                "progress_interval": 0.0,
                "disable_gc": True,
                "adapter": "base",
            }
            request_parameters = phases.get("parameters", {})
            if {
                key: request_parameters.get(key)
                for key in expected_request_parameters
            } != expected_request_parameters:
                raise ExperimentError(
                    f"request protocol mismatch in {phases_path}"
                )
            if phases.get("network_name") != (
                f"unreliable-{topology.name}-graph-failure"
            ):
                raise ExperimentError(f"network name mismatch in {phases_path}")
            first_tokens = phases.get("first_token_ns", [])
            phase_times = [
                int(phases["request_start_ns"]),
                int(phases["prefill_end_ns"]),
                *(int(value) for value in first_tokens),
                int(phases["end_ns"]),
            ]
            if len(first_tokens) != 1 or any(
                current < previous
                for previous, current in zip(phase_times, phase_times[1:])
            ):
                raise ExperimentError(f"invalid phase timestamps in {phases_path}")
            spec_path, spec_hash = validate_network_spec(
                run_id, topology, phases
            )
            if not graph["decoder_enabled"] or not graph["decoder_captured"]:
                raise ExperimentError(f"CUDA graph incomplete in {repeat_dir}")
            if graph["decoder_replays"] <= 0 or graph["fallbacks"] != 0:
                raise ExperimentError(f"invalid graph counters in {repeat_dir}")
            if set(graph_raw) != {dealer for dealer, _ in topology.dealers}:
                raise ExperimentError(f"graph dealer mismatch in {graph_path}")
            rows.append({
                "deployment": deployment,
                "topology": topology.name,
                "probability": probability,
                "source_run": run_id,
                "source_repeat": int(
                    repeat_dir.name.removeprefix("repeat-")
                ),
                "throughput_128": legacy_throughput(record, 128),
                "throughput_1024": legacy_throughput(record, 1024),
                "timestamps": len(record),
                "graph_dealers": graph["dealer_count"],
                "graph_decoder_replays": graph["decoder_replays"],
                "graph_fallback_records": graph["fallbacks"],
                "keypoint_coverage": coverage,
                "network_spec": spec_path,
                "network_spec_sha256": spec_hash,
                "result": str(result_path.relative_to(EXP_ROOT)),
                "result_sha256": sha256_file(result_path),
                "tokens_sha256": sha256_file(tokens_path),
                "phases_sha256": sha256_file(phases_path),
                "decode_graph_sha256": sha256_file(graph_path),
                **source_evidence,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-id", required=True)
    args = parser.parse_args()
    if not args.output_id or "/" in args.output_id or args.output_id in {".", ".."}:
        parser.error("--output-id must be one path component")

    manifest = load_json(args.manifest)
    repeats = int(manifest["repeats"])
    topologies = {topology.name: topology for topology in TOPOLOGIES}
    allowed_probabilities = set(PAPER_PROBABILITIES)
    rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []

    for topology_name, deployments in manifest["cells"].items():
        if topology_name not in topologies:
            raise ExperimentError(f"unknown topology: {topology_name}")
        topology = topologies[topology_name]
        validate_topology(topology)
        for deployment, probabilities in deployments.items():
            if deployment not in {"combined-request-agent", "persistent-agent"}:
                raise ExperimentError(f"unknown deployment: {deployment}")
            for probability_text, source_runs in probabilities.items():
                probability = float(probability_text)
                if probability not in allowed_probabilities:
                    raise ExperimentError(
                        f"probability outside formal grid: {probability}"
                    )
                if deployment == "persistent-agent" and probability != 0.0:
                    raise ExperimentError("persistent Agent is valid only for p=0")
                selected = case_rows(
                    topology=topology,
                    deployment=deployment,
                    probability=probability,
                    sources=list(source_runs),
                )
                if len(selected) != repeats:
                    raise ExperimentError(
                        f"{topology_name}/{deployment}/p={probability} has "
                        f"{len(selected)} selected repeats, expected {repeats}"
                    )
                rows.extend(selected)
                rates_128 = [row["throughput_128"] for row in selected]
                rates_1024 = [row["throughput_1024"] for row in selected]
                aggregates.append({
                    "deployment": deployment,
                    "topology": topology_name,
                    "probability": probability,
                    "repeat_count": repeats,
                    "throughput_128_mean": statistics.mean(rates_128),
                    "throughput_128_min": min(rates_128),
                    "throughput_128_max": max(rates_128),
                    "throughput_1024_mean": statistics.mean(rates_1024),
                    "throughput_1024_min": min(rates_1024),
                    "throughput_1024_max": max(rates_1024),
                    "keypoint_coverage": selected[0]["keypoint_coverage"],
                    "source_runs": [
                        source if isinstance(source, str) else source["run_id"]
                        for source in source_runs
                    ],
                })

    provenance_by_run: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_run = row["source_run"]
        provenance = {
            "status": row["source_provenance"],
            "source_fingerprint_sha256": row[
                "source_fingerprint_sha256"
            ],
        }
        previous = provenance_by_run.setdefault(source_run, provenance)
        if previous != provenance:
            raise ExperimentError(
                f"inconsistent source provenance for {source_run}"
            )
    legacy_source_runs = sorted(
        run_id for run_id, provenance in provenance_by_run.items()
        if provenance["status"] == "legacy-unfingerprinted"
    )
    fingerprinted_source_runs = {
        run_id: provenance["source_fingerprint_sha256"]
        for run_id, provenance in sorted(provenance_by_run.items())
        if provenance["status"] == "fingerprinted"
    }
    provenance_warning = (
        "This consolidated matrix contains legacy-unfingerprinted runs. "
        "Their raw artifacts, topology, request protocol, CUDA graph replay, "
        "and zero-fallback conditions passed strict validation, but their "
        "exact runtime source tree cannot be cryptographically established."
        if legacy_source_runs else None
    )

    output = EXP_ROOT / "results" / args.output_id
    if output.exists():
        raise ExperimentError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)
    summary = {
        "schema_version": 2,
        "metric": "paper-compatible r/(t_r-t_1)",
        "protocol": (
            "Each formal cell is assembled only from the manifest-selected "
            f"raw artifacts. Every cell requires exactly {repeats} complete repeats, "
            "strict topology coverage, captured CUDA graphs, and zero fallback. "
            "Source provenance is reported independently and does not waive "
            "artifact validation."
        ),
        "provenance": {
            "warning": provenance_warning,
            "legacy_source_runs": legacy_source_runs,
            "fingerprinted_source_runs": fingerprinted_source_runs,
        },
        "manifest": str(args.manifest.resolve()),
        "repeats": repeats,
        "runs": rows,
        "aggregates": aggregates,
    }
    write_json(output / "summary.json", summary)
    write_csv(output / "summary.csv", aggregates)
    print(json.dumps({
        "output": str(output),
        "run_rows": len(rows),
        "aggregate_rows": len(aggregates),
    }, indent=2))


if __name__ == "__main__":
    main()
