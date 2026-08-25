#!/usr/bin/env python3
"""Run one isolated DROFELL benchmark case against a serving session."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from session import (
    AGENT,
    EXP_ROOT,
    NSPREFIX,
    PORT,
    REPO_ROOT,
    TEST_NODE_ID,
    SessionError,
    case_root,
    cleanup_namespace,
    load_manifest,
    manifest_path,
    net_cluster_env,
    proc_start_ticks,
    serving_alive,
    utc_now,
    write_json,
    require_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="workload", required=True)

    inference = commands.add_parser("inference")
    add_common(inference)
    inference.add_argument("--max-tokens", type=int, required=True)
    inference.add_argument("--warmup-tokens", type=int, default=16)

    training = commands.add_parser("finetuning")
    add_common(training)
    training.add_argument("--minibatch-size", type=int, required=True)
    training.add_argument("--effective-batch", type=int, required=True)
    training.add_argument("--samples-per-instance", type=int, default=128)
    training.add_argument("--sample-len", type=int, default=128)
    training.add_argument("--backend", choices=("exllama", "exllama_fuse", "marlin"), default="exllama_fuse")

    return parser.parse_args()


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--retry", action="store_true")


def case_directory(args: argparse.Namespace) -> Path:
    root = case_root(args.session_id) / args.workload / f"clients-{args.clients}"
    if args.workload == "inference":
        return root / f"tokens-{args.max_tokens - 2}" / f"repeat-{args.repeat}"
    return root / f"batch-{args.effective_batch}" / f"repeat-{args.repeat}"


def archive_existing(path: Path) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archived = path.parent / "failed_attempts" / stamp / path.name
    archived.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(archived))


def command_for(args: argparse.Namespace, manifest: dict, output: Path) -> list[str]:
    network = manifest["network"]
    orbit = str(network["orbit"])
    python = str(manifest["python"])
    deployment = str(manifest["deployment"])
    model = str(manifest["model"])
    host_prefix = f"{args.workload}-{manifest['session_id']}-c{args.clients}"
    no_netns = bool(network.get("no_netns", False))
    advertise_host = (
        "127.0.0.1"
        if no_netns
        else f"{network.get('subnet', '10.203.0')}.{TEST_NODE_ID}"
    )
    env_command = [] if no_netns else [
        str(REPO_ROOT / "scripts/online/benchmark/net_cluster"), "run"
    ]
    command: list[str]
    if args.workload == "inference":
        command = [
            *env_command,
            "/usr/bin/uclampset", "-m", "1024", "--", python,
            "-m", "scripts.online.benchmark.benchmark_infer",
            "--measurement", "request", "--orbit", orbit,
            "--host-id", f"{host_prefix}-r{args.repeat}",
            "--advertise-host", advertise_host,
            "--model", model, "--output", str(output),
            "--seqs", str(args.clients),
            "--max-tokens", str(args.max_tokens),
            "--warmup-tokens", str(args.warmup_tokens),
            "--num-paths", "1", "--logprobs", "0", "--disable-gc",
            "--async-backend", "golang", "--busy-wait",
        ]
        if deployment == "persistent-agent":
            command += ["--agent", AGENT]
    else:
        grad_accum = args.effective_batch // args.minibatch_size
        command = [
            *env_command,
            "/usr/bin/uclampset", "-m", "1024", "--", python,
            "-m", "scripts.online.benchmark.benchmark_train",
            "--orbit", orbit,
            "--host-id", f"{host_prefix}-b{args.effective_batch}-r{args.repeat}",
            "--advertise-host", advertise_host,
            "--model", model, "--output", str(output),
            "--instances", str(args.clients),
            "--samples-per-instance", str(args.samples_per_instance),
            "--minibatch-size", str(args.minibatch_size),
            "--grad-accum-steps", str(grad_accum),
            "--sample-len", str(args.sample_len),
            "--adapter", "ptuning", "--ptuning-prefix-len", "32",
            "--adapter-dropout", "0.1", "--learning-rate", "1e-2",
            "--ckpt-period", "16", "--detail-trace",
            "--async-backend", "golang", "--busy-wait",
        ]
        if deployment == "persistent-agent":
            command += ["--agent", AGENT]
    return command


def required_artifacts(workload: str) -> tuple[str, ...]:
    if workload == "inference":
        return ("args.json", "result.json", "tokens.json", "phases.json", "decode_graph.json")
    return ("args.json", "results.json", "network.json")


def kill_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_case(args: argparse.Namespace) -> int:
    require_root()
    if args.clients <= 0 or args.repeat < 0 or args.timeout <= 0:
        raise SessionError("clients must be positive, repeat non-negative, timeout positive")
    manifest = load_manifest(args.session_id)
    if manifest.get("state") != "serving":
        raise SessionError(f"serving session is not active: {manifest.get('state')}")
    if manifest.get("workload") != args.workload:
        raise SessionError(
            f"session workload is {manifest.get('workload')!r}, requested {args.workload!r}"
        )
    alive, missing = serving_alive(manifest)
    if not alive:
        raise SessionError(f"serving session is unhealthy: {', '.join(missing)}")
    if args.workload == "inference" and args.max_tokens not in {130, 2050}:
        raise SessionError("inference --max-tokens must be 130 or 2050")
    if args.workload == "finetuning":
        configured_backend = manifest.get("train_backend", "exllama_fuse")
        if args.backend != configured_backend:
            raise SessionError(
                "benchmark backend does not match serving session: "
                f"session={configured_backend!r}, requested={args.backend!r}"
            )
        if args.minibatch_size <= 0 or args.effective_batch <= 0:
            raise SessionError("batch sizes must be positive")
        if args.effective_batch % args.minibatch_size:
            raise SessionError("effective batch must be divisible by minibatch size")
        if args.samples_per_instance < 2 * args.minibatch_size:
            raise SessionError("samples-per-instance must contain at least two minibatches")

    output = case_directory(args)
    if output.exists():
        if not args.retry:
            raise SessionError(f"case output exists; use --retry: {output}")
        archive_existing(output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "benchmark.log"
    command = command_for(args, manifest, output)
    env = net_cluster_env(manifest, TEST_NODE_ID)
    write_json(output / "case.json", {
        "schema_version": 1,
        "session_id": args.session_id,
        "workload": args.workload,
        "clients": args.clients,
        "repeat": args.repeat,
        "command": " ".join(command),
        "started_at": utc_now(),
    })
    cleanup_namespace(manifest, TEST_NODE_ID)
    process: subprocess.Popen[bytes] | None = None
    timed_out = False
    returncode: int | None = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_group(process)
                returncode = process.returncode
    except OSError as error:
        write_json(output / "failure.json", {
            "kind": "launch_error", "error": repr(error), "finished_at": utc_now(),
        })
        return 1
    finally:
        cleanup_namespace(manifest, TEST_NODE_ID)

    missing_artifacts = [name for name in required_artifacts(args.workload) if not (output / name).is_file()]
    if timed_out or returncode != 0 or missing_artifacts:
        write_json(output / "failure.json", {
            "kind": "timeout" if timed_out else "benchmark_failure",
            "returncode": returncode,
            "missing_artifacts": missing_artifacts,
            "finished_at": utc_now(),
            "log": str(log_path),
        })
        print(f"case failed: {output}", file=sys.stderr)
        return 1
    write_json(output / "complete.json", {
        "schema_version": 1,
        "session_id": args.session_id,
        "workload": args.workload,
        "clients": args.clients,
        "repeat": args.repeat,
        "artifacts": list(required_artifacts(args.workload)),
        "finished_at": utc_now(),
    })
    print(f"case complete: {output}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run_case(args)
    except SessionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
