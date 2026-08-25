#!/usr/bin/env python3
"""Start and manage a persistent DROFELL serving network.

This entrypoint deliberately does not launch a benchmark node.  A separate
run_benchmark.py process can attach node 10 to the serving network repeatedly.
Run this program as root (normally through ``sudo -E``).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import signal
import shutil
import sys
import time

from serving_runtime import ExperimentError, ServingRuntime
from session import (
    BRIDGE,
    NSPREFIX,
    ORBIT_NODE_ID,
    PORT,
    SCHEMA_VERSION,
    SUBNET,
    SessionError,
    cleanup_all_network,
    kill_process_group,
    load_manifest,
    manifest_path,
    proc_start_ticks,
    read_json,
    require_root,
    serving_alive,
    session_dir,
    utc_now,
    write_json,
    model_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start", help="start Orbit and serving dealers")
    start.add_argument("--session-id", required=True)
    start.add_argument("--python", type=Path, required=True)
    start.add_argument("--model-size", choices=("7b", "70b"), default="70b")
    start.add_argument("--workload", choices=("inference", "finetuning"), required=True)
    start.add_argument("--cards", type=int, choices=(2, 4, 8), default=2)
    start.add_argument(
        "--deployment",
        choices=("combined-request-agent", "persistent-agent"),
        default="combined-request-agent",
    )
    start.add_argument("--bandwidth-mbit", type=int, required=True)
    start.add_argument("--target-rtt-ms", type=float, required=True)
    start.add_argument(
        "--no-netns", action="store_true",
        help="run unshaped loopback serving without namespaces (autodl)",
    )

    for command in ("status", "stop"):
        item = commands.add_parser(command)
        item.add_argument("--session-id", required=True)
        if command == "stop":
            item.add_argument("--force", action="store_true")
    return parser.parse_args()


def process_entry(pid: int, *, role: str, namespace: str | None, log: Path) -> dict:
    start_ticks = proc_start_ticks(pid)
    if start_ticks is None:
        raise SessionError(f"cannot inspect serving process {pid}")
    return {
        "pid": pid,
        "pgid": os.getpgid(pid),
        "start_ticks": start_ticks,
        "role": role,
        "namespace": namespace,
        "log": str(log),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_runner(args: argparse.Namespace) -> ServingRuntime:
    runner = ServingRuntime(
        python=args.python,
        run_id=args.session_id,
        model_size=args.model_size,
        cards=args.cards,
        workload=args.workload,
        deployment=args.deployment,
        bandwidth_mbit=args.bandwidth_mbit,
        target_rtt_ms=args.target_rtt_ms,
        no_netns=args.no_netns,
        train_backend="exllama_fuse",
    )
    # sudo may reset PATH, while ninja is installed in the experiment user's
    # local bin directory.  Network generation invokes it through a CUDA
    # extension build, so make that dependency explicit for root runs.
    path_entries = [
        "/home/dai/.local/bin",
        "/root/.local/bin",
        str(args.python.parent),
        *runner.base_env.get("PATH", "").split(os.pathsep),
    ]
    unique_path: list[str] = []
    for item in path_entries:
        try:
            is_dir = bool(item) and Path(item).is_dir()
        except OSError:
            is_dir = False
        if item and item not in unique_path and is_dir:
            unique_path.append(item)
    runner.base_env["PATH"] = os.pathsep.join(unique_path)
    if shutil.which("ninja", path=runner.base_env["PATH"]) is None:
        raise SessionError(
            "ninja is required for network generation but is not available "
            f"on PATH={runner.base_env['PATH']}"
        )
    return runner


def start(args: argparse.Namespace) -> int:
    require_root()
    if args.bandwidth_mbit <= 0 or args.target_rtt_ms < 0:
        raise SessionError("bandwidth must be positive and RTT must be non-negative")
    if args.model_size == "7b" and args.workload != "inference":
        raise SessionError("the throughput target only defines 7B inference")
    existing = manifest_path(args.session_id)
    if existing.is_file():
        try:
            previous = read_json(existing)
            alive, missing = serving_alive(previous)
            if previous.get("state") == "serving" and alive:
                raise SessionError(
                    f"serving session is already active: {args.session_id}"
                )
        except SessionError:
            raise
        except Exception as error:
            raise SessionError(f"cannot inspect existing session {existing}: {error}") from error

    runner = make_runner(args)
    network = runner.network()
    orbit_host = "127.0.0.1" if args.no_netns else f"{SUBNET}.{ORBIT_NODE_ID}"
    session_dir(args.session_id).mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "schema_version": SCHEMA_VERSION,
        "session_id": args.session_id,
        "state": "starting",
        "created_at": utc_now(),
        "workload": args.workload,
        "deployment": args.deployment,
        "train_backend": runner.train_backend,
        "model_size": args.model_size,
        "model": str(model_spec(args.model_size)["name"]),
        "model_decoder_layers": int(model_spec(args.model_size)["decoder_layers"]),
        "python": str(args.python),
        "cards": args.cards,
        "network_spec": str(network),
        "network_spec_sha256": None,
        "network": {
            "bandwidth_mbit": args.bandwidth_mbit,
            "target_rtt_ms": args.target_rtt_ms,
            "bridge": BRIDGE,
            "subnet": SUBNET,
            "namespace_prefix": NSPREFIX,
            "no_netns": args.no_netns,
            "orbit": f"{orbit_host}:{PORT}",
        },
        "processes": {"orbit": None, "dealers": {}},
        "result_root": str(runner.result_root),
        "log_root": str(runner.log_root),
    }
    write_json(manifest_path(args.session_id), manifest)

    try:
        runner.generate_network()
        manifest["network_spec_sha256"] = sha256_file(network)
        runner.launch_network(network)
        phase_logs = runner.card_root(runner.log_root, args.cards) / args.workload / args.deployment
        processes = runner.processes
        if len(processes) != args.cards + 1:
            raise SessionError(
                f"expected Orbit plus {args.cards} dealers, got {len(processes)}"
            )
        manifest["processes"]["orbit"] = process_entry(
            processes[0].pid,
            role="orbit",
            namespace=None if args.no_netns else f"{NSPREFIX}{ORBIT_NODE_ID}",
            log=phase_logs / "orbit.log",
        )
        manifest["processes"]["dealers"] = {
            runner.dealer_name(index): process_entry(
                processes[index + 1].pid,
                role=runner.dealer_name(index),
                namespace=(
                    None
                    if args.no_netns
                    else f"{NSPREFIX}{index + 1}"
                ),
                log=phase_logs / f"{runner.dealer_name(index)}.log",
            )
            for index in range(args.cards)
        }
        manifest["state"] = "serving"
        manifest["ready_at"] = utc_now()
        write_json(manifest_path(args.session_id), manifest)
        print(f"serving session ready: {args.session_id}")
        print(f"manifest: {manifest_path(args.session_id)}")
        print(f"orbit: {manifest['network']['orbit']}")

        stop_requested = False

        def request_stop(signum: int, _frame: object) -> None:
            nonlocal stop_requested
            stop_requested = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        while not stop_requested:
            # An external ``stop`` writes ``stopping`` before killing the
            # serving process groups.  Observe that state so this supervisor
            # does not overwrite the final ``stopped`` state with ``degraded``
            # while it notices the intentionally removed PIDs.
            try:
                external_state = read_json(manifest_path(args.session_id)).get("state")
            except SessionError:
                external_state = None
            if external_state in {"stopping", "stopped"}:
                stop_requested = True
                continue
            alive, missing = serving_alive(manifest)
            if not alive:
                manifest["state"] = "degraded"
                manifest["degraded_at"] = utc_now()
                manifest["missing_processes"] = missing
                write_json(manifest_path(args.session_id), manifest)
                print(f"serving session degraded: missing {', '.join(missing)}", file=sys.stderr)
                # A crashed dealer must not leave bridge/namespaces behind.
                runner.stop_network()
                return 1
            time.sleep(2)
        manifest["state"] = "stopping"
        write_json(manifest_path(args.session_id), manifest)
        runner.stop_network()
        manifest["state"] = "stopped"
        manifest["stopped_at"] = utc_now()
        write_json(manifest_path(args.session_id), manifest)
        return 0
    except BaseException as error:
        manifest["state"] = "failed"
        manifest["failed_at"] = utc_now()
        manifest["error"] = repr(error)
        write_json(manifest_path(args.session_id), manifest)
        runner.stop_network()
        raise


def status(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.session_id)
    alive, missing = serving_alive(manifest)
    result = {
        "session_id": args.session_id,
        "state": manifest.get("state"),
        "processes_alive": alive,
        "missing_processes": missing,
        "manifest": str(manifest_path(args.session_id)),
    }
    print(__import__("json").dumps(result, indent=2, ensure_ascii=False))
    return 0 if alive and manifest.get("state") == "serving" else 1


def stop(args: argparse.Namespace) -> int:
    require_root()
    manifest = load_manifest(args.session_id)
    alive, missing = serving_alive(manifest)
    if not alive and manifest.get("state") not in {"starting", "serving", "degraded"} and not args.force:
        print(f"session is already stopped: {args.session_id}")
        return 0
    manifest["state"] = "stopping"
    manifest["stopping_at"] = utc_now()
    write_json(manifest_path(args.session_id), manifest)
    for entry in list(manifest.get("processes", {}).get("dealers", {}).values())[::-1]:
        if isinstance(entry, dict):
            kill_process_group(entry)
    orbit = manifest.get("processes", {}).get("orbit")
    if isinstance(orbit, dict):
        kill_process_group(orbit)
    cleanup_all_network(manifest)
    manifest["state"] = "stopped"
    manifest["stopped_at"] = utc_now()
    manifest["missing_processes"] = missing
    write_json(manifest_path(args.session_id), manifest)
    print(f"serving session stopped: {args.session_id}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.command == "start":
            return start(args)
        if args.command == "status":
            return status(args)
        if args.command == "stop":
            return stop(args)
        raise SessionError(f"unsupported command: {args.command}")
    except (SessionError, ExperimentError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
