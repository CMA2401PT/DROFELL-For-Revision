#!/usr/bin/env python3
"""Shared session state and process helpers for split throughput entrypoints."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = Path(__file__).resolve().parent
SESSION_ROOT = EXP_ROOT / "sessions"
RESULT_ROOT = EXP_ROOT / "results"
LOG_ROOT = EXP_ROOT / "logs"
MODEL_SPECS = {
    "7b": {
        "name": "teared-Llama-2-7B-GPTQ",
        "decoder_layers": 32,
    },
    "70b": {
        "name": "teared-Llama-2-70B-GPTQ",
        "decoder_layers": 80,
    },
}
AGENT = "throughput-agent"
BRIDGE = "dtpbr0"
SUBNET = "10.203.0"
NSPREFIX = "dtp-"
PORT = 7260
TEST_NODE_ID = 10
ORBIT_NODE_ID = 100
SERVICE_NODE_IDS = tuple(range(1, 9))
SCHEMA_VERSION = 1


class SessionError(RuntimeError):
    pass


def model_spec(model_size: str) -> dict[str, Any]:
    try:
        return MODEL_SPECS[model_size]
    except KeyError as error:
        raise SessionError(f"unsupported model size: {model_size}") from error


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def require_root() -> None:
    if os.geteuid() != 0:
        raise SessionError(
            "this entrypoint must run as root; invoke it with sudo -E so "
            "namespace and uclampset processes do not prompt mid-experiment"
        )


def session_dir(session_id: str) -> Path:
    if not session_id or "/" in session_id or session_id in {".", ".."}:
        raise SessionError(f"invalid session id: {session_id!r}")
    return SESSION_ROOT / session_id


def manifest_path(session_id: str) -> Path:
    return session_dir(session_id) / "session.json"


def case_root(session_id: str) -> Path:
    return RESULT_ROOT / session_id / "cases"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SessionError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SessionError(f"JSON must contain an object: {path}")
    return value


def load_manifest(session_id: str) -> dict[str, Any]:
    path = manifest_path(session_id)
    if not path.is_file():
        raise SessionError(f"session manifest does not exist: {path}")
    manifest = read_json(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SessionError(f"unsupported session manifest schema: {path}")
    if manifest.get("session_id") != session_id:
        raise SessionError(f"session id mismatch in manifest: {path}")
    return manifest


def proc_start_ticks(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError:
        return None
    close = stat.rfind(")")
    if close < 0:
        return None
    fields = stat[close + 2 :].split()
    # fields[0] is state; a zombie is no longer a usable serving process.
    if not fields or fields[0] == "Z":
        return None
    # field 22 in procfs is starttime.
    return fields[19] if len(fields) > 19 else None


def process_matches(entry: dict[str, Any]) -> bool:
    try:
        pid = int(entry["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    if proc_start_ticks(pid) != str(entry.get("start_ticks")):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def manifest_process_entries(manifest: dict[str, Any]) -> Iterable[dict[str, Any]]:
    processes = manifest.get("processes", {})
    if not isinstance(processes, dict):
        return ()
    entries: list[dict[str, Any]] = []
    orbit = processes.get("orbit")
    if isinstance(orbit, dict):
        entries.append(orbit)
    dealers = processes.get("dealers", {})
    if isinstance(dealers, dict):
        entries.extend(item for item in dealers.values() if isinstance(item, dict))
    return entries


def serving_alive(manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    processes = manifest.get("processes", {})
    orbit = processes.get("orbit") if isinstance(processes, dict) else None
    if not isinstance(orbit, dict) or not process_matches(orbit):
        missing.append("orbit")
    dealers = processes.get("dealers", {}) if isinstance(processes, dict) else {}
    if isinstance(dealers, dict):
        for name, entry in dealers.items():
            if not isinstance(entry, dict) or not process_matches(entry):
                missing.append(str(name))
    return not missing, missing


def session_env(manifest: dict[str, Any]) -> dict[str, str]:
    network = manifest.get("network")
    if not isinstance(network, dict):
        raise SessionError("session has no network configuration")
    try:
        bandwidth = int(network["bandwidth_mbit"])
        rtt = float(network["target_rtt_ms"])
    except (KeyError, TypeError, ValueError) as error:
        raise SessionError("invalid network configuration in session") from error
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONUNBUFFERED": "1",
        "BRIDGE": str(network.get("bridge", BRIDGE)),
        "SUBNET": str(network.get("subnet", SUBNET)),
        "NSPREFIX": str(network.get("namespace_prefix", NSPREFIX)),
        "LAT": f"{rtt / 2:g}ms",
        "BW": f"{bandwidth}mbit",
    })
    return env


def net_cluster_env(manifest: dict[str, Any], node_id: int | None = None) -> dict[str, str]:
    env = session_env(manifest)
    if node_id is not None:
        env["ID"] = str(node_id)
    return env


def kill_process_group(entry: dict[str, Any], grace_seconds: float = 10.0) -> None:
    try:
        pid = int(entry["pid"])
    except (KeyError, TypeError, ValueError):
        return
    if proc_start_ticks(pid) != str(entry.get("start_ticks")):
        return
    try:
        pgid = int(entry.get("pgid", pid))
    except (TypeError, ValueError):
        pgid = pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and proc_start_ticks(pid) is not None:
        time.sleep(0.1)
    if proc_start_ticks(pid) is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def cleanup_namespace(manifest: dict[str, Any], node_id: int) -> None:
    """Remove one namespace and its host veth, without touching serving nodes."""
    network = manifest.get("network")
    if isinstance(network, dict) and network.get("no_netns"):
        return
    env = net_cluster_env(manifest)
    prefix = env["NSPREFIX"]
    namespace = f"{prefix}{node_id}"
    host_veth = f"v-{namespace}"
    subprocess.run(
        ["ip", "netns", "del", namespace],
        env=env, cwd=REPO_ROOT, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=False,
    )
    subprocess.run(
        ["ip", "link", "del", host_veth],
        env=env, cwd=REPO_ROOT, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=False,
    )


def cleanup_all_network(manifest: dict[str, Any]) -> None:
    network = manifest.get("network")
    if isinstance(network, dict) and network.get("no_netns"):
        return
    env = net_cluster_env(manifest)
    subprocess.run(
        [str(REPO_ROOT / "scripts/online/benchmark/net_cluster"), "down"],
        env=env, cwd=REPO_ROOT, check=False,
    )


def run_checked(
    command: Sequence[str], *, env: dict[str, str], log_path: Path | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    if log_path is None:
        return subprocess.run(
            list(command), cwd=REPO_ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(
            list(command), cwd=REPO_ROOT, env=env, text=True,
            stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False,
        )
