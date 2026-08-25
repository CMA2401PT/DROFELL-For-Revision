#!/usr/bin/env python3
"""Run the Llama2-7B consolidation / batch-merging / dual-queue experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = Path(__file__).resolve().parent
PATCH_ROOT = EXP_ROOT / "patches"
DEFAULT_PYTHON = Path("/home/dai/projects/LLM/dflash_related/.venv/bin/python")
TORCH_EXTENSIONS_DIR = Path("/home/dai/.cache/torch_extensions/py312_cu130")
USER_LOCAL_BIN = Path("/home/dai/.local/bin")
UCLAMPSET = Path("/usr/bin/uclampset")
MODEL = "teared-Llama-2-7B-GPTQ"
PORT = 7274
ORBIT = f"127.0.0.1:{PORT}"

COMMON_PATCH = PATCH_ROOT / "common-instrumentation.patch"
CAP_RELEASE_PATCH = PATCH_ROOT / "cap-release.patch"
CONSOLIDATION_PATCH = PATCH_ROOT / "disable-consolidation.patch"
MERGING_PATCH = PATCH_ROOT / "disable-batch-merging.patch"
DUAL_QUEUE_PATCH = PATCH_ROOT / "disable-dual-queues.patch"
HISTORICAL_PATCH = PATCH_ROOT / "historical-no-opt.patch"
ALL_PATCHES = (
    COMMON_PATCH,
    CAP_RELEASE_PATCH,
    CONSOLIDATION_PATCH,
    MERGING_PATCH,
    DUAL_QUEUE_PATCH,
    HISTORICAL_PATCH,
)


class ExperimentError(RuntimeError):
    pass


@dataclass(frozen=True)
class Case:
    case_id: str
    consolidation: bool
    batch_merging: bool
    dual_queues: bool
    historical: bool = False

    @property
    def async_backend(self) -> str:
        return "golang" if self.dual_queues else "py"

    @property
    def busy_wait(self) -> bool:
        return self.dual_queues

    def patches(self, cap_release: bool = True) -> tuple[Path, ...]:
        if self.historical:
            return (COMMON_PATCH, HISTORICAL_PATCH)
        selected = [COMMON_PATCH]
        if self.consolidation and cap_release:
            selected.append(CAP_RELEASE_PATCH)
        if not self.batch_merging:
            selected.append(MERGING_PATCH)
        if not self.dual_queues:
            selected.append(DUAL_QUEUE_PATCH)
        return tuple(selected)


CASES = (
    Case("c1-m1-d1", True, True, True),
    Case("c0-m1-d1", False, True, True),
    Case("c1-m0-d1", True, False, True),
    Case("c1-m1-d0", True, True, False),
    Case("c0-m0-d1", False, False, True),
    Case("c0-m1-d0", False, True, False),
    Case("c1-m0-d0", True, False, False),
    Case("c0-m0-d0", False, False, False),
    Case("historical-no-opt", False, False, False, historical=True),
)


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def command_text(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer_leaf_total(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return sum(integer_leaf_total(item) for item in value.values())
    if isinstance(value, list):
        return sum(integer_leaf_total(item) for item in value)
    return 0


def counter_total(value: Any, key: str) -> int:
    if isinstance(value, dict):
        return sum(
            integer_leaf_total(item) if name == key
            else counter_total(item, key)
            for name, item in value.items()
        )
    if isinstance(value, list):
        return sum(counter_total(item, key) for item in value)
    return 0


class Runner:
    def __init__(
        self,
        *,
        python: Path,
        run_id: str,
        repeats: int,
        max_tokens: int,
        selected_cases: tuple[str, ...],
        intercept_window_ms: float,
        cap_release: bool,
        merge_limit: int,
        num_seqs: int,
        busy_wait_override: bool | None,
        resource_monitor_interval: float,
        idle_seconds: float,
        timeout: float,
        resume: bool,
    ) -> None:
        self.python = python.absolute()
        if not self.python.exists():
            raise ExperimentError(f"Python does not exist: {self.python}")
        if not UCLAMPSET.exists():
            raise ExperimentError(f"uclampset does not exist: {UCLAMPSET}")
        if repeats <= 0:
            raise ExperimentError("repeats must be positive")
        if max_tokens < 4 or max_tokens > 220:
            raise ExperimentError("max_tokens must be in [4, 220]")
        if intercept_window_ms <= 0:
            raise ExperimentError("intercept_window_ms must be positive")
        if merge_limit not in (1, 2, 4, 8, 16):
            raise ExperimentError("merge_limit must be one of 1, 2, 4, 8, 16")
        if num_seqs <= 0 or num_seqs > 16:
            raise ExperimentError("num_seqs must be in [1, 16]")
        if resource_monitor_interval < 0:
            raise ExperimentError("resource_monitor_interval must be non-negative")
        if idle_seconds < 0:
            raise ExperimentError("idle_seconds must be non-negative")
        if idle_seconds and not resource_monitor_interval:
            raise ExperimentError(
                "idle_seconds requires a positive resource_monitor_interval"
            )
        missing = [path for path in ALL_PATCHES if not path.exists()]
        if missing:
            raise ExperimentError(f"missing patches: {missing}")
        self.run_id = run_id
        self.repeats = repeats
        self.max_tokens = max_tokens
        self.selected_cases = selected_cases
        self.intercept_window_ms = intercept_window_ms
        self.cap_release = cap_release
        self.merge_limit = merge_limit
        self.num_seqs = num_seqs
        self.busy_wait_override = busy_wait_override
        self.resource_monitor_interval = resource_monitor_interval
        self.idle_seconds = idle_seconds
        self.timeout = timeout
        self.resume = resume
        self.result_root = EXP_ROOT / "results" / run_id
        self.log_root = EXP_ROOT / "logs" / run_id
        self.environment_path = EXP_ROOT / "environment" / f"{run_id}.json"
        self.network_path = EXP_ROOT / "networks" / "llama2-7b-consolidation.json"
        self.commands: list[dict[str, Any]] = []
        self.worktrees: dict[str, Path] = {}
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self.network_process: subprocess.Popen[bytes] | None = None
        self.network_log: Path | None = None
        self.base_env = os.environ.copy()
        self.base_env.update({
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "PATH": f"{USER_LOCAL_BIN}:{self.base_env.get('PATH', '')}",
            "TORCH_EXTENSIONS_DIR": str(TORCH_EXTENSIONS_DIR),
            "INTERCEPT_WINDOW_TIME": f"{intercept_window_ms / 1000:.9f}",
            "INTERCEPT_MAX_SIZE": "8",
        })

    def case_busy_wait(self, case: Case) -> bool:
        if self.busy_wait_override is not None:
            return self.busy_wait_override
        return case.busy_wait

    def output(self, *command: str) -> str:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=self.base_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return completed.stdout.strip()

    def source_env(
        self,
        source_root: Path,
        metrics_path: Path | None = None,
        group_inference: bool | None = None,
    ) -> dict[str, str]:
        env = self.base_env.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{source_root}:{existing}" if existing else str(source_root)
        )
        if metrics_path is not None:
            env["DROFELL_METRICS_PATH"] = str(metrics_path.absolute())
        else:
            env.pop("DROFELL_METRICS_PATH", None)
        if group_inference is not None:
            env["GROUP_INFERENCE"] = "1" if group_inference else "0"
        return env

    def record(self, command: Sequence[str], **extra: Any) -> None:
        self.commands.append({"command": command_text(command), **extra})

    def run_command(
        self,
        command: Sequence[str],
        *,
        cwd: Path = REPO_ROOT,
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        actual_env = self.base_env if env is None else env
        if log_path is None:
            completed = subprocess.run(
                list(command), cwd=cwd, env=actual_env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=timeout, check=False,
            )
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    list(command), cwd=cwd, env=actual_env, text=True,
                    stdout=log, stderr=subprocess.STDOUT,
                    timeout=timeout, check=False,
                )
        self.record(
            command, cwd=str(cwd), returncode=completed.returncode,
            log=str(log_path) if log_path else None,
        )
        if check and completed.returncode != 0:
            content = completed.stdout or (
                log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
                if log_path and log_path.exists() else ""
            )
            raise ExperimentError(
                f"command failed ({completed.returncode}): "
                f"{command_text(command)}\n{content}"
            )
        return completed

    def uclamped_python(self, *args: str) -> tuple[str, ...]:
        return (str(UCLAMPSET), "-m", "1024", "--", str(self.python), *args)

    def generate_network(self) -> None:
        self.network_path.parent.mkdir(parents=True, exist_ok=True)
        command = (
            str(self.python), "-m",
            "scripts.online.benchmark.generate_network",
            "--name", "llama2-7b-infer-consolidation",
            "--model", MODEL,
            "--decoder-group-sizes", "8", "8", "8", "8",
            "--dealer", "dealer-1=0",
            "--dealer", "dealer-2=1",
            "--dealer", "dealer-3=2",
            "--dealer", "dealer-4=3",
            "--decoder-backend", "marlin",
            "--max-seq-len", "256",
            "--max-concurrency", str(self.merge_limit),
            "--cuda-graph",
            "--memory-policy", "static",
            "--major-reserved-mib", "8192",
            "--unmanaged-reserved-mib", "32",
            "--minor-min-mib", "0",
            "--minor-max-mib", "4096",
            "--operate-reserved-mib", "1024",
            "--graph-budget-mib", "1024",
            "--output", str(self.network_path),
        )
        self.run_command(command, env=self.source_env(REPO_ROOT))
        snapshot = self.result_root / "network_specs" / self.network_path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.network_path, snapshot)

    def ensure_worktree(self, case: Case) -> Path:
        if case.case_id in self.worktrees:
            return self.worktrees[case.case_id]
        if self._tempdir is None:
            self._tempdir = tempfile.TemporaryDirectory(
                prefix="drofell-infer-consolidation-"
            )
        target = Path(self._tempdir.name) / case.case_id
        self.run_command((
            "git", "worktree", "add", "--detach", str(target), "HEAD"
        ))
        for patch in case.patches(self.cap_release):
            self.run_command(("git", "apply", "--check", str(patch)), cwd=target)
            self.run_command(("git", "apply", str(patch)), cwd=target)

        linked_runtime_paths: list[str] = []
        for runtime_name in ("cache", "data", "split_qllm_models"):
            source = REPO_ROOT / runtime_name
            if not source.exists():
                continue
            destination = target / runtime_name
            if destination.exists() or destination.is_symlink():
                if runtime_name == "split_qllm_models" and destination.is_dir():
                    for child in source.resolve().iterdir():
                        destination_child = destination / child.name
                        if not destination_child.exists():
                            destination_child.symlink_to(
                                child, target_is_directory=child.is_dir()
                            )
                    linked_runtime_paths.append("split_qllm_models/*")
                continue
            destination.symlink_to(
                source, target_is_directory=source.is_dir()
            )
            linked_runtime_paths.append(runtime_name)

        extension_count = 0
        for source in REPO_ROOT.rglob("*.so"):
            relative = source.relative_to(REPO_ROOT)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source)
            extension_count += 1
        if extension_count == 0:
            raise ExperimentError("no local extension modules found to link")
        self.run_command(("git", "diff", "--check"), cwd=target)
        self.commands.append({
            "action": "prepare_worktree",
            "case": case.case_id,
            "worktree": str(target),
            "patches": [str(path) for path in case.patches()],
            "extension_count": extension_count,
            "runtime_paths": linked_runtime_paths,
        })
        self.worktrees[case.case_id] = target
        return target

    def cleanup_worktrees(self) -> None:
        for target in self.worktrees.values():
            subprocess.run(
                ("git", "worktree", "remove", "--force", str(target)),
                cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=False,
            )
        self.worktrees.clear()
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    @staticmethod
    def unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        attempt = 2
        while True:
            candidate = path.with_name(
                f"{path.stem}.attempt-{attempt}{path.suffix}"
            )
            if not candidate.exists():
                return candidate
            attempt += 1

    @staticmethod
    def wait_for_text(
        process: subprocess.Popen[bytes],
        log_path: Path,
        needle: str,
        timeout: float,
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                content = log_path.read_text(encoding="utf-8", errors="replace")
                raise ExperimentError(
                    f"network exited before {needle!r}:\n{content[-20000:]}"
                )
            content = log_path.read_text(encoding="utf-8", errors="replace")
            if needle in content:
                return
            if "Traceback (most recent call last)" in content:
                raise ExperimentError(
                    f"network failed during startup:\n{content[-20000:]}"
                )
            time.sleep(0.2)
        raise ExperimentError(f"timed out waiting for {needle!r} in {log_path}")

    @staticmethod
    def descendant_pids(root_pid: int) -> list[int]:
        discovered = {root_pid}
        changed = True
        while changed:
            changed = False
            for status_path in Path("/proc").glob("[0-9]*/status"):
                try:
                    pid = int(status_path.parent.name)
                    fields = {}
                    for line in status_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines():
                        name, _, value = line.partition(":")
                        fields[name] = value.strip()
                    parent = int(fields.get("PPid", "-1"))
                except (OSError, ValueError):
                    continue
                if parent in discovered and pid not in discovered:
                    discovered.add(pid)
                    changed = True
        return sorted(discovered)

    def uclamp_snapshot(self, root_pid: int, label: str) -> list[dict[str, Any]]:
        snapshot = []
        for pid in self.descendant_pids(root_pid):
            try:
                sched = Path(f"/proc/{pid}/sched").read_text(
                    encoding="utf-8", errors="replace"
                )
                command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(
                    b"\0", b" "
                ).decode("utf-8", errors="replace").strip()
            except OSError:
                continue
            uclamp = [
                line.strip() for line in sched.splitlines()
                if "uclamp.min" in line or "uclamp.max" in line
            ]
            effective_min = None
            for line in uclamp:
                if line.startswith("effective uclamp.min"):
                    try:
                        effective_min = int(float(line.rsplit(":", 1)[1]))
                    except ValueError:
                        pass
            snapshot.append({
                "label": label,
                "pid": pid,
                "command": command,
                "uclamp": uclamp,
                "effective_uclamp_min": effective_min,
            })
        if not snapshot:
            raise ExperimentError(f"no live process in uclamp snapshot {label}")
        invalid = [
            item for item in snapshot
            if item["effective_uclamp_min"] not in (None, 1024)
        ]
        if invalid:
            raise ExperimentError(f"uclamp.min is not 1024: {invalid}")
        return snapshot

    @staticmethod
    def process_counters(pid: int) -> dict[str, int] | None:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(
                encoding="utf-8", errors="replace"
            )
            fields = stat[stat.rfind(") ") + 2:].split()
            cpu_ticks = int(fields[11]) + int(fields[12])
            voluntary = 0
            involuntary = 0
            for status_path in Path(f"/proc/{pid}/task").glob("*/status"):
                for line in status_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines():
                    if line.startswith("voluntary_ctxt_switches:"):
                        voluntary += int(line.split(":", 1)[1])
                    elif line.startswith("nonvoluntary_ctxt_switches:"):
                        involuntary += int(line.split(":", 1)[1])
            return {
                "cpu_ticks": cpu_ticks,
                "voluntary_context_switches": voluntary,
                "involuntary_context_switches": involuntary,
            }
        except (OSError, ValueError, IndexError):
            return None

    def process_tree_counters(self, root_pids: Sequence[int]) -> dict[int, dict[str, int]]:
        pids: set[int] = set()
        for root_pid in root_pids:
            pids.update(self.descendant_pids(root_pid))
        counters = {}
        for pid in sorted(pids):
            value = self.process_counters(pid)
            if value is not None:
                counters[pid] = value
        return counters

    def monitor_process_tree(
        self,
        *,
        root_pids: Sequence[int],
        output_path: Path,
        duration: float | None = None,
        process: subprocess.Popen[bytes] | None = None,
    ) -> dict[str, Any]:
        if not self.resource_monitor_interval:
            return {}
        if (duration is None) == (process is None):
            raise ExperimentError("process monitor requires exactly one stop condition")
        started = time.monotonic()
        deadline = started + duration if duration is not None else started + self.timeout
        previous: dict[int, dict[str, int]] = {}
        deltas = {
            "cpu_ticks": 0,
            "voluntary_context_switches": 0,
            "involuntary_context_switches": 0,
        }
        samples = []
        timed_out = False
        while True:
            now = time.monotonic()
            current = self.process_tree_counters(root_pids)
            for pid, values in current.items():
                prior = previous.get(pid)
                if prior is None:
                    continue
                for name in deltas:
                    deltas[name] += max(0, values[name] - prior[name])
            previous = current
            samples.append({
                "elapsed_seconds": now - started,
                "process_count": len(current),
                **{
                    name: sum(item[name] for item in current.values())
                    for name in deltas
                },
            })
            if duration is not None and now >= deadline:
                break
            if process is not None and process.poll() is not None:
                break
            if process is not None and now >= deadline:
                timed_out = True
                break
            time.sleep(self.resource_monitor_interval)
        wall_seconds = time.monotonic() - started
        clock_ticks = int(os.sysconf("SC_CLK_TCK"))
        result = {
            "schema_version": 1,
            "root_pids": list(root_pids),
            "sample_interval_seconds": self.resource_monitor_interval,
            "sample_count": len(samples),
            "wall_seconds": wall_seconds,
            "cpu_seconds": deltas["cpu_ticks"] / clock_ticks,
            "average_cpu_cores": (
                deltas["cpu_ticks"] / clock_ticks / wall_seconds
                if wall_seconds else None
            ),
            "voluntary_context_switches": deltas[
                "voluntary_context_switches"
            ],
            "involuntary_context_switches": deltas[
                "involuntary_context_switches"
            ],
            "timed_out": timed_out,
            "measurement_note": (
                "Linux /proc process-tree CPU time and scheduler context "
                "switches. Context switches are an OS scheduling proxy, not "
                "an instrumented count of event-loop wakeups. Processes that "
                "start and exit entirely between samples may be missed."
            ),
            "samples": samples,
        }
        write_json(output_path, result)
        return result

    def start_network(self, case: Case, source_root: Path) -> list[dict[str, Any]]:
        if self.network_process is not None:
            raise ExperimentError("network is already running")
        case_root = self.result_root / case.case_id
        case_root.mkdir(parents=True, exist_ok=True)
        metrics_path = case_root / "runtime_metrics.jsonl"
        log_path = self.unique_path(self.log_root / case.case_id / "network.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        busy_flag = (
            "--busy-wait" if self.case_busy_wait(case) else "--no-busy-wait"
        )
        command = self.uclamped_python(
            "-m", "scripts.online.benchmark.launch_network", "local",
            "--network", str(self.network_path),
            "--port", str(PORT),
            "--advertise-host", "127.0.0.1",
            "--async-backend", case.async_backend,
            busy_flag,
            "--place", "dealer-1", "0", "0", "cuda:0",
            "--place", "dealer-2", "0", "0", "cuda:0",
            "--place", "dealer-3", "1", "0", "cuda:1",
            "--place", "dealer-4", "1", "0", "cuda:1",
        )
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                list(command), cwd=source_root,
                env=self.source_env(
                    source_root, metrics_path,
                    group_inference=case.consolidation,
                ),
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.network_process = process
        self.network_log = log_path
        self.record(
            command, cwd=str(source_root), pid=process.pid, log=str(log_path)
        )
        self.wait_for_text(
            process, log_path,
            "network 'llama2-7b-infer-consolidation' ready",
            timeout=900,
        )
        return self.uclamp_snapshot(process.pid, f"{case.case_id}:network")

    def stop_network(self) -> None:
        process = self.network_process
        self.network_process = None
        self.network_log = None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=20)

    def start_gpu_monitor(self, path: Path) -> subprocess.Popen[bytes]:
        path.parent.mkdir(parents=True, exist_ok=True)
        query = (
            "timestamp,index,utilization.gpu,memory.used,memory.free,"
            "clocks.sm,power.draw"
        )
        command = (
            "nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits",
            "--loop-ms=500",
        )
        with path.open("wb") as output:
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        self.record(command, pid=process.pid, log=str(path), monitor=True)
        return process

    @staticmethod
    def stop_gpu_monitor(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

    def repeat_complete(self, repeat_dir: Path) -> bool:
        return all((repeat_dir / name).exists() for name in (
            "args.json", "result.json", "traces.json", "phases.json",
            "decode_graph.json",
        ))

    def audit_repeat(self, case: Case, repeat_dir: Path) -> dict[str, Any]:
        args = json.loads((repeat_dir / "args.json").read_text(encoding="utf-8"))
        phases = json.loads(
            (repeat_dir / "phases.json").read_text(encoding="utf-8")
        )
        graph = json.loads(
            (repeat_dir / "decode_graph.json").read_text(encoding="utf-8")
        )
        record = json.loads(
            (repeat_dir / "result.json").read_text(encoding="utf-8")
        )
        traces = json.loads(
            (repeat_dir / "traces.json").read_text(encoding="utf-8")
        )
        benchmark = args["benchmark"]
        if benchmark["parallel_seqs"] != self.num_seqs:
            raise ExperimentError(f"unexpected concurrency: {repeat_dir}")
        if benchmark["max_tokens"] != self.max_tokens:
            raise ExperimentError(f"unexpected max_tokens: {repeat_dir}")
        if benchmark["logprobs"] is not None:
            raise ExperimentError(f"logprobs is enabled: {repeat_dir}")
        if not benchmark["disable_gc"]:
            raise ExperimentError(f"GC is unexpectedly enabled: {repeat_dir}")
        expected_per_task = self.max_tokens - 1
        expected_count = self.num_seqs * expected_per_task
        counts = Counter(int(item[0]) for item in record)
        if len(record) != expected_count or counts != Counter({
            index: expected_per_task for index in range(self.num_seqs)
        }):
            raise ExperimentError(f"incomplete timed tokens: {repeat_dir}")
        if len(traces) != self.num_seqs or any(not task for task in traces):
            raise ExperimentError(f"incomplete detail traces: {repeat_dir}")
        if any(
            int(left[1]) > int(right[1])
            for left, right in zip(record, record[1:])
        ):
            raise ExperimentError(f"non-monotonic token record: {repeat_dir}")
        if "decode_start_ns" not in phases:
            raise ExperimentError(f"common timing patch absent: {repeat_dir}")
        expected_batches = (
            list(range(1, self.merge_limit + 1))
            if case.batch_merging else [1]
        )
        for dealer, stats in graph.items():
            if not stats.get("enabled"):
                raise ExperimentError(f"graph disabled for {dealer}: {repeat_dir}")
            if stats.get("configured_batch_sizes") != expected_batches:
                raise ExperimentError(
                    f"unexpected graph batches for {dealer}: {repeat_dir}"
                )
        fallbacks = counter_total(graph, "fallbacks")
        decoder_only_replays = sum(
            int(stats.get("decoder_only_replays", 0))
            for stats in graph.values()
        )
        fused_replays = sum(
            int(stats.get("fused_replays", 0)) for stats in graph.values()
        )
        head_replays = int(graph["dealer-1"]["head"].get("replays", 0))
        tail_replays = int(graph["dealer-4"]["tail"].get("replays", 0))
        if fallbacks:
            raise ExperimentError(f"graph fallback {fallbacks}: {repeat_dir}")
        if decoder_only_replays <= 0 or head_replays <= 0 or tail_replays <= 0:
            raise ExperimentError(
                "standalone H/D/T graph did not replay: "
                f"D={decoder_only_replays}, H={head_replays}, "
                f"T={tail_replays}: {repeat_dir}"
            )
        if fused_replays:
            raise ExperimentError(
                f"cross-stage graph unexpectedly replayed {fused_replays}: "
                f"{repeat_dir}"
            )
        metrics_path = repeat_dir.parent / "runtime_metrics.jsonl"
        if metrics_path.exists():
            flush_sizes = []
            for line in metrics_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if not line.strip():
                    continue
                metric = json.loads(line)
                timestamp = int(metric.get("time_ns", -1))
                if (
                    phases["decode_start_ns"] <= timestamp <= phases["end_ns"]
                    and metric.get("event") == "consolidation_flush"
                ):
                    flush_sizes.append(int(metric["actual_size"]))
            if self.cap_release and any(size > 8 for size in flush_sizes):
                raise ExperimentError(
                    f"consolidation packet exceeds cap 8: {repeat_dir}"
                )
        return {
            "timed_tokens": len(record),
            "trace_steps": sum(len(task) for task in traces),
            "graph_fallbacks": fallbacks,
            "decoder_only_replays_snapshot": decoder_only_replays,
            "head_replays_snapshot": head_replays,
            "tail_replays_snapshot": tail_replays,
            "fused_replays_snapshot": fused_replays,
            "decode_start_ns": phases["decode_start_ns"],
            "end_ns": phases["end_ns"],
        }

    def run_repeat(
        self,
        case: Case,
        source_root: Path,
        repeat_index: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        repeat_dir = self.result_root / case.case_id / f"repeat-{repeat_index}"
        if self.resume and self.repeat_complete(repeat_dir):
            return self.audit_repeat(case, repeat_dir), []
        if repeat_dir.exists() and any(repeat_dir.iterdir()):
            failed_root = (
                self.result_root / case.case_id / "failed_attempts"
                / f"repeat-{repeat_index}"
            )
            attempt = 1
            failed = failed_root / f"attempt-{attempt}"
            while failed.exists():
                attempt += 1
                failed = failed_root / f"attempt-{attempt}"
            failed.parent.mkdir(parents=True, exist_ok=True)
            repeat_dir.rename(failed)
        repeat_dir.parent.mkdir(parents=True, exist_ok=True)
        busy_flag = (
            "--busy-wait" if self.case_busy_wait(case) else "--no-busy-wait"
        )
        command = self.uclamped_python(
            "-m", "scripts.online.benchmark.benchmark_infer",
            "--measurement", "request",
            "--orbit", ORBIT,
            "--host-id", f"infer-consolidation-{case.case_id}-{repeat_index}",
            "--advertise-host", "127.0.0.1",
            "--model", MODEL,
            "--output", str(repeat_dir),
            "--prompt", "once upon a time",
            "--seqs", str(self.num_seqs),
            "--max-tokens", str(self.max_tokens),
            "--warmup-tokens", "32",
            "--num-paths", "1",
            "--logprobs", "0",
            "--detail-trace",
            "--disable-gc",
            "--progress-interval", "0",
            "--async-backend", case.async_backend,
            busy_flag,
        )
        log_path = self.unique_path(
            self.log_root / case.case_id / f"repeat-{repeat_index}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path = self.result_root / case.case_id / "runtime_metrics.jsonl"
        started = time.time()
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                list(command), cwd=source_root,
                env=self.source_env(
                    source_root, metrics_path,
                    group_inference=case.consolidation,
                ),
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.record(
            command, cwd=str(source_root), pid=process.pid, log=str(log_path)
        )
        time.sleep(0.5)
        benchmark_uclamp = (
            self.uclamp_snapshot(
                process.pid, f"{case.case_id}:repeat-{repeat_index}"
            )
            if process.poll() is None else []
        )
        process_metrics = self.monitor_process_tree(
            root_pids=(self.network_process.pid, process.pid),
            output_path=repeat_dir / "process_metrics.json",
            process=process,
        ) if self.resource_monitor_interval and self.network_process else {}
        try:
            if process_metrics.get("timed_out"):
                raise subprocess.TimeoutExpired(command, self.timeout)
            returncode = process.wait(timeout=20 if process_metrics else self.timeout)
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=20)
            raise ExperimentError(
                f"benchmark timeout for {case.case_id} repeat {repeat_index}"
            ) from error
        self.commands[-1]["returncode"] = returncode
        self.commands[-1]["wall_seconds"] = time.time() - started
        if returncode != 0:
            content = log_path.read_text(encoding="utf-8", errors="replace")
            raise ExperimentError(
                f"benchmark failed ({returncode}): {case.case_id} "
                f"repeat {repeat_index}\n{content[-20000:]}"
            )
        audit = self.audit_repeat(case, repeat_dir)
        audit["wall_seconds"] = time.time() - started
        if process_metrics:
            audit["process_metrics"] = {
                key: value for key, value in process_metrics.items()
                if key != "samples"
            }
        return audit, benchmark_uclamp

    def case_complete(self, case: Case) -> bool:
        return all(self.repeat_complete(
            self.result_root / case.case_id / f"repeat-{index}"
        ) for index in range(self.repeats))

    def run_case(self, case: Case) -> None:
        if (
            self.resume and self.case_complete(case)
            and (self.result_root / case.case_id / "case.json").exists()
        ):
            for index in range(self.repeats):
                self.audit_repeat(
                    case,
                    self.result_root / case.case_id / f"repeat-{index}",
                )
            print(f"resume: skip complete case {case.case_id}")
            return
        source_root = self.ensure_worktree(case)
        case_root = self.result_root / case.case_id
        case_root.mkdir(parents=True, exist_ok=True)
        monitor = self.start_gpu_monitor(case_root / "gpu_monitor.csv")
        network_uclamp: list[dict[str, Any]] = []
        repeat_audits: list[dict[str, Any]] = []
        benchmark_uclamp: list[dict[str, Any]] = []
        try:
            network_uclamp = self.start_network(case, source_root)
            if self.idle_seconds:
                self.monitor_process_tree(
                    root_pids=(self.network_process.pid,),
                    output_path=case_root / "idle_process_metrics.json",
                    duration=self.idle_seconds,
                )
            for repeat_index in range(self.repeats):
                audit, clamp = self.run_repeat(
                    case, source_root, repeat_index
                )
                repeat_audits.append(audit)
                benchmark_uclamp.extend(clamp)
        finally:
            self.stop_network()
            self.stop_gpu_monitor(monitor)
        write_json(case_root / "case.json", {
            **asdict(case),
            "async_backend": case.async_backend,
            "busy_wait": self.case_busy_wait(case),
            "source_root": str(source_root),
            "patches": [
                {"path": str(path), "sha256": sha256(path)}
                for path in case.patches(self.cap_release)
            ],
            "parameters": {
                "model": MODEL,
                "backend": "marlin",
                "cuda_graph": True,
                "group_inference_env": (
                    "1" if case.consolidation else "0"
                ),
                "clients": self.num_seqs,
                "max_tokens": self.max_tokens,
                "timed_tokens_per_client": self.max_tokens - 1,
                "warmup_tokens": 32,
                "initial_stagger_seconds_per_sequence_index": 0.001,
                "intercept_window_seconds": self.intercept_window_ms / 1000,
                "cap_release": self.cap_release,
                "merge_limit": self.merge_limit if case.batch_merging else 1,
                "repeats": self.repeats,
            },
            "uclamp": network_uclamp + benchmark_uclamp,
            "repeat_audits": repeat_audits,
        })

    def selected(self) -> tuple[Case, ...]:
        selected = set(self.selected_cases)
        return tuple(case for case in CASES if case.case_id in selected)

    def run_matrix(self) -> None:
        for case in self.selected():
            self.run_case(case)

    def capture_environment(self) -> None:
        governors = []
        for path in sorted(Path("/sys/devices/system/cpu").glob(
            "cpu[0-9]*/cpufreq/scaling_governor"
        )):
            try:
                governors.append(path.read_text(encoding="utf-8").strip())
            except OSError:
                pass
        old_commands: list[dict[str, Any]] = []
        old_invocations: list[str] = []
        if self.environment_path.exists():
            try:
                old = json.loads(
                    self.environment_path.read_text(encoding="utf-8")
                )
                old_commands = list(old.get("commands", []))
                old_invocations = list(old.get("top_level_commands", []))
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        invocation = command_text((str(self.python), *sys.argv))
        write_json(self.environment_path, {
            "run_id": self.run_id,
            "top_level_commands": old_invocations + [invocation],
            "git_commit": self.output("git", "rev-parse", "HEAD"),
            "git_status": self.output("git", "status", "--short"),
            "python": self.output(str(self.python), "--version"),
            "torch": self.output(
                str(self.python), "-c",
                "import torch; print(torch.__version__, torch.version.cuda)",
            ),
            "gpus": self.output(
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,power.limit",
                "--format=csv,noheader",
            ),
            "gpu_compute_processes_after": self.output(
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
                "--format=csv,noheader",
            ),
            "cpu": self.output("lscpu"),
            "cpu_governors": sorted(set(governors)),
            "uclampset": self.output("uclampset", "--version"),
            "uclamp_min": 1024,
            "parameters": {
                "cases": [asdict(case) for case in CASES],
                "selected_cases": list(self.selected_cases),
                "repeats": self.repeats,
                "max_tokens": self.max_tokens,
                "num_seqs": self.num_seqs,
                "merge_limit": self.merge_limit,
                "busy_wait_override": self.busy_wait_override,
                "resource_monitor_interval": self.resource_monitor_interval,
                "idle_seconds": self.idle_seconds,
                "initial_stagger_seconds_per_sequence_index": 0.001,
                "intercept_window_seconds": self.intercept_window_ms / 1000,
                "cap_release": self.cap_release,
                "backend": "marlin",
                "cuda_graph": True,
                "decoder_group_sizes": [8, 8, 8, 8],
                "dealer_grouping": {
                    "dealer-1": [0], "dealer-2": [1],
                    "dealer-3": [2], "dealer-4": [3],
                },
                "placement": {
                    "dealer-1": "cuda:0", "dealer-2": "cuda:0",
                    "dealer-3": "cuda:1", "dealer-4": "cuda:1",
                },
                "graph_policy": {
                    "standalone_head": True,
                    "standalone_decoder": True,
                    "standalone_tail": True,
                    "cross_stage_folding": False,
                },
            },
            "patches": [
                {"path": str(path), "sha256": sha256(path)}
                for path in ALL_PATCHES
            ],
            "commands": old_commands + self.commands,
        })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the inference consolidation 2x2x2 experiment"
    )
    parser.add_argument(
        "phase", nargs="?", default="all",
        choices=("prepare", "run", "analyze", "all"),
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--intercept-window-ms", type=float, default=10.0)
    parser.add_argument(
        "--merge-limit", type=int, choices=(1, 2, 4, 8, 16), default=8,
        help="Brain merge/concurrency limit; defaults to the paper value 8.",
    )
    parser.add_argument(
        "--num-seqs", type=int, default=16,
        help="Concurrent inference sequences (default: 16).",
    )
    parser.add_argument(
        "--busy-wait", action=argparse.BooleanOptionalAction, default=None,
        help=(
            "Override only the busy-wait flag while preserving each case's "
            "async backend; useful for the golang on/off single-variable scan."
        ),
    )
    parser.add_argument(
        "--resource-monitor-interval", type=float, default=0.0,
        help=(
            "Sample /proc CPU time and scheduler context switches for the "
            "network+benchmark process trees; 0 disables monitoring."
        ),
    )
    parser.add_argument(
        "--idle-seconds", type=float, default=0.0,
        help="Measure the loaded network without clients before each case.",
    )
    parser.add_argument(
        "--legacy-fixed-window", action="store_true",
        help="reproduce the archived sender that always waits for the deadline",
    )
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(case.case_id for case in CASES),
        default=tuple(case.case_id for case in CASES),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    runner = Runner(
        python=args.python,
        run_id=args.run_id,
        repeats=args.repeats,
        max_tokens=args.max_tokens,
        selected_cases=tuple(dict.fromkeys(args.cases)),
        intercept_window_ms=args.intercept_window_ms,
        cap_release=not args.legacy_fixed_window,
        merge_limit=args.merge_limit,
        num_seqs=args.num_seqs,
        busy_wait_override=args.busy_wait,
        resource_monitor_interval=args.resource_monitor_interval,
        idle_seconds=args.idle_seconds,
        timeout=args.timeout,
        resume=args.resume,
    )
    try:
        if args.phase in {"prepare", "run", "all"}:
            runner.generate_network()
        if args.phase in {"run", "all"}:
            runner.run_matrix()
        if args.phase in {"analyze", "all"}:
            runner.run_command((
                str(args.python), str(EXP_ROOT / "analyze.py"),
                str(runner.result_root),
            ))
    finally:
        runner.stop_network()
        runner.cleanup_worktrees()
        runner.capture_environment()
    print(f"run-id: {runner.run_id}")
    print(f"results: {runner.result_root}")
    print(f"environment: {runner.environment_path}")


if __name__ == "__main__":
    main()
