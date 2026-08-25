#!/usr/bin/env python3
"""Run the graph-only Binding/async/merge revision experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/home/dai/projects/LLM/dflash_related/.venv/bin/python")
DEFAULT_VLLM_MODEL = Path(
    "/home/dai/projects/LLM/dflash_related/models/Llama-2-7B-GPTQ"
)
MODEL = "teared-Llama-2-7B-GPTQ"
PROMPT = "Explain why the sky appears blue in three sentences."
PORT = 7255
ORBIT = f"127.0.0.1:{PORT}"
AGENT = "decode-agent"
SEQ_COUNTS = (1, 8, 16)
DEPLOYMENTS = ("persistent-agent", "combined-request-agent")


class ExperimentError(RuntimeError):
    pass


@dataclass(frozen=True)
class OnlineCase:
    case_id: str
    async_backend: str
    busy_wait: bool
    container: str
    binding_size: int
    merge_limit: int
    axes: tuple[str, ...]

    @property
    def decoder_group_sizes(self) -> tuple[int, ...]:
        return (self.binding_size,) * (32 // self.binding_size)

    @property
    def is_async_scan(self) -> bool:
        return "async" in self.axes


ONLINE_CASES = (
    OnlineCase(
        "async-py-sync-single", "py", False, "nonasync", 16, 8,
        ("async",),
    ),
    OnlineCase(
        "default-go-blocking-dual", "golang", False, "shared", 16, 8,
        ("async", "binding", "merge"),
    ),
    OnlineCase(
        "async-go-poll-dual", "golang", True, "shared", 16, 8,
        ("async",),
    ),
    OnlineCase(
        "binding-32", "golang", False, "shared", 32, 8,
        ("binding",),
    ),
    OnlineCase(
        "binding-8", "golang", False, "shared", 8, 8,
        ("binding",),
    ),
    OnlineCase(
        "merge-4", "golang", False, "shared", 16, 4,
        ("merge",),
    ),
    OnlineCase(
        "merge-16", "golang", False, "shared", 16, 16,
        ("merge",),
    ),
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def recursive_values(value: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, dict):
        for name, item in value.items():
            if name == key:
                values.append(item)
            values.extend(recursive_values(item, key))
    elif isinstance(value, list):
        for item in value:
            values.extend(recursive_values(item, key))
    return values


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


def audit_graph_result(path: Path, expected_limit: int | None = None) -> None:
    report = read_json(path)
    if report.get("execution") != "graph":
        raise ExperimentError(f"result is not graph execution: {path}")
    graph = report.get("decode_graph", report.get("graph"))
    if graph is None:
        raise ExperimentError(f"result has no graph evidence: {path}")
    fallbacks = recursive_values(graph, "fallbacks")
    if any(integer_leaf_total(value) != 0 for value in fallbacks):
        raise ExperimentError(f"CUDA graph fallback detected: {path}")
    enabled = recursive_values(graph, "enabled")
    if enabled and not all(bool(value) for value in enabled):
        raise ExperimentError(f"a CUDA graph component is disabled: {path}")
    if expected_limit is not None:
        configured = recursive_values(graph, "configured_batch_sizes")
        expected = list(range(1, expected_limit + 1))
        if not configured or any(list(value) != expected for value in configured):
            raise ExperimentError(
                f"unexpected graph batch sizes in {path}: "
                f"expected={expected}, observed={configured}"
            )


def build_plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "design": "one-factor-at-a-time",
        "graph_only": True,
        "sequence_counts": list(SEQ_COUNTS),
        "default_online": {
            "async_backend": "golang",
            "busy_wait": False,
            "container": "shared",
            "binding_size": 16,
            "merge_limit": 8,
        },
        "online_deployments": list(DEPLOYMENTS),
        "online_cases": [asdict(case) for case in ONLINE_CASES],
        "baselines": [
            {"id": "base"},
            {"id": "offline-binding-16", "decoder_splits": [16, 16]},
            {"id": "offline-binding-32", "decoder_splits": [32]},
            {"id": "vllm"},
        ],
        "loaded_idle": {
            "cases": [case.case_id for case in ONLINE_CASES if case.is_async_scan],
            "deployments": list(DEPLOYMENTS),
        },
        "interpretation": {
            "async-py-sync-single": (
                "CUDA work bypasses the blocking queue; this is a single-queue "
                "execution path, not a separate single-queue event-loop implementation."
            ),
            "missing_cell": (
                "No single-queue polling cell exists because the synchronous CUDA "
                "path only exposes synchronization."
            ),
            "binding_scope": (
                "The one-Dealer Binding scan measures local throughput/latency "
                "sensitivity, not migration, recovery, or network communication cost."
            ),
        },
    }


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # Keep the virtual-environment entrypoint itself. Resolving its symlink
        # can bypass the venv and execute the base interpreter without the
        # experiment dependencies installed.
        self.python = args.python.absolute()
        if not self.python.is_file():
            raise ExperimentError(f"Python does not exist: {self.python}")
        self.result_root = EXP_ROOT / "results" / args.run_id
        self.log_root = EXP_ROOT / "logs" / args.run_id
        self.network_root = self.result_root / "network_specs"
        self.environment_path = EXP_ROOT / "environment" / f"{args.run_id}.json"
        self.commands: list[dict[str, Any]] = []
        self.network_process: subprocess.Popen[bytes] | None = None
        self.base_env = os.environ.copy()
        self.base_env.update({
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
        })

    def python_command(self, *arguments: str) -> tuple[str, ...]:
        command = (str(self.python), *arguments)
        if self.args.uclamp and Path("/usr/bin/uclampset").is_file():
            return ("/usr/bin/uclampset", "-m", "1024", "--", *command)
        return command

    def record(self, command: Sequence[str], **extra: Any) -> None:
        self.commands.append({"command": command_text(command), **extra})

    def run_logged(
        self,
        command: Sequence[str],
        log_path: Path,
        *,
        timeout: float | None = None,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log_path.open("wb") as log:
            completed = subprocess.run(
                list(command), cwd=REPO_ROOT, env=self.base_env,
                stdout=log, stderr=subprocess.STDOUT,
                timeout=self.args.timeout if timeout is None else timeout,
                check=False,
            )
        self.record(
            command, log=str(log_path), returncode=completed.returncode,
            wall_seconds=time.time() - started,
        )
        if completed.returncode != 0:
            output = log_path.read_text(encoding="utf-8", errors="replace")
            raise ExperimentError(
                f"command failed ({completed.returncode}): {command_text(command)}\n"
                f"{output[-20000:]}"
            )

    def complete(self, path: Path, expected_limit: int | None = None) -> bool:
        if not self.args.resume or not path.is_file():
            return False
        audit_graph_result(path, expected_limit)
        print(f"resume: {path}")
        return True

    def common_decode_arguments(self) -> tuple[str, ...]:
        return (
            "--num-seqs", *(str(value) for value in SEQ_COUNTS),
            "--prompt", PROMPT,
            "--runs", str(self.args.runs),
            "--warmup-requests", str(self.args.warmup_requests),
            "--warmup-decode-tokens", str(self.args.warmup_decode_tokens),
            "--decode-tokens", str(self.args.decode_tokens),
        )

    def run_baselines(self) -> None:
        common = self.common_decode_arguments()
        base_result = self.result_root / "baseline" / "base" / "result.json"
        if not self.complete(base_result, 16):
            command = self.python_command(
                "scripts/base/bench.py", *common,
                "--num-blocks", "256", "--device", "cuda:0",
                "--decoder-backend", "marlin", "--max-seq-len", "256",
                "--async-backend", "golang", "--no-busy-wait",
                "--cuda-graph", "--output", str(base_result),
            )
            self.run_logged(command, self.log_root / "baseline" / "base.log")
            audit_graph_result(base_result, 16)

        for binding_size, splits in ((16, (16, 16)), (32, (32,))):
            case_id = f"offline-binding-{binding_size}"
            output = self.result_root / "baseline" / case_id
            result = output / "result.json"
            if self.complete(result, 8):
                continue
            command = self.python_command(
                "scripts/offline/benchmark/inference/benchmark.py",
                "--measurement", "decode", "--model", MODEL,
                "--devices", "cuda:0", "--decoder-splits",
                *(str(value) for value in splits),
                "--decoder-backend", "marlin", "--max-seq-len", "256",
                "--max-concurrency", "8", "--cuda-graph",
                *common,
                "--memory-policy", "static", "--major-reserved-mib", "8192",
                "--major-percentage", "0.8", "--unmanaged-reserved-mib", "32",
                "--minor-min-mib", "0", "--minor-max-mib", "4096",
                "--operate-reserved-mib", "1024", "--graph-budget-mib", "1024",
                "--async-backend", "golang", "--no-busy-wait",
                "--container", "nonasync", "--output", str(output),
            )
            self.run_logged(command, self.log_root / "baseline" / f"{case_id}.log")
            audit_graph_result(result, 8)

        vllm_result = self.result_root / "baseline" / "vllm" / "result.json"
        if not self.complete(vllm_result):
            command = self.python_command(
                "scripts/vllm/decode_benchmark.py", *common,
                "--model", str(self.args.vllm_model),
                "--max-model-len", "256", "--gpu-memory-utilization", "0.9",
                "--cuda-graph", "--output", str(vllm_result),
            )
            self.run_logged(command, self.log_root / "baseline" / "vllm.log")
            audit_graph_result(vllm_result)

    def network_path(self, case: OnlineCase) -> Path:
        return self.network_root / f"{case.case_id}.json"

    def generate_network(self, case: OnlineCase) -> Path:
        output = self.network_path(case)
        group_ids = ",".join(str(index) for index in range(len(case.decoder_group_sizes)))
        command = self.python_command(
            "-m", "scripts.online.benchmark.generate_network",
            "--name", f"binding-async-merge-{case.case_id}",
            "--model", MODEL,
            "--decoder-group-sizes",
            *(str(value) for value in case.decoder_group_sizes),
            "--dealer", f"dealer-0={group_ids}",
            "--decoder-backend", "marlin", "--max-seq-len", "256",
            "--max-concurrency", str(case.merge_limit), "--cuda-graph",
            "--memory-policy", "static", "--major-reserved-mib", "8192",
            "--major-percentage", "0.8", "--unmanaged-reserved-mib", "32",
            "--minor-min-mib", "0", "--minor-max-mib", "4096",
            "--operate-reserved-mib", "1024", "--graph-budget-mib", "1024",
            "--output", str(output),
        )
        self.run_logged(
            command, self.log_root / "network_specs" / f"{case.case_id}.log",
            timeout=300,
        )
        return output

    def wait_for_network(self, process: subprocess.Popen[bytes], log_path: Path, case: OnlineCase) -> None:
        needle = f"network 'binding-async-merge-{case.case_id}' ready"
        deadline = time.monotonic() + self.args.network_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = log_path.read_text(encoding="utf-8", errors="replace")
                raise ExperimentError(f"network exited before ready:\n{output[-20000:]}")
            if needle in log_path.read_text(encoding="utf-8", errors="replace"):
                return
            time.sleep(0.5)
        output = log_path.read_text(encoding="utf-8", errors="replace")
        raise ExperimentError(f"network readiness timeout:\n{output[-20000:]}")

    def start_network(self, case: OnlineCase, deployment: str, network: Path) -> subprocess.Popen[bytes]:
        if self.network_process is not None:
            raise ExperimentError("network already running")
        busy_flag = "--busy-wait" if case.busy_wait else "--no-busy-wait"
        command = [
            *self.python_command(
                "-m", "scripts.online.benchmark.launch_network", "local",
                "--network", str(network), "--port", str(PORT),
                "--advertise-host", "127.0.0.1",
                "--async-backend", case.async_backend, busy_flag,
                "--container", case.container,
                "--place", "dealer-0", "0", "0", "cuda:0",
            )
        ]
        if deployment == "persistent-agent":
            command.extend(("--agent", AGENT, "dealer-0"))
        log_path = self.log_root / "online" / case.case_id / deployment / "network.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, env=self.base_env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        self.network_process = process
        self.record(command, log=str(log_path), pid=process.pid)
        self.wait_for_network(process, log_path, case)
        return process

    def stop_network(self) -> None:
        process = self.network_process
        self.network_process = None
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
            process.wait(timeout=10)

    @staticmethod
    def descendant_pids(root_pid: int) -> set[int]:
        descendants = {root_pid}
        changed = True
        while changed:
            changed = False
            try:
                entries = list(Path("/proc").iterdir())
            except OSError:
                return descendants
            for entry in entries:
                if not entry.name.isdigit():
                    continue
                try:
                    fields = (entry / "stat").read_text(encoding="utf-8").split()
                    pid = int(fields[0])
                    parent = int(fields[3])
                except (OSError, ValueError, IndexError):
                    continue
                if parent in descendants and pid not in descendants:
                    descendants.add(pid)
                    changed = True
        return descendants

    @staticmethod
    def process_counters(pid: int) -> dict[str, int] | None:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
            cpu_ticks = int(stat[13]) + int(stat[14])
            voluntary = 0
            involuntary = 0
            for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("voluntary_ctxt_switches:"):
                    voluntary = int(line.split(":", 1)[1])
                elif line.startswith("nonvoluntary_ctxt_switches:"):
                    involuntary = int(line.split(":", 1)[1])
            return {
                "cpu_ticks": cpu_ticks,
                "voluntary_context_switches": voluntary,
                "involuntary_context_switches": involuntary,
            }
        except (OSError, ValueError, IndexError):
            return None

    def process_tree_counters(self, roots: Iterable[int]) -> dict[int, dict[str, int]]:
        pids: set[int] = set()
        for root in roots:
            pids.update(self.descendant_pids(root))
        counters = {}
        for pid in pids:
            value = self.process_counters(pid)
            if value is not None:
                counters[pid] = value
        return counters

    def monitor_processes(
        self,
        roots: Sequence[int],
        output: Path,
        *,
        duration: float | None = None,
        process: subprocess.Popen[bytes] | None = None,
    ) -> dict[str, Any]:
        if (duration is None) == (process is None):
            raise ExperimentError("monitor requires exactly one stop condition")
        started = time.monotonic()
        started_ns = time.time_ns()
        deadline = started + (duration if duration is not None else self.args.timeout)
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
            current = self.process_tree_counters(roots)
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
                **{name: sum(item[name] for item in current.values()) for name in deltas},
            })
            if duration is not None and now >= deadline:
                break
            if process is not None and process.poll() is not None:
                break
            if process is not None and now >= deadline:
                timed_out = True
                break
            time.sleep(self.args.monitor_interval)
        wall = time.monotonic() - started
        clock_ticks = int(os.sysconf("SC_CLK_TCK"))
        result = {
            "schema_version": 1,
            "root_pids": list(roots),
            "started_at_ns": started_ns,
            "finished_at_ns": time.time_ns(),
            "sample_interval_seconds": self.args.monitor_interval,
            "wall_seconds": wall,
            "cpu_seconds": deltas["cpu_ticks"] / clock_ticks,
            "average_cpu_cores": deltas["cpu_ticks"] / clock_ticks / wall if wall else None,
            "voluntary_context_switches": deltas["voluntary_context_switches"],
            "involuntary_context_switches": deltas["involuntary_context_switches"],
            "timed_out": timed_out,
            "measurement_note": (
                "Linux process-tree CPU time and scheduler context switches. "
                "Context switches are an OS proxy, not event-loop wakeup counts."
            ),
            "samples": samples,
        }
        write_json(output, result)
        return result

    def start_gpu_monitor(self, output: Path) -> subprocess.Popen[bytes] | None:
        if not self.args.gpu_monitor:
            return None
        executable = shutil.which("nvidia-smi")
        if executable is None:
            raise ExperimentError("nvidia-smi is required for GPU monitoring")
        output.parent.mkdir(parents=True, exist_ok=True)
        query = "timestamp,index,utilization.gpu,memory.used,memory.free,clocks.sm,power.draw"
        command = (
            executable, f"--id={self.args.gpu}", f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.args.gpu_monitor_interval_ms}",
        )
        with output.open("wb") as stream:
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        self.record(command, log=str(output), pid=process.pid, monitor=True)
        return process

    @staticmethod
    def stop_monitor(process: subprocess.Popen[bytes] | None) -> None:
        if process is None or process.poll() is not None:
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

    def run_online_benchmark(
        self,
        case: OnlineCase,
        deployment: str,
        network_process: subprocess.Popen[bytes],
        output_root: Path,
    ) -> None:
        result = output_root / "result.json"
        busy_flag = "--busy-wait" if case.busy_wait else "--no-busy-wait"
        command = [
            *self.python_command(
                "-m", "scripts.online.benchmark.benchmark_infer",
                "--measurement", "decode", "--orbit", ORBIT,
                "--host-id", f"binding-async-merge-{case.case_id}-{deployment}",
                "--advertise-host", "127.0.0.1", "--model", MODEL,
                "--expected-dealers", "1", "--output", str(output_root),
                *self.common_decode_arguments(), "--max-model-len", "256",
                "--async-backend", case.async_backend, busy_flag,
            )
        ]
        if deployment == "persistent-agent":
            command.extend(("--agent", AGENT))
        log_path = self.log_root / "online" / case.case_id / deployment / "benchmark.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, env=self.base_env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        self.record(command, log=str(log_path), pid=process.pid)
        gpu_monitor = self.start_gpu_monitor(output_root / "gpu_monitor.csv")
        try:
            metrics = self.monitor_processes(
                (network_process.pid, process.pid),
                output_root / "process_metrics.json", process=process,
            )
            if metrics["timed_out"]:
                raise subprocess.TimeoutExpired(command, self.args.timeout)
            returncode = process.wait(timeout=20)
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            raise ExperimentError(f"online benchmark timeout: {case.case_id}/{deployment}") from error
        finally:
            self.stop_monitor(gpu_monitor)
        self.commands[-2 if gpu_monitor is not None else -1].update({
            "returncode": returncode,
            "wall_seconds": time.time() - started,
        })
        if returncode != 0:
            output = log_path.read_text(encoding="utf-8", errors="replace")
            raise ExperimentError(
                f"online benchmark failed ({returncode}): "
                f"{case.case_id}/{deployment}\n{output[-20000:]}"
            )
        audit_graph_result(result, case.merge_limit)

    def run_online(self) -> None:
        selected = set(self.args.cases)
        for case in ONLINE_CASES:
            if case.case_id not in selected:
                continue
            network = self.generate_network(case)
            for deployment in DEPLOYMENTS:
                output_root = self.result_root / "online" / case.case_id / deployment
                result = output_root / "result.json"
                case_path = output_root / "case.json"
                required = [case_path, output_root / "process_metrics.json"]
                if case.is_async_scan and self.args.idle_seconds:
                    required.append(output_root / "idle_process_metrics.json")
                if self.args.gpu_monitor:
                    required.append(output_root / "gpu_monitor.csv")
                    if case.is_async_scan and self.args.idle_seconds:
                        required.append(output_root / "idle_gpu_monitor.csv")
                if (
                    self.complete(result, case.merge_limit)
                    and all(path.is_file() for path in required)
                ):
                    continue
                network_process = None
                try:
                    network_process = self.start_network(case, deployment, network)
                    if case.is_async_scan and self.args.idle_seconds:
                        idle_gpu = self.start_gpu_monitor(output_root / "idle_gpu_monitor.csv")
                        try:
                            self.monitor_processes(
                                (network_process.pid,),
                                output_root / "idle_process_metrics.json",
                                duration=self.args.idle_seconds,
                            )
                        finally:
                            self.stop_monitor(idle_gpu)
                    self.run_online_benchmark(
                        case, deployment, network_process, output_root,
                    )
                    write_json(case_path, {
                        "schema_version": 1,
                        "case": asdict(case),
                        "deployment": deployment,
                        "network_spec": str(network),
                        "network_spec_snapshot": read_json(network),
                        "sequence_counts": list(SEQ_COUNTS),
                        "runs": self.args.runs,
                        "loaded_idle_seconds": (
                            self.args.idle_seconds if case.is_async_scan else 0
                        ),
                    })
                finally:
                    self.stop_network()

    def capture_environment(self) -> None:
        def output(*command: str) -> str:
            try:
                return subprocess.run(
                    command, cwd=REPO_ROOT, env=self.base_env, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=30, check=False,
                ).stdout.strip()
            except (OSError, subprocess.TimeoutExpired) as error:
                return repr(error)

        previous_commands = []
        previous_invocations = []
        if self.environment_path.is_file():
            try:
                previous = read_json(self.environment_path)
                previous_commands = list(previous.get("commands", []))
                previous_invocations = list(previous.get("invocations", []))
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        invocation = command_text((str(self.python), *sys.argv))
        write_json(self.environment_path, {
            "schema_version": 1,
            "run_id": self.args.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "plan": build_plan(),
            "invocations": previous_invocations + [invocation],
            "git_commit": output("git", "rev-parse", "HEAD"),
            "git_status": output("git", "status", "--short"),
            "python": output(str(self.python), "--version"),
            "torch": output(
                str(self.python), "-c",
                "import torch; print(torch.__version__, torch.version.cuda)",
            ),
            "gpus": output(
                "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,power.limit",
                "--format=csv,noheader",
            ),
            "parameters": {
                "runs": self.args.runs,
                "warmup_requests": self.args.warmup_requests,
                "warmup_decode_tokens": self.args.warmup_decode_tokens,
                "decode_tokens": self.args.decode_tokens,
                "idle_seconds": self.args.idle_seconds,
                "monitor_interval": self.args.monitor_interval,
                "gpu_monitor_interval_ms": self.args.gpu_monitor_interval_ms,
                "selected_cases": list(self.args.cases),
            },
            "commands": previous_commands + self.commands,
        })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", nargs="?", default="all",
        choices=("plan", "baseline", "online", "analyze", "all"),
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--vllm-model", type=Path, default=DEFAULT_VLLM_MODEL)
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--warmup-decode-tokens", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=200)
    parser.add_argument("--idle-seconds", type=float, default=30.0)
    parser.add_argument("--monitor-interval", type=float, default=0.2)
    parser.add_argument("--gpu-monitor-interval-ms", type=int, default=500)
    parser.add_argument(
        "--gpu-monitor", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--uclamp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--network-timeout", type=float, default=900)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--cases", nargs="+",
        choices=tuple(case.case_id for case in ONLINE_CASES),
        default=tuple(case.case_id for case in ONLINE_CASES),
    )
    args = parser.parse_args()
    if args.gpu < 0 or args.runs <= 0 or args.decode_tokens <= 0:
        parser.error("GPU index must be non-negative; runs/decode tokens must be positive")
    if args.warmup_requests < 0 or args.warmup_decode_tokens < 0:
        parser.error("warmup counts must be non-negative")
    if args.idle_seconds < 0 or args.monitor_interval <= 0:
        parser.error("idle seconds must be non-negative and monitor interval positive")
    if args.gpu_monitor_interval_ms <= 0 or args.timeout <= 0 or args.network_timeout <= 0:
        parser.error("monitor interval and timeouts must be positive")
    args.cases = tuple(dict.fromkeys(args.cases))
    return args


def main() -> int:
    args = parse_args()
    if args.phase == "plan":
        print(json.dumps(build_plan(), indent=2, ensure_ascii=False))
        return 0
    runner = Runner(args)
    try:
        if args.phase in {"baseline", "all"}:
            runner.run_baselines()
        if args.phase in {"online", "all"}:
            runner.run_online()
        if args.phase in {"analyze", "all"}:
            runner.run_logged(
                runner.python_command(
                    str(EXP_ROOT / "analyze.py"), str(runner.result_root),
                ),
                runner.log_root / "analyze.log",
                timeout=300,
            )
    finally:
        runner.stop_network()
        runner.capture_environment()
    print(f"results: {runner.result_root}")
    print(f"environment: {runner.environment_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
