#!/usr/bin/env python3
"""Run the Llama2-7B concurrency/scalability reproduction."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/root/miniconda3/bin/python")
MODEL = "teared-Llama-2-7B-GPTQ"
AGENT = "infer-scale-agent"
PORT = 7270
ORBIT = f"127.0.0.1:{PORT}"
UCLAMPSET = Path("/usr/bin/uclampset")
DEPLOYMENTS = ("combined-request-agent", "persistent-agent")
MAJOR_PERCENTAGE = 0.99
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
MAXIMUM_KEEP_ALIVES = 4096


def utc_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def command_text(command: Sequence[str]) -> str:
    import shlex
    return shlex.join(command)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


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
    if isinstance(value, bool):
        return 0
    if isinstance(value, dict):
        return sum(
            integer_leaf_total(item) if name == key
            else counter_total(item, key)
            for name, item in value.items()
        )
    if isinstance(value, list):
        return sum(counter_total(item, key) for item in value)
    return 0


class ExperimentError(RuntimeError):
    pass


class Runner:
    def __init__(
        self,
        *,
        python: Path,
        run_id: str,
        clients: tuple[int, ...],
        scales: tuple[int, ...],
        deployments: tuple[str, ...],
        total_tokens: int,
        tokens_per_client_cap: int,
        timeout: float,
        resume: bool,
        metrics_events: str,
        metrics_sample_every: int,
        monitor_interval_ms: int,
    ) -> None:
        # Do not resolve this symlink: resolving a venv interpreter to its base
        # executable discards the venv prefix and therefore its packages.
        self.python = Path(os.path.abspath(python))
        self.run_id = run_id
        self.clients = clients
        self.scales = scales
        self.deployments = deployments
        self.total_tokens = total_tokens
        self.tokens_per_client_cap = tokens_per_client_cap
        self.timeout = timeout
        self.resume = resume
        self.metrics_events = metrics_events
        self.metrics_sample_every = metrics_sample_every
        self.monitor_interval_ms = monitor_interval_ms
        self.result_root = EXP_ROOT / "results" / run_id
        self.log_root = EXP_ROOT / "logs" / run_id
        self.environment_path = EXP_ROOT / "environment" / f"{run_id}.json"
        self.network_dir = EXP_ROOT / "networks"
        self.commands: list[dict[str, Any]] = []
        self.processes: list[subprocess.Popen[bytes]] = []
        self.base_env = os.environ.copy()
        self.base_env["PYTHONPATH"] = str(REPO_ROOT) + (
            f":{self.base_env['PYTHONPATH']}"
            if self.base_env.get("PYTHONPATH") else ""
        )
        self.base_env["PYTHONUNBUFFERED"] = "1"
        # With 16 decoder layers each scalable KV block is 8 MiB.  The native
        # caching allocator otherwise places those allocations in 20 MiB
        # segments, stranding roughly 4 MiB per pair of blocks.
        self.base_env["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF
        # scale=8 with 128 clients creates at least 1024 simultaneous path
        # setup messages before the Go backend reports frame releases.  Keep
        # the library's conservative default and raise it only for this
        # explicitly bounded matrix.
        self.base_env["DROFELL_MAXIMUM_KEEP_ALIVES"] = str(MAXIMUM_KEEP_ALIVES)
        if metrics_events:
            self.base_env["DROFELL_METRICS_PATH"] = str(
                self.result_root / "runtime_metrics.jsonl"
            )
            self.base_env["DROFELL_METRICS_EVENTS"] = metrics_events
            self.base_env["DROFELL_METRICS_SAMPLE_EVERY"] = str(
                metrics_sample_every
            )
        else:
            self.base_env.pop("DROFELL_METRICS_PATH", None)
            self.base_env.pop("DROFELL_METRICS_EVENTS", None)
            self.base_env.pop("DROFELL_METRICS_SAMPLE_EVERY", None)

    def record(self, command: Sequence[str], **extra: Any) -> None:
        self.commands.append({"command": command_text(command), **extra})

    def uclamped_python(self, *args: str) -> tuple[str, ...]:
        return (
            str(UCLAMPSET), "-m", "1024", "--", str(self.python), *args
        )

    def run(
        self,
        command: Sequence[str],
        *,
        log_path: Path | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if log_path is None:
            completed = subprocess.run(
                list(command), cwd=REPO_ROOT, env=self.base_env,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=timeout, check=False,
            )
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    list(command), cwd=REPO_ROOT, env=self.base_env,
                    text=True, stdout=log, stderr=subprocess.STDOUT,
                    timeout=timeout, check=False,
                )
        self.record(
            command,
            returncode=completed.returncode,
            log=str(log_path) if log_path else None,
        )
        if check and completed.returncode != 0:
            output = completed.stdout or (
                log_path.read_text(encoding="utf-8", errors="replace")[-16000:]
                if log_path else ""
            )
            raise ExperimentError(
                f"command failed ({completed.returncode}): "
                f"{command_text(command)}\n{output}"
            )
        return completed

    def start(
        self, command: Sequence[str], log_path: Path
    ) -> subprocess.Popen[bytes]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("wb")
        process = subprocess.Popen(
            list(command), cwd=REPO_ROOT, env=self.base_env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        log.close()
        self.processes.append(process)
        self.record(command, pid=process.pid, log=str(log_path))
        return process

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
                text = log_path.read_text(encoding="utf-8", errors="replace")
                raise ExperimentError(
                    f"network exited before {needle!r}:\n{text[-16000:]}"
                )
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if needle in text:
                return
            time.sleep(0.5)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        raise ExperimentError(f"timeout waiting for {needle!r}:\n{text[-16000:]}")

    def stop_process(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        if process in self.processes:
            self.processes.remove(process)

    def stop_all(self) -> None:
        for process in list(reversed(self.processes)):
            self.stop_process(process)

    def network_path(self, scale: int) -> Path:
        return self.network_dir / f"llama2-7b-infer-scale-{scale}gpu.json"

    def max_seq_len(self) -> int:
        # The longest measured trace is clients=1. Reserve 128 positions for
        # the prompt, the un-timed first token, and block rounding.
        return min(self.total_tokens, self.tokens_per_client_cap) + 128

    def generate_network(self, scale: int) -> None:
        common = [
            "-m", "scripts.online.benchmark.generate_network",
            "--name", f"llama2-7b-infer-scale-{scale}gpu-graph",
            "--model", MODEL,
            "--decoder-backend", "marlin",
            "--max-seq-len", str(self.max_seq_len()),
            "--max-concurrency", "8", "--cuda-graph",
            "--memory-policy", "percentage",
            "--major-percentage", str(MAJOR_PERCENTAGE),
            "--unmanaged-reserved-mib", "32",
            "--minor-min-mib", "0", "--minor-max-mib", "0",
            "--operate-reserved-mib", "512",
            "--graph-budget-mib", "512",
        ]
        if scale not in (1, 2, 4, 8) or 32 % scale:
            raise ValueError(f"unsupported scale: {scale}")
        group_size = 32 // scale
        common += [
            "--decoder-group-sizes",
            *(str(group_size) for _ in range(scale)),
        ]
        for index in range(scale):
            common += ["--dealer", f"dealer-g{index}={index}"]
        output = self.network_path(scale)
        common += ["--output", str(output)]
        self.run(self.uclamped_python(*common))
        snapshot = self.result_root / "network_specs" / output.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            if snapshot.read_bytes() != output.read_bytes():
                raise ExperimentError(
                    "archived network spec differs from regenerated spec: "
                    f"{snapshot}"
                )
            return
        shutil.copy2(output, snapshot)

    def prepare(self) -> None:
        for scale in self.scales:
            self.generate_network(scale)

    def launch_network(self, scale: int, deployment: str) -> None:
        log_dir = self.log_root / f"scale-{scale}" / deployment
        args = [
            "-m", "scripts.online.benchmark.launch_network", "local",
            "--network", str(self.network_path(scale)),
            "--port", str(PORT), "--advertise-host", "127.0.0.1",
            "--async-backend", "golang", "--busy-wait",
        ]
        for index in range(scale):
            args += [
                "--place", f"dealer-g{index}", str(index), "0",
                f"cuda:{index}",
            ]
        if deployment == "persistent-agent":
            args += ["--agent", AGENT, "dealer-g0"]
        log_path = self.unique_log_path(log_dir / "network.log")
        process = self.start(self.uclamped_python(*args), log_path)
        self.wait_for_text(process, log_path, "ready at", 900)

    def output_tokens_per_client(self, clients: int) -> int:
        return min(
            self.total_tokens // clients,
            self.tokens_per_client_cap,
        )

    def budget_capped(self, clients: int) -> bool:
        return self.total_tokens // clients > self.tokens_per_client_cap

    @staticmethod
    def case_complete(case_dir: Path) -> bool:
        required = (
            "args.json", "result.json", "phases.json", "tokens.json",
            "decode_graph.json", "gpu_monitor.csv", "case.json",
        )
        return all((case_dir / name).is_file() for name in required)

    @staticmethod
    def unique_log_path(path: Path) -> Path:
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

    def case_should_run(self, case_dir: Path) -> bool:
        if not case_dir.exists():
            return True
        if not self.resume:
            raise ExperimentError(
                f"case output already exists (use --resume): {case_dir}"
            )
        if self.case_complete(case_dir):
            self.audit_case(case_dir)
            print(f"resume: skip complete case {case_dir}")
            return False
        relative = case_dir.relative_to(self.result_root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        archived = self.result_root / "failed_attempts" / stamp / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(case_dir), str(archived))
        print(f"resume: archived partial case to {archived}")
        return True

    def start_gpu_monitor(self, path: Path) -> subprocess.Popen[bytes]:
        path.parent.mkdir(parents=True, exist_ok=True)
        log = path.open("wb")
        command = (
            "nvidia-smi",
            "--query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,"
            "pstate,clocks.current.sm,power.draw",
            "--format=csv,noheader,nounits", "-lms",
            str(self.monitor_interval_ms),
        )
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=self.base_env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        log.close()
        self.record(command, pid=process.pid, log=str(path), monitor=True)
        return process

    @staticmethod
    def stop_gpu_monitor(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

    def start_system_monitors(
        self, case_dir: Path
    ) -> list[subprocess.Popen[bytes]]:
        if not self.metrics_events:
            return []
        specs: list[tuple[tuple[str, ...], Path]] = []
        if shutil.which("pidstat", path=self.base_env.get("PATH")):
            specs.append((
                ("pidstat", "-d", "-u", "-r", "-w", "-t", "1"),
                case_dir / "pidstat.log",
            ))
        if shutil.which("mpstat", path=self.base_env.get("PATH")):
            specs.append((
                ("mpstat", "-P", "ALL", "1"),
                case_dir / "mpstat.log",
            ))
        processes: list[subprocess.Popen[bytes]] = []
        for command, path in specs:
            path.parent.mkdir(parents=True, exist_ok=True)
            log = path.open("wb")
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, env=self.base_env,
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log.close()
            self.record(command, pid=process.pid, log=str(path), monitor=True)
            processes.append(process)
        return processes

    def audit_case(self, case_dir: Path) -> dict[str, Any]:
        args = json.loads((case_dir / "args.json").read_text(encoding="utf-8"))
        phases = json.loads(
            (case_dir / "phases.json").read_text(encoding="utf-8")
        )
        graph = json.loads(
            (case_dir / "decode_graph.json").read_text(encoding="utf-8")
        )
        record = json.loads(
            (case_dir / "result.json").read_text(encoding="utf-8")
        )
        # CLI --logprobs 0 is normalized to None in InferBenchmarkConfig;
        # phases.json serializes that disabled state back to the public value 0.
        if args["benchmark"]["logprobs"] is not None:
            raise ExperimentError(f"logprobs was not disabled: {case_dir}")
        if phases["parameters"]["logprobs"] != 0:
            raise ExperimentError(f"reported logprobs was not 0: {case_dir}")
        if args["benchmark"]["disable_gc"]:
            raise ExperimentError(f"GC was disabled: {case_dir}")
        if phases["deployment"] != case_dir.name:
            raise ExperimentError(f"deployment mismatch: {case_dir}")
        clients = args["benchmark"]["parallel_seqs"]
        output_tokens = args["benchmark"]["max_tokens"] - 1
        expected_records = clients * output_tokens
        counts = Counter(int(item[0]) for item in record)
        if len(record) != expected_records or counts != Counter({
            client: output_tokens for client in range(clients)
        }):
            raise ExperimentError(f"incomplete token record: {case_dir}")
        if any(
            int(left[1]) > int(right[1])
            for left, right in zip(record, record[1:])
        ):
            raise ExperimentError(f"non-monotonic token timestamps: {case_dir}")
        for dealer, stats in graph.items():
            if not stats.get("enabled"):
                raise ExperimentError(f"graph disabled for {dealer}: {case_dir}")
            if stats.get("configured_batch_sizes") != list(range(1, 9)):
                raise ExperimentError(
                    f"unexpected graph batch sizes for {dealer}: {case_dir}"
                )
        fallbacks = counter_total(graph, "fallbacks")
        fused_replays = counter_total(graph, "fused_replays")
        if fallbacks:
            raise ExperimentError(f"graph fallback count {fallbacks}: {case_dir}")
        if fused_replays <= 0:
            raise ExperimentError(f"no fused graph replay: {case_dir}")
        return {
            "graph_fallbacks": fallbacks,
            "fused_replays_snapshot": fused_replays,
        }

    def run_case(
        self,
        *,
        scale: int,
        clients: int,
        deployment: str,
        smoke_tokens: int | None = None,
    ) -> None:
        base = self.result_root if smoke_tokens is None else self.result_root / "smoke"
        case_dir = (
            base / f"scale-{scale}" / f"clients-{clients}" / deployment
        )
        if not self.case_should_run(case_dir):
            return
        output_tokens = (
            smoke_tokens if smoke_tokens is not None
            else self.output_tokens_per_client(clients)
        )
        max_tokens = output_tokens + 1  # prefill returns the first output token.
        args = [
            "-m", "scripts.online.benchmark.benchmark_infer",
            "--measurement", "request", "--orbit", ORBIT,
            "--host-id", f"infer-scale-{scale}-{clients}-{deployment}",
            "--advertise-host", "127.0.0.1",
            "--model", MODEL, "--output", str(case_dir),
            "--prompt", "once upon a time",
            "--seqs", str(clients), "--max-tokens", str(max_tokens),
            "--warmup-tokens", "32", "--num-paths", "1",
            "--logprobs", "0", "--no-disable-gc",
            "--progress-interval", "60",
            "--async-backend", "golang", "--busy-wait",
        ]
        if deployment == "persistent-agent":
            args += ["--agent", AGENT]
        monitor = self.start_gpu_monitor(case_dir / "gpu_monitor.csv")
        system_monitors = self.start_system_monitors(case_dir)
        started = time.time()
        try:
            try:
                self.run(
                    self.uclamped_python(*args),
                    log_path=self.unique_log_path(
                        self.log_root / ("smoke" if smoke_tokens is not None else "")
                        / f"scale-{scale}" / deployment / f"clients-{clients}.log"
                    ),
                    timeout=self.timeout,
                )
            finally:
                self.stop_gpu_monitor(monitor)
                for system_monitor in system_monitors:
                    self.stop_gpu_monitor(system_monitor)
            audit = self.audit_case(case_dir)
        except BaseException as exc:
            write_json(case_dir / "failure.json", {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scale": scale,
                "clients": clients,
                "deployment": deployment,
                "smoke": smoke_tokens is not None,
                "output_tokens_per_client": output_tokens,
                "wall_seconds": time.time() - started,
            })
            raise
        capped = smoke_tokens is None and self.budget_capped(clients)
        actual_total_tokens = output_tokens * clients
        network_spec = json.loads(
            self.network_path(scale).read_text(encoding="utf-8")
        )
        write_json(case_dir / "case.json", {
            "scale": scale,
            "clients": clients,
            "deployment": deployment,
            # Keep the old field as an alias for readers of archived runs.  It
            # has always meant the number of timed tokens actually requested.
            "total_output_token_budget": actual_total_tokens,
            "nominal_total_output_token_budget": (
                self.total_tokens if smoke_tokens is None else None
            ),
            "actual_total_output_token_budget": actual_total_tokens,
            "tokens_per_client_cap": self.tokens_per_client_cap,
            "capped": capped,
            "output_tokens_per_client": output_tokens,
            "max_tokens_argument": max_tokens,
            "network_max_seq_len": self.max_seq_len(),
            "spec_sha256": network_spec["spec_sha256"],
            "memory_policy": {
                "type": "percentage", "major_percentage": MAJOR_PERCENTAGE,
                "operate_reserved_mib": 512,
                "graph_budget_mib": 512,
                "unmanaged_reserved_mib": 32,
                "minor_min_mib": 0, "minor_max_mib": 0,
            },
            "wall_seconds": time.time() - started,
            "smoke": smoke_tokens is not None,
            **audit,
        })

    def run_matrix(self, smoke_tokens: int | None = None) -> None:
        smoke_clients = (
            tuple(dict.fromkeys((self.clients[0], self.clients[-1])))
            if smoke_tokens is not None else self.clients
        )
        for scale in self.scales:
            for deployment in self.deployments:
                self.launch_network(scale, deployment)
                try:
                    # Shorter high-concurrency cases fail fast before the long
                    # clients=1 trace. Preserve the user's requested order in
                    # metadata but execute descending concurrency.
                    for clients in sorted(smoke_clients, reverse=True):
                        self.run_case(
                            scale=scale, clients=clients,
                            deployment=deployment, smoke_tokens=smoke_tokens,
                        )
                finally:
                    self.stop_all()

    def capture_environment(self) -> None:
        def output(*command: str) -> str:
            try:
                result = subprocess.run(
                    command, cwd=REPO_ROOT, env=self.base_env, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    check=False,
                )
            except FileNotFoundError:
                return f"unavailable: {command[0]}"
            return result.stdout.strip()

        previous_commands: list[dict[str, Any]] = []
        if self.environment_path.exists():
            try:
                previous = json.loads(
                    self.environment_path.read_text(encoding="utf-8")
                )
                previous_commands = list(previous.get("commands", []))
            except (json.JSONDecodeError, OSError, TypeError):
                pass
        governors = {}
        for path in sorted(Path("/sys/devices/system/cpu").glob(
            "cpu[0-9]*/cpufreq/scaling_governor"
        )):
            governors[path.parts[-3]] = path.read_text().strip()
        write_json(self.environment_path, {
            "run_id": self.run_id,
            "git_commit": output("git", "rev-parse", "HEAD"),
            "git_status": output("git", "status", "--short"),
            "python": output(str(self.python), "--version"),
            "torch": output(
                str(self.python), "-c",
                "import torch; print(torch.__version__, torch.version.cuda)",
            ),
            "gpus": output(
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,power.limit",
                "--format=csv,noheader",
            ),
            "gpu_topology": output("nvidia-smi", "topo", "-m"),
            "lscpu": output("lscpu"),
            "numa": output("numactl", "--hardware"),
            "pci_tree": output("lspci", "-tv"),
            "uclampset": output("uclampset", "--version"),
            "uclamp_probe": output(
                str(UCLAMPSET), "-m", "1024", "--", "sh", "-c",
                "grep -E 'uclamp.min|uclamp.max' /proc/self/sched",
            ),
            "cpu_governors": governors,
            "parameters": {
                "model": MODEL,
                "clients": self.clients,
                "scales": self.scales,
                "deployments": self.deployments,
                "total_output_token_budget": self.total_tokens,
                "tokens_per_client_cap": self.tokens_per_client_cap,
                "token_budget_semantics": (
                    "min(total_output_token_budget // clients, "
                    "tokens_per_client_cap) per client"
                ),
                "max_concurrency": 8,
                "decoder_backend": "marlin",
                "cuda_graph": True,
                "pytorch_cuda_alloc_conf": PYTORCH_CUDA_ALLOC_CONF,
                "maximum_keep_alives": MAXIMUM_KEEP_ALIVES,
                "logprobs": 0,
                "gc_enabled": True,
                "warmup_tokens": 32,
                "metrics_events": self.metrics_events,
                "metrics_sample_every": self.metrics_sample_every,
                "gpu_monitor_interval_ms": self.monitor_interval_ms,
            },
            "commands": previous_commands + self.commands,
        })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", nargs="?", default="all",
        choices=("prepare", "smoke", "run", "analyze", "all"),
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--clients", nargs="+", type=positive_int, default=(1, 16, 32))
    parser.add_argument(
        "--scales", nargs="+", type=int, choices=(1, 2, 4, 8),
        default=(1, 2),
    )
    parser.add_argument("--deployment", choices=DEPLOYMENTS)
    parser.add_argument("--total-tokens", type=positive_int, default=35072)
    parser.add_argument(
        "--tokens-per-client-cap", type=positive_int, default=36864,
    )
    parser.add_argument("--smoke-tokens", type=positive_int, default=64)
    parser.add_argument("--timeout", type=float, default=6 * 60 * 60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--metrics-events", default="",
        help="comma-separated opt-in event categories (transport,transfer,sync,batch)",
    )
    parser.add_argument(
        "--metrics-sample-every", type=positive_int, default=1,
    )
    parser.add_argument(
        "--monitor-interval-ms", type=positive_int, default=1000,
    )
    args = parser.parse_args()
    if not args.python.is_file():
        parser.error(f"Python interpreter does not exist: {args.python}")
    if not UCLAMPSET.is_file():
        parser.error(f"uclampset does not exist: {UCLAMPSET}")
    if len(set(args.clients)) != len(args.clients):
        parser.error("--clients values must be unique")
    if len(set(args.scales)) != len(args.scales):
        parser.error("--scales values must be unique")
    if any(args.total_tokens // clients < 1 for clients in args.clients):
        parser.error("--total-tokens must provide at least one token per client")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> None:
    args = parse_args()
    runner = Runner(
        python=args.python,
        run_id=args.run_id,
        clients=tuple(args.clients),
        scales=tuple(args.scales),
        deployments=(args.deployment,) if args.deployment else DEPLOYMENTS,
        total_tokens=args.total_tokens,
        tokens_per_client_cap=args.tokens_per_client_cap,
        timeout=args.timeout,
        resume=args.resume,
        metrics_events=args.metrics_events,
        metrics_sample_every=args.metrics_sample_every,
        monitor_interval_ms=args.monitor_interval_ms,
    )
    try:
        if args.phase in {"prepare", "smoke", "run", "all"}:
            runner.prepare()
        if args.phase in {"smoke", "all"}:
            runner.run_matrix(smoke_tokens=args.smoke_tokens)
        if args.phase in {"run", "all"}:
            runner.run_matrix()
        if args.phase in {"analyze", "all"}:
            runner.run(runner.uclamped_python(
                "-m", "exps.infer_scale.analyze",
                "--run-dir", str(runner.result_root),
            ))
    finally:
        runner.stop_all()
        runner.capture_environment()
    print(f"run-id: {runner.run_id}")
    print(f"results: {runner.result_root}")
    print(f"environment: {runner.environment_path}")


if __name__ == "__main__":
    main()
