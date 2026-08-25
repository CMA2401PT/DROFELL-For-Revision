#!/usr/bin/env python3
"""Measure ShadowState overhead across model and decoder-shard sizes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import statistics
import subprocess
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/home/dai/projects/LLM/dflash_related/.venv/bin/python")
MODEL_ROOT = REPO_ROOT / "split_qllm_models"
UCLAMPSET = Path("/usr/bin/uclampset")
PROMPT = "once upon a time"


@dataclass(frozen=True)
class Case:
    name: str
    model: str
    num_layers: int
    shard_layers: int
    group_sizes: tuple[int, ...]
    placements: tuple[tuple[str, int, str], ...]
    memory_policy: str
    major_percentage: float
    operate_reserved_mib: int
    graph_budget_mib: int


CASES = (
    Case("7b-shard32", "teared-Llama-2-7B-GPTQ", 32, 32, (32,),
         (("dealer-g0", 0, "cuda:0"),), "static", 0.8, 1024, 1024),
    Case("7b-shard16", "teared-Llama-2-7B-GPTQ", 32, 16, (16, 16),
         (("dealer-g0", 0, "cuda:0"), ("dealer-g1", 0, "cuda:0")),
         "static", 0.8, 1024, 1024),
    Case("7b-shard8", "teared-Llama-2-7B-GPTQ", 32, 8, (8, 8, 8, 8),
         tuple((f"dealer-g{i}", 0, "cuda:0") for i in range(4)),
         "static", 0.8, 1024, 1024),
    Case("70b-shard40", "teared-Llama-2-70B-GPTQ", 80, 40, (40, 40),
         (("dealer-g0", 0, "cuda:0"), ("dealer-g1", 1, "cuda:1")),
         "percentage", 0.5, 100, 100),
    Case("70b-shard20", "teared-Llama-2-70B-GPTQ", 80, 20,
         (20, 20, 20, 20),
         (("dealer-g0", 0, "cuda:0"), ("dealer-g1", 0, "cuda:0"),
          ("dealer-g2", 1, "cuda:1"), ("dealer-g3", 1, "cuda:1")),
         "percentage", 0.5, 100, 100),
)
METRIC_NAMES = (
    "metadata_bytes", "prompt_bytes", "generating_bytes",
    "pending_task_bytes", "kv_bytes", "total_retained_bytes",
    "update_cpu_ns", "task_input_bytes_observed", "kv_bytes_received",
    "replay_task_count", "checkpoint_block_count",
)


class ExperimentError(RuntimeError):
    pass


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


def capture(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return {
        "command": command_text(command),
        "returncode": completed.returncode,
        "output": completed.stdout,
    }


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def wait_ready(process: subprocess.Popen[bytes], log_path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ExperimentError(
                f"network exited before ready ({process.returncode}):\n"
                f"{read_text(log_path)[-16000:]}"
            )
        text = read_text(log_path)
        if "ready at" in text:
            return
        if "Traceback (most recent call last)" in text:
            raise ExperimentError(f"network startup traceback:\n{text[-16000:]}")
        time.sleep(0.5)
    raise ExperimentError(f"network startup timeout:\n{read_text(log_path)[-16000:]}")


def stop_group(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 30), (signal.SIGTERM, 15),
                         (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def uclamp_snapshot(pid: int) -> str:
    for _ in range(40):
        result = capture(("uclampset", "-p", str(pid)))
        if "min: 1024" in result["output"]:
            return result["output"].strip()
        time.sleep(0.05)
    return result["output"].strip()


def distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None,
                "sample_sd": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def parse_snapshots(case: Case, path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(read_text(path).splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") != "shadow_state_snapshot":
            continue
        metrics = event.get("metrics", {})
        missing = [name for name in METRIC_NAMES if name not in metrics]
        if missing:
            raise ExperimentError(f"snapshot missing {missing}: {path}:{line_number}")
        compute_node = str(event.get("compute_node", ""))
        match = re.search(r"decoders-(\d+)-(\d+)", compute_node)
        observed_layers = (
            int(match.group(2)) - int(match.group(1)) if match else case.shard_layers
        )
        if observed_layers != case.shard_layers:
            raise ExperimentError(
                f"unexpected failed-handle layers {observed_layers} in {compute_node}"
            )
        row = {
            "case": case.name,
            "model": case.model,
            "model_layers": case.num_layers,
            "shard_layers": observed_layers,
            "recovery_id": event.get("recovery_id"),
            "compute_node": compute_node,
            "line_number": line_number,
        }
        row.update({name: int(metrics[name]) for name in METRIC_NAMES})
        rows.append(row)
    return rows


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.python = args.python.absolute()
        self.run_id = args.run_id
        self.result_root = EXP_ROOT / "results" / self.run_id
        self.log_root = EXP_ROOT / "logs" / self.run_id
        self.environment_path = EXP_ROOT / "environment" / f"{self.run_id}.json"
        self.network_root = EXP_ROOT / "networks" / self.run_id
        self.commands: list[dict[str, Any]] = []
        self.case_records: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []
        self.probes: dict[str, Any] = {}
        self.model_configs: dict[str, Any] = {}
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.finished_utc: str | None = None
        selected = set(args.cases)
        self.cases = tuple(case for case in CASES if case.name in selected)

    def initialize(self) -> None:
        if not self.python.is_file():
            raise ExperimentError(f"Python does not exist: {self.python}")
        if not UCLAMPSET.is_file():
            raise ExperimentError(f"uclampset does not exist: {UCLAMPSET}")
        if not port_available(self.args.port):
            raise ExperimentError(f"port is in use: {self.args.port}")
        for path in (self.result_root, self.log_root, self.environment_path,
                     self.network_root):
            if path.exists():
                raise ExperimentError(f"refusing to overwrite: {path}")
        self.result_root.mkdir(parents=True)
        self.log_root.mkdir(parents=True)
        self.network_root.mkdir(parents=True)
        for case in self.cases:
            config = MODEL_ROOT / case.model / "config.json"
            self.model_configs[case.model] = {
                "path": str(config), "sha256": sha256(config),
            }
        self.probes = {
            "git": capture(("git", "rev-parse", "HEAD")),
            "git_status": capture(("git", "status", "--short")),
            "python_torch": capture((
                str(self.python), "-c",
                "import platform,torch; print(platform.python_version()); "
                "print(torch.__version__, torch.version.cuda)",
            )),
            "gpus": capture((
                "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            )),
        }
        self.flush_environment()

    def flush_environment(self) -> None:
        write_json(self.environment_path, {
            "schema_version": 1,
            "run_id": self.run_id,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "python": str(self.python),
            "parameters": {
                "cases": [asdict(case) for case in self.cases],
                "drop_probability": self.args.drop_probability,
                "max_tokens": self.args.max_tokens,
                "warmup_tokens": 32,
                "repeats": 1,
                "inspection_postpone_seconds": 0.3,
                "recovery_replay": "chunk",
                "uclamp_min": 1024,
                "metrics_enabled": True,
            },
            "model_configs": self.model_configs,
            "probes": self.probes,
            "commands": self.commands,
            "case_records": self.case_records,
        })

    def record(self, purpose: str, command: Sequence[str], **extra: Any) -> dict[str, Any]:
        item = {"purpose": purpose, "command": command_text(command), **extra}
        self.commands.append(item)
        self.flush_environment()
        return item

    def generate_network(self, case: Case) -> Path:
        output = self.network_root / f"{case.name}.json"
        command = [
            str(self.python), "-m", "scripts.online.benchmark.generate_network",
            "--name", f"shadow-overhead-{case.name}",
            "--model", case.model,
            "--decoder-group-sizes", *(str(value) for value in case.group_sizes),
        ]
        for index in range(len(case.group_sizes)):
            command.extend(("--dealer", f"dealer-g{index}={index}"))
        command.extend((
            "--decoder-backend", "marlin",
            "--max-seq-len", "1024",
            "--max-concurrency", "1",
            "--cuda-graph", "--failure-injection",
            "--memory-policy", case.memory_policy,
            "--major-reserved-mib", "8192",
            "--major-percentage", str(case.major_percentage),
            "--unmanaged-reserved-mib", "10",
            "--minor-min-mib", "0", "--minor-max-mib", "262144",
            "--operate-reserved-mib", str(case.operate_reserved_mib),
            "--graph-budget-mib", str(case.graph_budget_mib),
            "--output", str(output),
        ))
        completed = subprocess.run(
            command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        self.record(f"generate:{case.name}", command,
                    returncode=completed.returncode, output=completed.stdout)
        if completed.returncode:
            raise ExperimentError(f"network generation failed:\n{completed.stdout}")
        return output

    def run_case(self, case: Case) -> None:
        case_root = self.result_root / case.name
        case_root.mkdir(parents=True)
        metrics_path = case_root / "runtime_metrics.jsonl"
        network_log = self.log_root / f"{case.name}-network.log"
        benchmark_log = self.log_root / f"{case.name}-benchmark.log"
        process: subprocess.Popen[bytes] | None = None
        case_record: dict[str, Any] = {
            "case": case.name, "status": "running", "error": None,
        }
        self.case_records.append(case_record)
        self.flush_environment()
        try:
            network = self.generate_network(case)
            network_command = [
                str(UCLAMPSET), "-m", "1024", "--", str(self.python),
                "-m", "scripts.online.benchmark.launch_network", "local",
                "--network", str(network), "--port", str(self.args.port),
                "--advertise-host", "127.0.0.1",
                "--async-backend", "golang", "--busy-wait",
            ]
            for dealer, process_index, device in case.placements:
                network_command.extend((
                    "--place", dealer, str(process_index), "0", device,
                ))
            network_env = os.environ.copy()
            network_env.update({"PYTHONPATH": str(REPO_ROOT),
                                "PYTHONUNBUFFERED": "1",
                                "RECOVERY_REPLAY": "chunk"})
            network_env.pop("DROFELL_METRICS_PATH", None)
            with network_log.open("wb") as log:
                process = subprocess.Popen(
                    network_command, cwd=REPO_ROOT, env=network_env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            network_record = self.record(
                f"network:{case.name}", network_command, pid=process.pid,
                log=str(network_log.relative_to(EXP_ROOT)),
                uclamp=uclamp_snapshot(process.pid), returncode=None,
            )
            wait_ready(process, network_log, timeout=1200)

            benchmark_command = (
                str(UCLAMPSET), "-m", "1024", "--", str(self.python),
                "-m", "scripts.online.benchmark.benchmark_infer",
                "--measurement", "failure",
                "--orbit", f"127.0.0.1:{self.args.port}",
                "--host-id", f"shadow-{self.run_id}-{case.name}",
                "--advertise-host", "127.0.0.1", "--model", case.model,
                "--output", str(case_root / "benchmark"),
                "--prompt", PROMPT, "--max-tokens", str(self.args.max_tokens),
                "--warmup-tokens", "32", "--seqs", "1",
                "--drop-probabilities", format(self.args.drop_probability, ".8g"),
                "--repeats", "1", "--inspection-postpone", "0.3",
                "--num-paths", "1", "--logprobs", "0", "--disable-gc",
                "--async-backend", "golang", "--busy-wait",
            )
            benchmark_env = network_env.copy()
            benchmark_env["DROFELL_METRICS_PATH"] = str(metrics_path.resolve())
            with benchmark_log.open("wb") as log:
                benchmark = subprocess.Popen(
                    benchmark_command, cwd=REPO_ROOT, env=benchmark_env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            benchmark_record = self.record(
                f"benchmark:{case.name}", benchmark_command, pid=benchmark.pid,
                log=str(benchmark_log.relative_to(EXP_ROOT)),
                uclamp=uclamp_snapshot(benchmark.pid), returncode=None,
            )
            try:
                returncode = benchmark.wait(timeout=3600)
            except subprocess.TimeoutExpired as error:
                stop_group(benchmark)
                raise ExperimentError("benchmark timeout") from error
            benchmark_record["returncode"] = returncode
            self.flush_environment()
            if returncode:
                raise ExperimentError(
                    f"benchmark failed ({returncode}):\n"
                    f"{read_text(benchmark_log)[-16000:]}"
                )
            rows = parse_snapshots(case, metrics_path)
            if not rows:
                raise ExperimentError("no shadow_state_snapshot event was recorded")
            injected_failures = read_text(network_log).count(
                "trigger compute-node drop"
            )
            if injected_failures <= 0:
                raise ExperimentError("snapshots exist but no injected failure was logged")
            self.rows.extend(rows)
            case_record.update({
                "status": "completed", "snapshot_count": len(rows),
                "injected_failures": injected_failures,
                "metrics": str(metrics_path.relative_to(EXP_ROOT)),
            })
        except Exception as error:
            case_record.update({
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            })
        finally:
            stop_group(process)
            if process is not None:
                for item in reversed(self.commands):
                    if item["purpose"] == f"network:{case.name}":
                        item["returncode"] = process.returncode
                        break
            self.flush_environment()

    def finish(self) -> None:
        for case in self.cases:
            self.run_case(case)
        if self.rows:
            with (self.result_root / "snapshots.csv").open(
                "w", encoding="utf-8", newline=""
            ) as file:
                writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
        aggregates = []
        for case in self.cases:
            items = [row for row in self.rows if row["case"] == case.name]
            if not items:
                continue
            aggregates.append({
                "case": case.name, "model": case.model,
                "model_layers": case.num_layers,
                "shard_layers": case.shard_layers,
                "snapshot_count": len(items),
                **{
                    name: distribution([int(item[name]) for item in items])
                    for name in METRIC_NAMES
                },
            })
        write_json(self.result_root / "summary.json", {
            "schema_version": 1,
            "run_id": self.run_id,
            "protocol": (
                "Dedicated instrumentation run. One fresh network and one "
                "request per case; p=0.01; JSONL metrics enabled; throughput "
                "is not reported because metrics I/O perturbs the hot path."
            ),
            "case_records": self.case_records,
            "snapshot_count": len(self.rows),
            "aggregates": aggregates,
        })
        self.finished_utc = datetime.now(timezone.utc).isoformat()
        self.flush_environment()
        print(json.dumps({
            "run_id": self.run_id,
            "completed_cases": sum(
                item["status"] == "completed" for item in self.case_records
            ),
            "failed_cases": sum(
                item["status"] == "failed" for item in self.case_records
            ),
            "snapshot_count": len(self.rows),
            "summary": str(self.result_root / "summary.json"),
        }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--port", type=int, default=7290)
    parser.add_argument("--drop-probability", type=float, default=0.01)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(case.name for case in CASES),
        default=tuple(case.name for case in CASES),
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.drop_probability <= 0 or args.drop_probability > 0.05:
        parser.error("--drop-probability must be in (0, 0.05]")
    if args.max_tokens < 64 or args.max_tokens > 1028:
        parser.error("--max-tokens must be in [64, 1028]")
    runner = Runner(args)
    runner.initialize()
    runner.finish()


if __name__ == "__main__":
    main()
