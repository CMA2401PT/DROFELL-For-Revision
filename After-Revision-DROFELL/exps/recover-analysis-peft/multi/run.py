#!/usr/bin/env python3
"""Run clean or delayed Backbone-internal PEFT multi-path experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time
from typing import Any, Sequence


EXP_ROOT = Path(__file__).resolve().parent
SUITE_ROOT = EXP_ROOT.parent
RESULTS_ROOT = SUITE_ROOT / "results"
REPO_ROOT = EXP_ROOT.parents[2]
DEFAULT_PYTHON = Path("/home/dai/projects/LLM/dflash_related/.venv/bin/python")
MODEL = "teared-Llama-2-7B-GPTQ"
NETWORK_NAME = "recover-analysis-peft-multi-backbone"


@dataclass(frozen=True)
class DealerPlacement:
    name: str
    groups: str
    process: int
    device: str


DEALERS = (
    DealerPlacement("T0-peft-fast-active", "0,1,2,3", 0, "cuda:0"),
    DealerPlacement("T1-peft-delayed-survivor", "0,1,2,3", 1, "cuda:1"),
    DealerPlacement("T2-peft-fast-recovery", "0,1,2,3", 2, "cuda:0"),
    DealerPlacement("T3-peft-delayed-recovery", "0,1,2,3", 3, "cuda:1"),
)
INNER_DEALERS = tuple(dealer.name for dealer in DEALERS)
INITIAL_DEALERS = ("T0-peft-fast-active", "T1-peft-delayed-survivor")
DELAYED_DEALERS = (
    "T1-peft-delayed-survivor",
    "T3-peft-delayed-recovery",
)
REPLACEMENT_MAP = {
    "T0-peft-fast-active": "T2-peft-fast-recovery",
    "T2-peft-fast-recovery": "T0-peft-fast-active",
    "T1-peft-delayed-survivor": "T3-peft-delayed-recovery",
    "T3-peft-delayed-recovery": "T1-peft-delayed-survivor",
}
BACKBONE = "T0-peft-backbone"


@dataclass(frozen=True)
class Condition:
    name: str
    failure: bool


class ExperimentError(RuntimeError):
    pass


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def command_text(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def capture(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command), cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
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


def wait_for_network(
    process: subprocess.Popen[bytes], log_path: Path, timeout: float,
) -> None:
    needle = f"network '{NETWORK_NAME}' ready"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ExperimentError(
                f"network exited before ready ({process.returncode}):\n"
                f"{read_text(log_path)[-20000:]}"
            )
        if needle in read_text(log_path):
            return
        time.sleep(0.5)
    raise ExperimentError(
        f"timed out waiting for {needle!r}:\n{read_text(log_path)[-20000:]}"
    )


def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    for sig, timeout in (
        (signal.SIGINT, 20.0),
        (signal.SIGTERM, 10.0),
        (signal.SIGKILL, 5.0),
    ):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
    raise ExperimentError(f"failed to stop process group {process.pid}")


class Experiment:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.python: Path = args.python
        self.run_id: str = args.run_id
        self.result_root = RESULTS_ROOT / self.run_id
        self.log_root = EXP_ROOT / "logs" / self.run_id
        self.environment_path = EXP_ROOT / "environment" / f"{self.run_id}.json"
        self.network_path = EXP_ROOT / "networks" / "backbone-multi-8x4.json"
        self.conditions = tuple(
            Condition(name, name == "recover") for name in args.cases
        )
        self.commands: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []

    def failure_plan(self, failure: bool) -> list[float]:
        return list(self.args.failure_times_s) if failure else []

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "experiment_profile": self.args.profile,
            "scope": {
                "backbone": True,
                "outer_active_paths": 1,
                "inner_active_paths_per_backbone": 2,
                "failure_count": len(self.args.failure_times_s),
                "split_or_degradation": False,
            },
            "model": MODEL,
            "adapter": "P-Tuning v2",
            "ptuning_prefix_len": self.args.prefix_len,
            "adapter_dropout": 0.0,
            "optimizer": "AdamW",
            "learning_rate": self.args.learning_rate,
            "optimizer_eps": 1e-4,
            "attention_backend": (
                "pytorch_math" if self.args.pytorch_math_attention else "flash"
            ),
            "samples_per_instance": self.args.samples,
            "minibatch_size": 1,
            "grad_accum_steps": self.args.grad_accum_steps,
            "submission_interval_ms": self.args.submission_interval_ms,
            "planned_submission_window_s": (
                self.args.samples * self.args.submission_interval_ms / 1000.0
            ),
            "dataset": "AdvertiseGen",
            "dataset_seed": self.args.seed,
            "dataset_query_tokens": self.args.dataset_query_tokens,
            "dataset_response_tokens": self.args.dataset_response_tokens,
            "ckpt_period": self.args.ckpt_period,
            "failure_times_s": list(self.args.failure_times_s),
            "failure_model": "agent_observed_active_winner_teardown",
            "delayed_path_per_direction_latency_ms": self.args.path_latency_ms,
            "delayed_dealers": list(DELAYED_DEALERS),
            "force_measured_train_path_probe": True,
            "placement": {
                "fast_active_and_replacement": "cuda:0",
                "delayed_survivor_and_replacement": "cuda:1",
            },
            "repeats": self.args.repeats,
            "random_seed": self.args.seed,
        }

    def flush_manifest(self) -> None:
        write_json(self.result_root / "manifest.json", {
            "schema_version": 1,
            "run_id": self.run_id,
            "parameters": self.parameters,
            "network": str(self.network_path.relative_to(SUITE_ROOT)),
            "commands": self.commands,
            "runs": self.runs,
        })

    def record(
        self, purpose: str, command: Sequence[str],
        log_path: Path | None = None, returncode: int | None = None,
    ) -> dict[str, Any]:
        item = {
            "purpose": purpose,
            "command": command_text(command),
            "log": None if log_path is None else str(log_path.relative_to(SUITE_ROOT)),
            "returncode": returncode,
        }
        self.commands.append(item)
        self.flush_manifest()
        return item

    def initialize(self) -> None:
        if not self.python.is_file():
            raise ExperimentError(f"Python interpreter not found: {self.python}")
        if (
            self.result_root.exists()
            or self.log_root.exists()
            or self.environment_path.exists()
        ):
            raise ExperimentError(f"refusing to overwrite run {self.run_id!r}")
        if not port_available(self.args.port):
            raise ExperimentError(f"port {self.args.port} is in use")
        self.result_root.mkdir(parents=True)
        self.log_root.mkdir(parents=True)
        environment = {
            "schema_version": 1,
            "run_id": self.run_id,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(REPO_ROOT),
            "python": str(self.python),
            "parameters": self.parameters,
            "probes": {
                "git_commit": capture(("git", "rev-parse", "HEAD")),
                "git_status": capture((
                    "git", "status", "--short", "--untracked-files=all"
                )),
                "nvidia_smi": capture((
                    "nvidia-smi", "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader",
                )),
            },
        }
        write_json(self.environment_path, environment)
        self.flush_manifest()

    def generate_network(self) -> None:
        self.network_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.python), "-m", "scripts.online.benchmark.generate_network",
            "--name", NETWORK_NAME,
            "--model", MODEL,
            "--decoder-group-sizes", "8", "8", "8", "8",
        ]
        for dealer in DEALERS:
            command.extend(("--dealer", f"{dealer.name}={dealer.groups}"))
        command.extend((
            "--decoder-backend", "exllama_fuse",
            "--max-seq-len", "256",
            "--max-concurrency", "32",
            "--failure-injection",
            "--memory-policy", "static",
            "--major-reserved-mib", "8192",
            "--major-percentage", "0.8",
            "--unmanaged-reserved-mib", "32",
            "--minor-min-mib", "0",
            "--minor-max-mib", "4096",
            "--operate-reserved-mib", "1024",
            "--output", str(self.network_path),
        ))
        result = capture(command)
        self.record("generate-network", command, returncode=result["returncode"])
        if result["returncode"] != 0:
            raise ExperimentError(result["output"])

    def network_command(self) -> list[str]:
        command = [
            "uclampset", "-m", "1024", "--",
            str(self.python), str(EXP_ROOT / "launch_network.py"),
            "--network", str(self.network_path),
            "--port", str(self.args.port),
            "--advertise-host", "127.0.0.1",
            "--async-backend", "golang", "--busy-wait",
            "--backbone-name", BACKBONE,
            "--backbone-process", "4",
            "--backbone-search-redundancy", "3",
            "--backbone-path-redundancy", "1",
        ]
        for dealer in DEALERS:
            command.extend((
                "--place", dealer.name, str(dealer.process), "0", dealer.device,
            ))
        return command

    def benchmark_command(self, raw_dir: Path, host_id: str) -> list[str]:
        return [
            "uclampset", "-m", "1024", "--",
            str(self.python), str(EXP_ROOT / "benchmark_train.py"),
            "--orbit", f"127.0.0.1:{self.args.port}",
            "--host-id", host_id,
            "--advertise-host", "127.0.0.1",
            "--model", MODEL,
            "--output", str(raw_dir),
            "--instances", "1",
            "--samples-per-instance", str(self.args.samples),
            "--minibatch-size", "1",
            "--grad-accum-steps", str(self.args.grad_accum_steps),
            "--submission-interval-ms", str(self.args.submission_interval_ms),
            "--dataset", "advertise-gen",
            "--dataset-seed", str(self.args.seed),
            "--dataset-query-tokens", str(self.args.dataset_query_tokens),
            "--dataset-response-tokens", str(self.args.dataset_response_tokens),
            "--adapter", "ptuning",
            "--ptuning-prefix-len", str(self.args.prefix_len),
            "--adapter-dropout", "0",
            "--ckpt-period", str(self.args.ckpt_period),
            "--optimizer", "AdamW",
            "--optimizer-eps", "1e-4",
            "--learning-rate", str(self.args.learning_rate),
            "--num-paths", "1",
            "--detail-trace",
            "--async-backend", "golang", "--busy-wait",
        ]

    def run_condition(self, condition: Condition, repeat: int) -> None:
        label = f"{condition.name}-repeat-{repeat}"
        relative = Path(condition.name) / f"repeat-{repeat}"
        run_dir = self.result_root / relative
        raw_dir = run_dir / "raw"
        metrics_path = run_dir / "events.jsonl"
        start_time_file = run_dir / "train_start_time_ns.txt"
        run_dir.mkdir(parents=True)
        network_log = self.log_root / f"{label}-network.log"
        benchmark_log = self.log_root / f"{label}-benchmark.log"
        host_id = f"peft-multi-{self.run_id}-{condition.name}-r{repeat}"
        failure_plan = self.failure_plan(condition.failure)
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "DROFELL_METRICS_PATH": str(metrics_path.resolve()),
            "DROFELL_FORCE_TRAIN_PATH_PROBE": "1",
            "PEFT_MULTI_SEED": str(self.args.seed),
            "PEFT_MULTI_AGENT_NAME": host_id,
            "PEFT_MULTI_INNER_DEALERS": json.dumps(INNER_DEALERS),
            "PEFT_MULTI_BACKBONE_NAME": BACKBONE,
            "PEFT_MULTI_DELAYED_DEALERS": json.dumps(DELAYED_DEALERS),
            "PEFT_MULTI_PATH_LATENCY_MS": str(self.args.path_latency_ms),
            "RECOVER_MULTI_INITIAL_BACKBONE_DEALERS": json.dumps(INITIAL_DEALERS),
            "PEFT_MULTI_REPLACEMENT_MAP": json.dumps(
                REPLACEMENT_MAP, separators=(",", ":")
            ),
            "PEFT_MULTI_ACTIVE_WINNER_FAILURE_TIMES_S": json.dumps(
                failure_plan, separators=(",", ":")
            ),
            "PEFT_MULTI_FAILURE_RELAY": BACKBONE,
            "PEFT_MULTI_TRAIN_START_TIME_FILE": str(start_time_file.resolve()),
            "REUSE_RECONSTRUCT_CACHE_WHEN_GRAD": "1",
            "DECODERS_TRAIN_METHOD": "each_decoder",
            "DECODERS_TRAIN_DOWNGRAD_METHOD": "first.2pass",
            "USE_PYTORCH_MATH_ATTENTION": (
                "1" if self.args.pytorch_math_attention else "0"
            ),
        })

        network_command = self.network_command()
        with network_log.open("wb") as output:
            network = subprocess.Popen(
                network_command, cwd=REPO_ROOT, env=env,
                stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        network_record = self.record(
            f"network:{label}", network_command, network_log
        )
        benchmark_returncode: int | None = None
        error: str | None = None
        try:
            wait_for_network(network, network_log, 1200)
            start_time_file.write_text(f"{time.time_ns()}\n", encoding="utf-8")
            benchmark_command = self.benchmark_command(raw_dir, host_id)
            with benchmark_log.open("wb") as output:
                completed = subprocess.run(
                    benchmark_command, cwd=REPO_ROOT, env=env,
                    stdout=output, stderr=subprocess.STDOUT,
                    timeout=3600, check=False,
                )
            benchmark_returncode = completed.returncode
            self.record(
                f"benchmark:{label}", benchmark_command,
                benchmark_log, benchmark_returncode,
            )
            if benchmark_returncode != 0:
                raise ExperimentError(
                    f"benchmark failed ({benchmark_returncode}):\n"
                    f"{read_text(benchmark_log)[-20000:]}"
                )
            required = (
                raw_dir / "results.json",
                raw_dir / "args.json",
                raw_dir / "network.json",
                metrics_path,
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise ExperimentError(f"missing outputs: {missing}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stop_process_group(network)
            network_record["returncode"] = network.returncode
            network_record["stopped_by_runner"] = True
            self.runs.append({
                "condition": condition.name,
                "case": condition.name,
                "repeat": repeat,
                "failure": condition.failure,
                "run_dir": str(relative),
                "failure_plan": failure_plan,
                "agent_name": host_id,
                "network_log": str(network_log.relative_to(SUITE_ROOT)),
                "benchmark_log": str(benchmark_log.relative_to(SUITE_ROOT)),
                "benchmark_returncode": benchmark_returncode,
                "error": error,
            })
            self.flush_manifest()
            deadline = time.monotonic() + 30
            while not port_available(self.args.port) and time.monotonic() < deadline:
                time.sleep(0.25)

    def execute(self) -> None:
        self.initialize()
        self.generate_network()
        for repeat in range(self.args.repeats):
            for condition in self.conditions:
                self.run_condition(condition, repeat)
        analysis_command = [
            str(self.python), str(EXP_ROOT / "analyze.py"), str(self.result_root)
        ]
        analysis_log = self.log_root / "analysis.log"
        with analysis_log.open("wb") as output:
            completed = subprocess.run(
                analysis_command, cwd=REPO_ROOT,
                stdout=output, stderr=subprocess.STDOUT, check=False,
            )
        self.record(
            "analyze", analysis_command, analysis_log, completed.returncode
        )
        if completed.returncode != 0:
            raise ExperimentError(read_text(analysis_log)[-20000:])
        environment = json.loads(
            self.environment_path.read_text(encoding="utf-8")
        )
        environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(self.environment_path, environment)
        write_json(self.result_root / "environment.json", environment)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--profile", choices=("clean", "delayed"), default="clean")
    parser.add_argument("--run-id")
    parser.add_argument("--port", type=int, default=7331)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--samples", type=int, default=960)
    parser.add_argument("--submission-interval-ms", type=float, default=250.0)
    parser.add_argument("--failure-times-s", nargs="+", type=float)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--ckpt-period", type=int, default=10)
    parser.add_argument("--prefix-len", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument(
        "--pytorch-math-attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="replace FlashAttention with PyTorch SDPA's math backend",
    )
    parser.add_argument("--dataset-query-tokens", type=int, default=32)
    parser.add_argument("--dataset-response-tokens", type=int, default=32)
    parser.add_argument("--path-latency-ms", type=float)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--case", action="append", choices=("normal", "recover"),
        dest="cases",
    )
    args = parser.parse_args(argv)
    if args.failure_times_s is None:
        args.failure_times_s = (
            [120.0] if args.profile == "clean" else [60.0, 120.0, 180.0]
        )
    args.failure_times_s = tuple(args.failure_times_s)
    if args.path_latency_ms is None:
        args.path_latency_ms = 0.0 if args.profile == "clean" else 50.0
    if args.run_id is None:
        failures = len(args.failure_times_s)
        args.run_id = f"multipath-peft-{args.profile}-f{failures}-r{args.repeats}-{utc_run_id()}"
    args.cases = args.cases or ["normal", "recover"]
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.repeats < 1 or args.samples < 16:
        parser.error("repeats must be positive and samples must be at least 16")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if min(args.dataset_query_tokens, args.dataset_response_tokens) < 1:
        parser.error("dataset token lengths must be positive")
    if args.prefix_len + args.dataset_query_tokens + args.dataset_response_tokens > 256:
        parser.error("prefix/query/response token lengths must fit --max-seq-len 256")
    if min(
        args.submission_interval_ms, args.path_latency_ms,
        *args.failure_times_s,
    ) < 0:
        parser.error("timing values must be non-negative")
    if (
        not args.failure_times_s
        or any(value <= 0 for value in args.failure_times_s)
        or tuple(sorted(args.failure_times_s)) != args.failure_times_s
        or len(set(args.failure_times_s)) != len(args.failure_times_s)
    ):
        parser.error("failure times must be unique increasing positive values")
    if args.profile == "clean" and (
        len(args.failure_times_s) != 1 or args.path_latency_ms != 0
    ):
        parser.error("clean profile requires one failure and 0 ms path latency")
    if args.profile == "delayed" and (
        len(args.failure_times_s) != 3 or args.path_latency_ms <= 0
    ):
        parser.error("delayed profile requires three failures and positive latency")
    window_s = args.samples * args.submission_interval_ms / 1000.0
    if args.failure_times_s[-1] >= window_s - 5:
        parser.error(
            f"final failure must leave at least 5 seconds in the {window_s:g}s window"
        )
    if args.grad_accum_steps < 1 or args.ckpt_period < 1:
        parser.error("gradient accumulation and checkpoint period must be positive")
    if len(set(args.cases)) != len(args.cases):
        parser.error("--case values must be unique")
    return args


def main() -> None:
    experiment = Experiment(parse_args())
    experiment.execute()
    print(f"completed: {experiment.result_root}")


if __name__ == "__main__":
    main()
