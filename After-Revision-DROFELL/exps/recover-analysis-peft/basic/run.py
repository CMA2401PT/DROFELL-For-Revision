#!/usr/bin/env python3
"""Run the basic PEFT nonblocking Shadow/Promise recovery experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
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
NETWORK_NAME = "recover-analysis-peft-basic-ordinary"
DEALERS = (
    ("T0-train-a", "0,1,2,3", 0, "cuda:0"),
    ("T0-train-b", "0,1,2,3", 1, "cuda:1"),
)


@dataclass(frozen=True)
class Condition:
    name: str
    failure: bool
    link_latency_ms: float


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


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


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


def wait_for_network(
    process: subprocess.Popen[bytes], log_path: Path, timeout: float
) -> None:
    needle = f"network {NETWORK_NAME!r} ready"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ExperimentError(
                f"network exited before ready (code={process.returncode}):\n"
                f"{read_text(log_path)[-16000:]}"
            )
        if needle in read_text(log_path):
            return
        time.sleep(0.5)
    raise ExperimentError(
        f"timed out waiting for {needle!r}:\n{read_text(log_path)[-16000:]}"
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
            pass
    raise ExperimentError(f"failed to stop process group {process.pid}")


def latency_label(value: float) -> str:
    return f"recover-latency-{f'{value:g}'.replace('.', 'p')}ms"


class Experiment:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.python: Path = args.python
        self.run_id: str = args.run_id
        self.result_root = (
            args.result_root.resolve()
            if args.result_root is not None else RESULTS_ROOT / self.run_id
        )
        self.log_root = (
            args.log_root.resolve()
            if args.log_root is not None else EXP_ROOT / "logs" / self.run_id
        )
        self.environment_path = (
            self.result_root / "environment.json"
            if args.result_root is not None
            else EXP_ROOT / "environment" / f"{self.run_id}.json"
        )
        self.network_path = EXP_ROOT / "networks" / "ordinary-8x4.json"
        self.commands: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []
        conditions: list[Condition] = []
        if "normal" in args.cases:
            conditions.append(Condition("normal", False, 0.0))
        if "recover" in args.cases:
            conditions.extend(
                Condition(latency_label(value), True, value)
                for value in args.recovery_latencies_ms
            )
        self.conditions = tuple(conditions)

    @property
    def failure_plan(self) -> dict[str, list[int]]:
        versions = list(self.args.drop_versions)
        return {name: versions for name, _, _, _ in DEALERS}

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "experiment_profile": self.args.experiment_profile,
            "scope": {"backbone": False, "redundant_active_paths": 0,
                      "split_or_degradation": False},
            "model": MODEL,
            "adapter": "P-Tuning v2" if self.args.adapter == "ptuning" else "LoRA",
            "ptuning_prefix_len": self.args.prefix_len,
            "lora_r_q": -1,
            "lora_r_k": 0,
            "lora_r_v": -1,
            "lora_len_limit": self.args.lora_len_limit,
            "adapter_dropout": 0.0,
            "optimizer": "AdamW",
            "learning_rate": self.args.learning_rate,
            "optimizer_eps": self.args.optimizer_eps,
            "attention_backend": (
                "pytorch_math" if self.args.pytorch_math_attention else "flash"
            ),
            "state_fingerprints": self.args.state_fingerprints,
            "samples_per_instance": self.args.samples,
            "minibatch_size": self.args.minibatch_size,
            "grad_accum_steps": self.args.grad_accum_steps,
            "submission_interval_ms": self.args.submission_interval_ms,
            "planned_submission_window_s": (
                math.ceil(self.args.samples / self.args.minibatch_size)
                * self.args.submission_interval_ms / 1000.0
            ),
            "dataset": "AdvertiseGen",
            "dataset_seed": self.args.seed,
            "dataset_query_tokens": self.args.dataset_query_tokens,
            "dataset_response_tokens": self.args.dataset_response_tokens,
            "max_seq_len": self.args.max_seq_len,
            "ckpt_period": self.args.ckpt_period,
            "drop_versions": list(self.args.drop_versions),
            "drop_times_s": list(self.args.drop_times_s),
            "recovery_symmetric_per_direction_latencies_ms": list(
                self.args.recovery_latencies_ms
            ),
            "recovery_link_bandwidth_mbps": self.args.bandwidth_mbps,
            "force_measured_train_path_probe": True,
            "failure_model": (
                "timed_train_compute_node_teardown"
                if self.args.drop_times_s
                else "deterministic_train_compute_node_teardown"
            ),
            "recovery": "checkpoint + pending minibatch replay",
            "random_seed": self.args.seed,
            "repeats": self.args.repeats,
        }

    def flush_manifest(self) -> None:
        write_json(self.result_root / "manifest.json", {
            "schema_version": 2, "run_id": self.run_id,
            "parameters": self.parameters,
            "network": str(self.network_path.relative_to(SUITE_ROOT)),
            "commands": self.commands, "runs": self.runs,
        })

    def initialize(self) -> None:
        if not self.python.is_file():
            raise ExperimentError(f"Python interpreter not found: {self.python}")
        if self.result_root.exists() or self.log_root.exists():
            raise ExperimentError(f"refusing to overwrite run {self.run_id!r}")
        if not port_available(self.args.port):
            raise ExperimentError(f"port {self.args.port} is in use")
        self.result_root.mkdir(parents=True)
        self.log_root.mkdir(parents=True)
        probes = {
            "git_commit": ("git", "rev-parse", "HEAD"),
            "git_status": ("git", "status", "--short", "--untracked-files=all"),
            "nvidia_smi": (
                "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader",
            ),
            "python_torch": (
                str(self.python), "-c",
                "import platform,torch; print(platform.python_version()); "
                "print(torch.__version__); print(torch.version.cuda)",
            ),
        }
        environment = {
            "run_id": self.run_id,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(REPO_ROOT), "python": str(self.python),
            "parameters": self.parameters,
            "probes": {name: capture(command) for name, command in probes.items()},
        }
        write_json(self.environment_path, environment)
        write_json(self.result_root / "environment.json", environment)
        self.flush_manifest()

    def record(self, purpose: str, command: Sequence[str],
               log: Path | None = None, returncode: int | None = None) -> dict[str, Any]:
        if log is None:
            log_text = None
        else:
            try:
                log_text = str(log.relative_to(SUITE_ROOT))
            except ValueError:
                log_text = str(log.relative_to(REPO_ROOT))
        item = {
            "purpose": purpose, "command": command_text(command),
            "log": log_text,
            "returncode": returncode,
        }
        self.commands.append(item)
        self.flush_manifest()
        return item

    def generate_network(self) -> None:
        self.network_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.python), "-m", "scripts.online.benchmark.generate_network",
            "--name", NETWORK_NAME, "--model", MODEL,
            "--decoder-group-sizes", "8", "8", "8", "8",
        ]
        for name, groups, _, _ in DEALERS:
            command.extend(("--dealer", f"{name}={groups}"))
        command.extend((
            "--decoder-backend", "exllama_fuse",
            "--max-seq-len", str(self.args.max_seq_len),
            "--max-concurrency", "32", "--failure-injection",
            "--memory-policy", "static", "--major-reserved-mib", "8192",
            "--major-percentage", "0.8", "--unmanaged-reserved-mib", "32",
            "--minor-min-mib", "0", "--minor-max-mib", "4096",
            "--operate-reserved-mib", "1024", "--output", str(self.network_path),
        ))
        result = capture(command)
        self.record("generate-network", command, returncode=result["returncode"])
        if result["returncode"] != 0:
            raise ExperimentError(result["output"])

    def network_command(self) -> list[str]:
        command = [
            "uclampset", "-m", "1024", "--", str(self.python),
            str(EXP_ROOT / "launch_network.py"), "local",
            "--network", str(self.network_path), "--port", str(self.args.port),
            "--advertise-host", "127.0.0.1", "--async-backend", "golang",
            "--busy-wait",
        ]
        for name, _, process, device in DEALERS:
            command.extend(("--place", name, str(process), "0", device))
        return command

    def benchmark_command(self, raw_dir: Path, host_id: str) -> list[str]:
        return [
            "uclampset", "-m", "1024", "--", str(self.python), "-m",
            "scripts.online.benchmark.benchmark_train", "--orbit",
            f"127.0.0.1:{self.args.port}", "--host-id", host_id,
            "--advertise-host", "127.0.0.1", "--model", MODEL,
            "--output", str(raw_dir), "--instances", "1",
            "--samples-per-instance", str(self.args.samples),
            "--minibatch-size", str(self.args.minibatch_size),
            "--grad-accum-steps", str(self.args.grad_accum_steps),
            "--submission-interval-ms", str(self.args.submission_interval_ms),
            "--dataset", "advertise-gen", "--dataset-seed", str(self.args.seed),
            "--dataset-query-tokens", str(self.args.dataset_query_tokens),
            "--dataset-response-tokens", str(self.args.dataset_response_tokens),
            "--adapter", self.args.adapter,
            "--ptuning-prefix-len", str(self.args.prefix_len),
            "--lora-r-q", "-1", "--lora-r-k", "0", "--lora-r-v", "-1",
            "--lora-len-limit", str(self.args.lora_len_limit),
            "--adapter-dropout", "0", "--ckpt-period", str(self.args.ckpt_period),
            "--optimizer", "AdamW", "--optimizer-eps", str(self.args.optimizer_eps),
            "--learning-rate", str(self.args.learning_rate), "--detail-trace",
            "--async-backend", "golang", "--busy-wait",
        ]

    def run_condition(self, condition: Condition, repeat: int) -> None:
        label = f"{condition.name}-repeat-{repeat}"
        relative = Path(condition.name) / f"repeat-{repeat}"
        run_dir = self.result_root / relative
        raw_dir = run_dir / "raw"
        metrics_path = run_dir / "events.jsonl"
        claim_dir = run_dir / "failure_claims"
        train_start_time_file = run_dir / "train_start_time_ns.txt"
        run_dir.mkdir(parents=True)
        network_log = self.log_root / f"{label}-network.log"
        benchmark_log = self.log_root / f"{label}-benchmark.log"
        failure_plan = self.failure_plan if condition.failure else {}
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(REPO_ROOT), "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "0", "PEFT_RECOVER_SEED": str(self.args.seed),
            "CUDA_VISIBLE_DEVICES": "0,1",
            "DROFELL_METRICS_PATH": str(metrics_path.resolve()),
            "PEFT_RECOVER_DROP_PLAN": json.dumps(failure_plan, separators=(",", ":")),
            "PEFT_RECOVER_DROP_TIMES_S": json.dumps(
                list(self.args.drop_times_s), separators=(",", ":")
            ) if condition.failure else "[]",
            "PEFT_RECOVER_TRAIN_START_TIME_FILE": str(
                train_start_time_file.resolve()
            ),
            "PEFT_RECOVER_CLAIM_DIR": str(claim_dir.resolve()),
            "PEFT_RECOVER_LINK_LATENCY_MS": str(condition.link_latency_ms),
            "PEFT_RECOVER_LINK_BANDWIDTH_MBPS": str(self.args.bandwidth_mbps),
            "DROFELL_FORCE_TRAIN_PATH_PROBE": "1",
            "REUSE_RECONSTRUCT_CACHE_WHEN_GRAD": "1",
            "DECODERS_TRAIN_METHOD": "each_decoder",
            "DECODERS_TRAIN_DOWNGRAD_METHOD": "first.2pass",
            "USE_PYTORCH_MATH_ATTENTION": (
                "1" if self.args.pytorch_math_attention else "0"
            ),
            "PEFT_RECOVER_STATE_FINGERPRINTS": (
                "1" if self.args.state_fingerprints else "0"
            ),
        })
        network_command = self.network_command()
        with network_log.open("wb") as output:
            network = subprocess.Popen(
                network_command, cwd=REPO_ROOT, env=env,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            )
        network_record = self.record(f"network:{label}", network_command, network_log)
        benchmark_returncode: int | None = None
        error: str | None = None
        try:
            wait_for_network(network, network_log, 900)
            train_start_time_file.write_text(
                f"{time.time_ns()}\n", encoding="utf-8"
            )
            host_id = f"peft-recover-{self.run_id}-{condition.name}-r{repeat}"
            benchmark_command = self.benchmark_command(raw_dir, host_id)
            with benchmark_log.open("wb") as output:
                completed = subprocess.run(
                    benchmark_command, cwd=REPO_ROOT, env=env,
                    stdout=output, stderr=subprocess.STDOUT,
                    timeout=3600, check=False,
                )
            benchmark_returncode = completed.returncode
            self.record(f"benchmark:{label}", benchmark_command,
                        benchmark_log, benchmark_returncode)
            if benchmark_returncode != 0:
                raise ExperimentError(
                    f"benchmark failed ({benchmark_returncode}):\n"
                    f"{read_text(benchmark_log)[-16000:]}"
                )
            required = (raw_dir / "results.json", raw_dir / "args.json",
                        raw_dir / "network.json", metrics_path)
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise ExperimentError(f"missing outputs: {missing}")
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            stop_process_group(network)
            network_record["returncode"] = network.returncode
            network_record["stopped_by_runner"] = True
            self.runs.append({
                "condition": condition.name,
                "case": "recover" if condition.failure else "normal",
                "repeat": repeat, "failure": condition.failure,
                "link_latency_ms": condition.link_latency_ms,
                "failure_plan": failure_plan, "run_dir": str(relative),
                "drop_times_s": list(self.args.drop_times_s),
                "network_log": str(network_log),
                "benchmark_log": str(benchmark_log),
                "benchmark_returncode": benchmark_returncode, "error": error,
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
        analysis_command = [str(self.python), str(EXP_ROOT / "analyze.py"),
                            str(self.result_root)]
        analysis_log = self.log_root / "analysis.log"
        with analysis_log.open("wb") as output:
            completed = subprocess.run(
                analysis_command, cwd=REPO_ROOT,
                stdout=output, stderr=subprocess.STDOUT, check=False,
            )
        self.record("analyze", analysis_command, analysis_log, completed.returncode)
        if completed.returncode != 0:
            raise ExperimentError(read_text(analysis_log)[-16000:])
        environment = json.loads(self.environment_path.read_text(encoding="utf-8"))
        environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(self.environment_path, environment)
        write_json(self.result_root / "environment.json", environment)


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_run_id: str | None = None,
    default_port: int = 7266,
    default_repeats: int = 3,
    default_drop_versions: tuple[int, ...] = (2, 5, 8),
    default_drop_times_s: tuple[float, ...] = (),
    default_latencies_ms: tuple[float, ...] = (50.0,),
    default_samples: int = 96,
    experiment_profile: str = "causality",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-profile", default=experiment_profile)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument(
        "--run-id", default=default_run_id or f"peft-real-{utc_run_id()}"
    )
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--repeats", type=int, default=default_repeats)
    parser.add_argument(
        "--drop-versions", nargs="*", type=int,
        default=default_drop_versions,
    )
    parser.add_argument(
        "--drop-times-s", nargs="+", type=float,
        default=default_drop_times_s,
        help=(
            "Fail the currently active TrainNode at these elapsed seconds "
            "from benchmark launch; mutually exclusive with --drop-versions."
        ),
    )
    parser.add_argument("--recovery-latencies-ms", nargs="+", type=float,
                        default=default_latencies_ms)
    parser.add_argument("--bandwidth-mbps", type=float, default=1024.0)
    parser.add_argument("--submission-interval-ms", type=float, default=250.0)
    parser.add_argument("--samples", type=int, default=default_samples)
    parser.add_argument("--minibatch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--dataset-query-tokens", type=int, default=32)
    parser.add_argument("--dataset-response-tokens", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--ckpt-period", type=int, default=2)
    parser.add_argument("--prefix-len", type=int, default=32)
    parser.add_argument("--lora-len-limit", type=int, default=128)
    parser.add_argument("--adapter", choices=("ptuning", "lora"), default="ptuning")
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--optimizer-eps", type=float, default=1e-4)
    parser.add_argument(
        "--pytorch-math-attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="replace FlashAttention with PyTorch SDPA's math backend",
    )
    parser.add_argument(
        "--state-fingerprints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="record exact adapter/optimizer checkpoint fingerprints",
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--case", action="append", choices=("normal", "recover"),
                        dest="cases")
    args = parser.parse_args(argv)
    args.drop_versions = tuple(args.drop_versions)
    args.drop_times_s = tuple(args.drop_times_s)
    args.recovery_latencies_ms = tuple(args.recovery_latencies_ms)
    args.cases = args.cases or ["normal", "recover"]
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.learning_rate <= 0 or args.optimizer_eps <= 0:
        parser.error("optimizer learning rate and epsilon must be positive")
    if (tuple(sorted(args.drop_versions)) != args.drop_versions
            or len(set(args.drop_versions)) != len(args.drop_versions)):
        parser.error("--drop-versions must be unique and increasing")
    if (tuple(sorted(args.drop_times_s)) != args.drop_times_s
            or len(set(args.drop_times_s)) != len(args.drop_times_s)
            or any(value <= 0 for value in args.drop_times_s)):
        parser.error("--drop-times-s must be positive, unique, and increasing")
    if args.drop_versions and args.drop_times_s:
        parser.error("--drop-versions and --drop-times-s are mutually exclusive")
    if not args.drop_versions and not args.drop_times_s:
        parser.error("a version or elapsed-time failure plan is required")
    total_minibatches = (args.samples + args.minibatch_size - 1) // args.minibatch_size
    max_version = (total_minibatches - 1) // args.grad_accum_steps
    if args.drop_versions and (
        args.drop_versions[0] < 1 or args.drop_versions[-1] >= max_version - 1
    ):
        parser.error("drop versions must start after 0 and leave two versions after the last failure")
    submission_duration_s = (
        total_minibatches * args.submission_interval_ms / 1000.0
    )
    if args.drop_times_s and args.drop_times_s[-1] >= submission_duration_s:
        parser.error(
            "last timed failure must occur before the open-loop submission "
            f"window ends ({submission_duration_s:g}s)"
        )
    if any(value < 0 for value in args.recovery_latencies_ms):
        parser.error("recovery latencies must be non-negative")
    if len(set(args.recovery_latencies_ms)) != len(args.recovery_latencies_ms):
        parser.error("recovery latencies must be unique")
    if args.bandwidth_mbps <= 0 or args.submission_interval_ms <= 0:
        parser.error("bandwidth and submission interval must be positive")
    if min(args.ckpt_period, args.minibatch_size, args.grad_accum_steps,
           args.max_seq_len) < 1:
        parser.error("batch and checkpoint parameters must be positive")
    if any(value < 0 or (value > 0 and value % 8 != 0) for value in (
        args.dataset_query_tokens, args.dataset_response_tokens
    )):
        parser.error("dataset force-token counts must be 0 or positive multiples of 8")
    if len(set(args.cases)) != len(args.cases):
        parser.error("--case values must be unique")
    return args


if __name__ == "__main__":
    experiment = Experiment(parse_args())
    experiment.execute()
    print(f"completed: {experiment.result_root}")
