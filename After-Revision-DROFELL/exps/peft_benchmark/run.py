#!/usr/bin/env python3
"""Run the Llama2-7B PEFT throughput-table reproduction."""

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
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/home/dai/projects/LLM/dflash_related/.venv/bin/python")
TORCH_EXTENSIONS_DIR = Path("/home/dai/.cache/torch_extensions/py312_cu130")
USER_LOCAL_BIN = Path("/home/dai/.local/bin")
MODEL = "teared-Llama-2-7B-GPTQ"
PORT = 7272
ORBIT_ADDR = f"127.0.0.1:{PORT}"
BACKENDS = ("exllama_fuse", "marlin")


class ExperimentError(RuntimeError):
    pass


@dataclass(frozen=True)
class Case:
    case_id: str
    clients: int
    adapter: str
    trainable: int
    effective_batch: int
    minibatch_size: int = 4

    @property
    def grad_accum_steps(self) -> int:
        if self.effective_batch % self.minibatch_size:
            raise ValueError("effective batch must be divisible by minibatch size")
        return self.effective_batch // self.minibatch_size

    @property
    def paper_key(self) -> str:
        prefix = "p" if self.adapter == "ptuning" else "r"
        return f"{prefix}{self.trainable}-b{self.effective_batch}-c{self.clients}"

    @property
    def lora_len_limit(self) -> int:
        if self.adapter != "lora":
            raise ValueError("LoRA length limit requested for a non-LoRA case")
        # The historical table labels r_q + r_v as Rank. For Llama2-7B,
        # len_limit=4*(per-projection rank)=2*(the table's aggregate rank).
        return self.trainable * 2

    @property
    def resolved_lora_projection_rank(self) -> int | None:
        return self.trainable // 2 if self.adapter == "lora" else None


EXLLAMA_CASES = tuple(
    Case(
        f"{adapter}-{'p' if adapter == 'ptuning' else 'r'}{trainable}"
        f"-b{effective_batch}-c{clients}",
        clients,
        adapter,
        trainable,
        effective_batch,
    )
    for clients in (1, 2, 4, 8)
    for adapter in ("ptuning", "lora")
    for trainable in (4, 16)
    for effective_batch in (8, 32)
)
MARLIN_CASES = (
    Case("ptuning-p4-b8-c4", 4, "ptuning", 4, 8),
    Case("lora-r4-b8-c4", 4, "lora", 4, 8),
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


class Runner:
    def __init__(
        self,
        python: Path,
        run_id: str,
        samples_per_client: int,
        backends: tuple[str, ...],
        selected_cases: tuple[str, ...] | None,
        resume: bool,
    ) -> None:
        if not python.exists():
            raise ExperimentError(f"Python does not exist: {python}")
        if samples_per_client < 64 or samples_per_client % 32:
            raise ExperimentError(
                "samples per client must be a multiple of 32 and at least 64"
            )
        self.python = python
        self.run_id = run_id
        self.samples_per_client = samples_per_client
        self.backends = backends
        self.selected_cases = selected_cases
        self.resume = resume
        self.result_root = EXP_ROOT / "results" / run_id
        self.log_root = EXP_ROOT / "logs" / run_id
        self.environment_path = EXP_ROOT / "environment" / f"{run_id}.json"
        self.network_dir = EXP_ROOT / "networks"
        self.commands: list[dict[str, Any]] = []
        self.network_process: subprocess.Popen[bytes] | None = None
        self.base_env = os.environ.copy()
        self.base_env.update({
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "PATH": f"{USER_LOCAL_BIN}:{self.base_env.get('PATH', '')}",
            "TORCH_EXTENSIONS_DIR": str(TORCH_EXTENSIONS_DIR),
            "REUSE_RECONSTRUCT_CACHE_WHEN_GRAD": "1",
            "DECODERS_TRAIN_METHOD": "each_decoder",
            "DECODERS_TRAIN_DOWNGRAD_METHOD": "first.2pass",
        })

    def network_path(self, backend: str) -> Path:
        return self.network_dir / f"llama2-7b-peft-{backend}.json"

    def record(self, command: Sequence[str], **extra: Any) -> None:
        self.commands.append({"command": command_text(command), **extra})

    def run_command(
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

    def uclamped_python(self, *args: str) -> tuple[str, ...]:
        return (
            "/usr/bin/uclampset", "-m", "1024", "--",
            str(self.python), *args,
        )

    def generate_networks(self) -> None:
        self.network_dir.mkdir(parents=True, exist_ok=True)
        for backend in self.backends:
            command = (
                str(self.python), "-m",
                "scripts.online.benchmark.generate_network",
                "--name", f"llama2-7b-peft-{backend}",
                "--model", MODEL,
                "--decoder-group-sizes", "8", "8", "8", "8",
                "--dealer", "dealer-a=0",
                "--dealer", "dealer-b=1",
                "--dealer", "dealer-c=2",
                "--dealer", "dealer-d=3",
                "--decoder-backend", backend,
                "--max-seq-len", "256",
                "--max-concurrency", "8",
                "--memory-policy", "static",
                "--major-reserved-mib", "8192",
                "--unmanaged-reserved-mib", "32",
                "--minor-min-mib", "0",
                "--minor-max-mib", "4096",
                "--operate-reserved-mib", "1024",
                "--output", str(self.network_path(backend)),
            )
            self.run_command(command)
            snapshot = self.result_root / "network_specs" / f"{backend}.json"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.network_path(backend), snapshot)

    def start_network(self, backend: str) -> None:
        if self.network_process is not None:
            raise ExperimentError("a network process is already running")
        log_path = self.unique_log_path(self.log_root / backend / "network.log")
        command = self.uclamped_python(
            "-m", "scripts.online.benchmark.launch_network", "local",
            "--network", str(self.network_path(backend)),
            "--port", str(PORT),
            "--advertise-host", "127.0.0.1",
            "--async-backend", "golang", "--busy-wait",
            "--place", "dealer-a", "0", "0", "cuda:0",
            "--place", "dealer-b", "0", "0", "cuda:0",
            "--place", "dealer-c", "1", "0", "cuda:1",
            "--place", "dealer-d", "1", "0", "cuda:1",
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("wb")
        process = subprocess.Popen(
            list(command), cwd=REPO_ROOT, env=self.base_env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        log.close()
        self.network_process = process
        self.record(command, pid=process.pid, log=str(log_path))
        self.wait_for_text(
            process,
            log_path,
            f"network 'llama2-7b-peft-{backend}' ready",
            timeout=900,
        )

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
                    f"network exited before {needle!r}:\n{content[-16000:]}"
                )
            content = log_path.read_text(encoding="utf-8", errors="replace")
            if needle in content:
                return
            time.sleep(0.2)
        raise ExperimentError(f"timed out waiting for {needle!r} in {log_path}")

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
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=15)

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

    def cases_for_backend(self, backend: str) -> tuple[Case, ...]:
        available = EXLLAMA_CASES if backend == "exllama_fuse" else MARLIN_CASES
        if self.selected_cases is None:
            return available
        return tuple(
            case for case in available if case.case_id in self.selected_cases
        )

    def run_case(self, backend: str, case: Case) -> None:
        output = self.result_root / backend / case.case_id
        if output.exists():
            if self.resume and (output / "results.json").exists():
                return
            if not self.resume:
                raise ExperimentError(
                    f"refusing to overwrite existing case: {output}"
                )
            failed_root = (
                self.result_root / "failed_attempts" / backend / case.case_id
            )
            attempt = 1
            failed = failed_root / f"attempt-{attempt}"
            while failed.exists():
                attempt += 1
                failed = failed_root / f"attempt-{attempt}"
            failed.parent.mkdir(parents=True, exist_ok=True)
            output.rename(failed)
        case_payload = {
            **asdict(case),
            "backend": backend,
            "samples_per_client": self.samples_per_client,
            "grad_accum_steps": case.grad_accum_steps,
            "paper_key": case.paper_key,
            "lora_len_limit": (
                case.lora_len_limit if case.adapter == "lora" else None
            ),
            "resolved_lora_projection_rank": (
                case.resolved_lora_projection_rank
            ),
            "deployment": "combined-request-agent",
            "repeat": 0,
        }
        write_json(output / "case.json", case_payload)
        args = [
            "-m", "scripts.online.benchmark.benchmark_train",
            "--orbit", ORBIT_ADDR,
            "--host-id", f"peft-{backend}-{case.case_id}",
            "--advertise-host", "127.0.0.1",
            "--model", MODEL,
            "--output", str(output),
            "--instances", str(case.clients),
            "--samples-per-instance", str(self.samples_per_client),
            "--minibatch-size", str(case.minibatch_size),
            "--grad-accum-steps", str(case.grad_accum_steps),
            "--sample-len", "128",
            "--adapter", case.adapter,
            "--ckpt-period", "16",
            "--detail-trace",
            "--optimizer", "AdamW",
            "--optimizer-eps", "1e-4",
            "--async-backend", "golang", "--busy-wait",
        ]
        if case.adapter == "ptuning":
            args += [
                "--ptuning-prefix-len", str(case.trainable),
                "--adapter-dropout", "0.1",
                "--learning-rate", "1e-2",
            ]
        else:
            args += [
                "--lora-r-q", "-1", "--lora-r-k", "0",
                "--lora-r-v", "-1",
                "--lora-len-limit", str(case.lora_len_limit),
                "--adapter-dropout", "0.0",
                "--learning-rate", "1e-3",
            ]
        self.run_command(
            self.uclamped_python(*args),
            log_path=self.unique_log_path(
                self.log_root / backend / f"{case.case_id}.log"
            ),
            timeout=1800,
        )

    def run_matrix(self) -> None:
        for backend in self.backends:
            cases = self.cases_for_backend(backend)
            if self.resume:
                cases = tuple(
                    case for case in cases
                    if not (
                        self.result_root / backend / case.case_id / "results.json"
                    ).exists()
                )
            if not cases:
                continue
            self.start_network(backend)
            try:
                for case in cases:
                    self.run_case(backend, case)
            finally:
                self.stop_network()

    def output(self, *command: str) -> str:
        result = subprocess.run(
            command, cwd=REPO_ROOT, env=self.base_env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False,
        )
        return result.stdout.strip()

    def capture_environment(self) -> None:
        governors = []
        for path in sorted(Path("/sys/devices/system/cpu").glob(
            "cpu[0-9]*/cpufreq/scaling_governor"
        )):
            try:
                governors.append(path.read_text(encoding="utf-8").strip())
            except OSError:
                pass
        previous_commands: list[dict[str, Any]] = []
        previous_top_level_commands: list[str] = []
        if self.environment_path.exists():
            try:
                previous = json.loads(
                    self.environment_path.read_text(encoding="utf-8")
                )
                previous_commands = list(previous.get("commands", []))
                previous_top_level_commands = list(
                    previous.get("top_level_commands", [])
                )
                old_top = previous.get("top_level_command")
                if old_top and old_top not in previous_top_level_commands:
                    previous_top_level_commands.append(old_top)
            except (json.JSONDecodeError, OSError, TypeError):
                pass
        current_top = command_text((str(self.python), *sys.argv))
        write_json(self.environment_path, {
            "run_id": self.run_id,
            "top_level_command": current_top,
            "top_level_commands": previous_top_level_commands + [current_top],
            "git_commit": self.output("git", "rev-parse", "HEAD"),
            "git_status": self.output("git", "status", "--short"),
            "python": self.output(str(self.python), "--version"),
            "torch": self.output(
                str(self.python), "-c",
                "import torch; print(torch.__version__, torch.version.cuda)",
            ),
            "gpus": self.output(
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ),
            "gpu_compute_processes": self.output(
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
                "--format=csv,noheader",
            ),
            "cpu": self.output("lscpu"),
            "cpu_governors": sorted(set(governors)),
            "uclampset": self.output("uclampset", "--version"),
            "uclamp_min": 1024,
            "cuda_visible_devices": self.base_env["CUDA_VISIBLE_DEVICES"],
            "torch_extensions_dir": str(TORCH_EXTENSIONS_DIR),
            "ninja": self.output("ninja", "--version"),
            "matrix": {
                "samples_per_client": self.samples_per_client,
                "repeats_per_case": 1,
                "backends": list(BACKENDS),
                "exllama_cases": [asdict(case) for case in EXLLAMA_CASES],
                "marlin_cases": [asdict(case) for case in MARLIN_CASES],
                "request_agent_deployment": "combined",
                "fp16": True,
                "reuse_reconstruct_cache_when_grad": True,
                "decoders_train_method": "each_decoder",
            },
            "current_invocation_selection": {
                "backends": list(self.backends),
                "selected_cases": (
                    list(self.selected_cases)
                    if self.selected_cases is not None else None
                ),
                "resume": self.resume,
            },
            "commands": previous_commands + self.commands,
        })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Llama2-7B PEFT throughput-table reproduction"
    )
    parser.add_argument(
        "phase", nargs="?", default="all",
        choices=("prepare", "run", "analyze", "all"),
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--samples-per-client", type=int, default=256)
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=BACKENDS,
    )
    parser.add_argument(
        "--cases", nargs="+", help="Run only the selected case IDs",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip completed cases and archive incomplete case directories",
    )
    args = parser.parse_args()
    backends = tuple(dict.fromkeys(args.backends))
    selected_cases = (
        tuple(dict.fromkeys(args.cases)) if args.cases is not None else None
    )
    known_cases = {
        case.case_id for case in (*EXLLAMA_CASES, *MARLIN_CASES)
    }
    if selected_cases is not None:
        unknown = sorted(set(selected_cases) - known_cases)
        if unknown:
            parser.error(f"unknown case IDs: {unknown}")
    runner = Runner(
        args.python, args.run_id, args.samples_per_client, backends,
        selected_cases, args.resume,
    )
    try:
        if args.phase in {"prepare", "run", "all"}:
            runner.generate_networks()
        if args.phase in {"run", "all"}:
            runner.run_matrix()
        if args.phase in {"analyze", "all"}:
            if not runner.result_root.exists():
                raise ExperimentError(
                    f"result root does not exist: {runner.result_root}"
                )
            runner.run_command((
                str(args.python), str(EXP_ROOT / "analyze.py"),
                str(runner.result_root),
            ))
    finally:
        runner.stop_network()
        runner.capture_environment()
    print(f"run-id: {runner.run_id}")
    print(f"results: {runner.result_root}")
    print(f"environment: {runner.environment_path}")


if __name__ == "__main__":
    main()
