#!/usr/bin/env python3
"""Run the PEFT out-of-order dispatch / async-poll 2x2 ablation."""

from __future__ import annotations

import argparse
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
DEFAULT_PYTHON = Path("/home/dai/projects/LLM/dflash_related/.venv/bin/python")
TORCH_EXTENSIONS_DIR = Path("/home/dai/.cache/torch_extensions/py312_cu130")
USER_LOCAL_BIN = Path("/home/dai/.local/bin")
MODEL = "teared-Llama-2-7B-GPTQ"
PORT = 7273
ORBIT_ADDR = f"127.0.0.1:{PORT}"
PATCH_DIR = EXP_ROOT / "patches"
DELAY_PATCH = PATCH_DIR / "emission-delay-200ms.patch"
OOO_PATCH = PATCH_DIR / "disable-ooo.patch"
ASYNC_PATCH = PATCH_DIR / "disable-async-poll.patch"
PATCHES = (DELAY_PATCH, OOO_PATCH, ASYNC_PATCH)


class ExperimentError(RuntimeError):
    pass


@dataclass(frozen=True)
class Case:
    case_id: str
    out_of_order_dispatch: bool
    async_poller: bool

    @property
    def async_backend(self) -> str:
        return "golang" if self.async_poller else "py"

    @property
    def busy_wait(self) -> bool:
        return self.async_poller


CASES = (
    Case("opt", True, True),
    Case("no-ooo", False, True),
    Case("no-async-poll", True, False),
    Case("non-opt", False, False),
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


class Runner:
    def __init__(
        self,
        python: Path,
        run_id: str,
        samples_per_client: int,
        selected_cases: tuple[str, ...],
        resume: bool,
    ) -> None:
        if not python.exists():
            raise ExperimentError(f"Python does not exist: {python}")
        if samples_per_client < 64 or samples_per_client % 32:
            raise ExperimentError(
                "samples per client must be a multiple of 32 and at least 64"
            )
        missing_patches = [path for path in PATCHES if not path.exists()]
        if missing_patches:
            raise ExperimentError(f"missing patches: {missing_patches}")
        # Do not resolve the venv launcher symlink: CPython uses its original
        # path to discover pyvenv.cfg and site-packages.
        self.python = python.absolute()
        self.run_id = run_id
        self.samples_per_client = samples_per_client
        self.selected_cases = selected_cases
        self.resume = resume
        self.result_root = EXP_ROOT / "results" / run_id
        self.log_root = EXP_ROOT / "logs" / run_id
        self.environment_path = EXP_ROOT / "environment" / f"{run_id}.json"
        self.network_path = EXP_ROOT / "networks" / "llama2-7b-oor-async.json"
        self.commands: list[dict[str, Any]] = []
        self.network_process: subprocess.Popen[bytes] | None = None
        self.worktrees: dict[str, Path] = {}
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
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

    def record(self, command: Sequence[str], **extra: Any) -> None:
        self.commands.append({"command": command_text(command), **extra})

    def run_command(
        self,
        command: Sequence[str],
        *,
        cwd: Path = REPO_ROOT,
        log_path: Path | None = None,
        timeout: float | None = None,
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        actual_env = env if env is not None else self.base_env
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
            command,
            cwd=str(cwd),
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

    def generate_network(self) -> None:
        self.network_path.parent.mkdir(parents=True, exist_ok=True)
        command = (
            str(self.python), "-m",
            "scripts.online.benchmark.generate_network",
            "--name", "llama2-7b-peft-oor-async",
            "--model", MODEL,
            "--decoder-group-sizes", "8", "8", "8", "8",
            "--dealer", "dealer-1=0,1",
            "--dealer", "dealer-2=2,3",
            "--decoder-backend", "exllama_fuse",
            "--max-seq-len", "256",
            "--max-concurrency", "8",
            "--time-trace",
            "--memory-policy", "static",
            "--major-reserved-mib", "8192",
            "--unmanaged-reserved-mib", "32",
            "--minor-min-mib", "0",
            "--minor-max-mib", "4096",
            "--operate-reserved-mib", "1024",
            "--output", str(self.network_path),
        )
        env = self.source_env(REPO_ROOT)
        self.run_command(command, env=env)
        snapshot = self.result_root / "network_specs" / self.network_path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.network_path, snapshot)

    def ensure_worktree(self, name: str, patches: tuple[Path, ...]) -> Path:
        if name in self.worktrees:
            return self.worktrees[name]
        if self._tempdir is None:
            self._tempdir = tempfile.TemporaryDirectory(
                prefix="drofell-peft-oor-async-"
            )
        worktree_dir = Path(self._tempdir.name) / name
        self.run_command((
            "git", "worktree", "add", "--detach",
            str(worktree_dir), "HEAD",
        ))
        self.worktrees[name] = worktree_dir
        for patch in patches:
            self.run_command((
                "git", "apply", "--check", str(patch),
            ), cwd=worktree_dir)
            self.run_command((
                "git", "apply", str(patch),
            ), cwd=worktree_dir)
        linked_runtime_paths = []
        for runtime_name in ("cache", "data", "split_qllm_models"):
            source = REPO_ROOT / runtime_name
            if not source.exists():
                continue
            target = worktree_dir / runtime_name
            if target.exists() or target.is_symlink():
                if runtime_name == "split_qllm_models" and target.is_dir():
                    # HEAD contains only the package stub. The main workspace
                    # uses a local model-library symlink, so expose each model
                    # directory to the detached worktree as well.
                    for child in source.resolve().iterdir():
                        target_child = target / child.name
                        if not target_child.exists():
                            target_child.symlink_to(
                                child, target_is_directory=child.is_dir()
                            )
                    linked_runtime_paths.append(
                        "split_qllm_models/*"
                    )
                continue
            target.symlink_to(source, target_is_directory=source.is_dir())
            linked_runtime_paths.append(runtime_name)
        # Detached worktrees do not contain ignored, locally built extension
        # modules. Link the ABI-matched artifacts from the main workspace so
        # the variant executes the same binaries without copying or rebuilding.
        extension_count = 0
        for source in REPO_ROOT.rglob("*.so"):
            relative = source.relative_to(REPO_ROOT)
            target = worktree_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
            extension_count += 1
        if extension_count == 0:
            raise ExperimentError("no local extension modules found to link")
        self.commands.append({
            "action": "link_local_extension_modules",
            "source_root": str(REPO_ROOT),
            "worktree": str(worktree_dir),
            "count": extension_count,
            "runtime_paths": linked_runtime_paths,
            "patches": [str(path) for path in patches],
        })
        status = self.run_command(
            ("git", "diff", "--check"), cwd=worktree_dir
        )
        if status.returncode != 0:
            raise ExperimentError(f"{name} worktree has an invalid diff")
        return worktree_dir

    def cleanup_worktree(self) -> None:
        for worktree_dir in self.worktrees.values():
            subprocess.run(
                ("git", "worktree", "remove", "--force", str(worktree_dir)),
                cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=False,
            )
        self.worktrees.clear()
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def source_env(self, source_root: Path) -> dict[str, str]:
        env = self.base_env.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{source_root}:{existing}" if existing else str(source_root)
        )
        return env

    def start_network(self, async_poller: bool) -> None:
        if self.network_process is not None:
            raise ExperimentError("a network process is already running")
        runtime = "async-on" if async_poller else "async-off"
        backend = "golang" if async_poller else "py"
        busy_flag = "--busy-wait" if async_poller else "--no-busy-wait"
        source_root = (
            REPO_ROOT if async_poller else self.ensure_worktree(
                "network-async-off", (ASYNC_PATCH,)
            )
        )
        log_path = self.unique_log_path(self.log_root / runtime / "network.log")
        command = self.uclamped_python(
            "-m", "scripts.online.benchmark.launch_network", "local",
            "--network", str(self.network_path),
            "--port", str(PORT),
            "--advertise-host", "127.0.0.1",
            "--async-backend", backend, busy_flag,
            "--place", "dealer-1", "0", "0", "cuda:0",
            "--place", "dealer-2", "1", "0", "cuda:1",
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("wb")
        process = subprocess.Popen(
            list(command), cwd=source_root, env=self.source_env(source_root),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        log.close()
        self.network_process = process
        self.record(
            command, cwd=str(source_root), pid=process.pid, log=str(log_path)
        )
        self.wait_for_text(
            process, log_path,
            "network 'llama2-7b-peft-oor-async' ready", timeout=900,
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
            if "Traceback (most recent call last)" in content:
                raise ExperimentError(
                    f"network failed during startup:\n{content[-16000:]}"
                )
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
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=20)

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

    def output_dir(self, case: Case) -> Path:
        return self.result_root / case.case_id

    def run_case(self, case: Case) -> None:
        output = self.output_dir(case)
        if output.exists():
            if self.resume and (output / "results.json").exists():
                return
            if not self.resume:
                raise ExperimentError(f"refusing to overwrite case: {output}")
            failed_root = self.result_root / "failed_attempts" / case.case_id
            attempt = 1
            failed = failed_root / f"attempt-{attempt}"
            while failed.exists():
                attempt += 1
                failed = failed_root / f"attempt-{attempt}"
            failed.parent.mkdir(parents=True, exist_ok=True)
            output.rename(failed)

        case_patches = [DELAY_PATCH]
        if not case.out_of_order_dispatch:
            case_patches.append(OOO_PATCH)
        if not case.async_poller:
            case_patches.append(ASYNC_PATCH)
        source_root = self.ensure_worktree(
            f"benchmark-{case.case_id}", tuple(case_patches)
        )
        case_payload = {
            **asdict(case),
            "async_backend": case.async_backend,
            "busy_wait": case.busy_wait,
            "backend": "exllama_fuse",
            "clients": 2,
            "samples_per_client": self.samples_per_client,
            "adapter": "ptuning",
            "ptuning_prefix_len": 16,
            "minibatch_size": 4,
            "grad_accum_steps": 2,
            "effective_batch": 8,
            "sample_len": 128,
            "deployment": "combined-request-agent",
            "source_root": str(source_root),
            "patches": [
                {"path": str(path), "sha256": sha256(path)}
                for path in case_patches
            ],
            "request_emission_delay_seconds": 0.2,
            "repeat": 0,
        }
        write_json(output / "case.json", case_payload)

        busy_flag = "--busy-wait" if case.busy_wait else "--no-busy-wait"
        args = (
            "-m", "scripts.online.benchmark.benchmark_train",
            "--orbit", ORBIT_ADDR,
            "--host-id", f"peft-oor-async-{case.case_id}",
            "--advertise-host", "127.0.0.1",
            "--model", MODEL,
            "--output", str(output),
            "--instances", "2",
            "--samples-per-instance", str(self.samples_per_client),
            "--minibatch-size", "4",
            "--grad-accum-steps", "2",
            "--sample-len", "128",
            "--adapter", "ptuning",
            "--ptuning-prefix-len", "16",
            "--adapter-dropout", "0.1",
            "--learning-rate", "1e-2",
            "--optimizer", "AdamW",
            "--optimizer-eps", "1e-4",
            "--ckpt-period", "16",
            "--detail-trace",
            "--async-backend", case.async_backend,
            busy_flag,
        )
        self.run_command(
            self.uclamped_python(*args),
            cwd=source_root,
            env=self.source_env(source_root),
            log_path=self.unique_log_path(
                self.log_root / case.case_id / "benchmark.log"
            ),
            timeout=1800,
        )

    def selected(self) -> tuple[Case, ...]:
        return tuple(c for c in CASES if c.case_id in self.selected_cases)

    def run_matrix(self) -> None:
        cases = self.selected()
        for async_poller in (True, False):
            group = tuple(
                case for case in cases
                if case.async_poller == async_poller
                and not (
                    self.resume and
                    (self.output_dir(case) / "results.json").exists()
                )
            )
            if not group:
                continue
            self.start_network(async_poller)
            try:
                for case in group:
                    self.run_case(case)
            finally:
                self.stop_network()

    def output(self, *command: str) -> str:
        completed = subprocess.run(
            command, cwd=REPO_ROOT, env=self.base_env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        return completed.stdout.strip()

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
                old = json.loads(self.environment_path.read_text(encoding="utf-8"))
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
                "--query-gpu=index,name,memory.total,driver_version",
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
            "cuda_visible_devices": self.base_env["CUDA_VISIBLE_DEVICES"],
            "torch_extensions_dir": str(TORCH_EXTENSIONS_DIR),
            "patches": [
                {"path": str(path), "sha256": sha256(path)}
                for path in PATCHES
            ],
            "parameters": {
                "cases": [asdict(case) for case in CASES],
                "selected_cases": list(self.selected_cases),
                "samples_per_client": self.samples_per_client,
                "clients": 2,
                "ptuning_prefix_len": 16,
                "effective_batch": 8,
                "sample_len": 128,
                "request_emission_delay_seconds": 0.2,
                "backend": "exllama_fuse",
                "cuda_graph": False,
                "dealer_grouping": {"dealer-1": [0, 1], "dealer-2": [2, 3]},
            },
            "commands": old_commands + self.commands,
        })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the PEFT OOO/async-poll 2x2 ablation"
    )
    parser.add_argument(
        "phase", nargs="?", default="all",
        choices=("prepare", "run", "analyze", "all"),
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--samples-per-client", type=int, default=256)
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(c.case_id for c in CASES),
        default=tuple(c.case_id for c in CASES),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    runner = Runner(
        args.python, args.run_id, args.samples_per_client,
        tuple(dict.fromkeys(args.cases)), args.resume,
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
        runner.cleanup_worktree()
        runner.capture_environment()
    print(f"run-id: {runner.run_id}")
    print(f"results: {runner.result_root}")
    print(f"environment: {runner.environment_path}")


if __name__ == "__main__":
    main()
