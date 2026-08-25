#!/usr/bin/env python3
"""Run the delayed/Mix ordinary and Backbone multi-path profile."""

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
PROMPT = "once upon a time"
NETWORK_NAMES = {
    "ordinary": "recover-analysis-multi-ordinary",
    "backbone": "recover-analysis-multi-backbone",
}


@dataclass(frozen=True)
class DealerPlacement:
    name: str
    groups: str
    process: int
    device: str


DEALERS = {
    "ordinary": (
        DealerPlacement("T0-ordinary-fast-a", "0,1,2,3", 0, "cuda:0"),
        DealerPlacement("T0-ordinary-survivor", "0,1,2,3", 1, "cuda:1"),
        DealerPlacement("T1-ordinary-fast-b", "0,1,2,3", 2, "cuda:0"),
    ),
    "backbone": (
        DealerPlacement("T0-backbone-direct", "0,1,2,3", 0, "cuda:0"),
        DealerPlacement("T1-backbone-inner-a", "0,1,2,3", 1, "cuda:1"),
        DealerPlacement("T2-backbone-inner-b", "0,1,2,3", 2, "cuda:1"),
    ),
}


@dataclass(frozen=True)
class Case:
    topology: str
    failure: bool
    retrieve: bool

    @property
    def method(self) -> str:
        return "mix" if self.retrieve else "reprefill"

    @property
    def name(self) -> str:
        mode = "recover" if self.failure else "normal"
        return f"{self.topology}-{mode}-{self.method}"


CASES = tuple(
    Case(topology, failure, retrieve)
    for topology in ("ordinary", "backbone")
    for retrieve in (False, True)
    for failure in (False, True)
)


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
        list(command),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command_text(command),
        "returncode": completed.returncode,
        "output": completed.stdout,
    }


def wait_for_network(
    process: subprocess.Popen[bytes],
    log_path: Path,
    network_name: str,
    timeout: float,
) -> None:
    needle = f"network '{network_name}' ready"
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
            continue
    raise ExperimentError(f"failed to stop network process group {process.pid}")


class Experiment:
    def __init__(
        self,
        *,
        python: Path,
        run_id: str,
        port: int,
        repeats: int,
        drop_steps: tuple[int, ...],
        max_tokens: int,
        selected_cases: tuple[str, ...],
        standby_latency_ms: float,
    ) -> None:
        self.python = python
        self.run_id = run_id
        self.port = port
        self.repeats = repeats
        self.drop_steps = drop_steps
        self.max_tokens = max_tokens
        self.selected_cases = selected_cases
        self.standby_latency_ms = standby_latency_ms
        self.result_root = RESULTS_ROOT / run_id
        self.log_root = EXP_ROOT / "logs" / run_id
        self.environment_path = EXP_ROOT / "environment" / f"{run_id}.json"
        self.network_paths = {
            topology: EXP_ROOT / "networks" / f"{topology}-multi-8x4.json"
            for topology in NETWORK_NAMES
        }
        self.network_names = NETWORK_NAMES
        self.dealers = DEALERS
        self.cases = CASES
        self.commands: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []

    def failure_plan(self, topology: str) -> dict[str, list[int]]:
        even = list(self.drop_steps[0::2])
        odd = list(self.drop_steps[1::2])
        if topology == "ordinary":
            return {
                "T0-ordinary-fast-a": even,
                "T0-ordinary-survivor": [],
                "T1-ordinary-fast-b": odd,
            }
        if topology == "backbone":
            return {
                "T0-backbone-direct": [],
                "T1-backbone-inner-a": even,
                "T2-backbone-inner-b": odd,
            }
        raise ValueError(topology)

    def standby_dealers(self, topology: str) -> list[str]:
        if topology == "ordinary":
            return ["T0-ordinary-survivor"]
        return ["T0-backbone-direct"]

    def inner_dealers(self, topology: str) -> list[str]:
        if topology != "backbone":
            return []
        return ["T1-backbone-inner-a", "T2-backbone-inner-b"]

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "experiment_profile": "delayed_mix_repeated_recovery",
            "model": MODEL,
            "prompt": PROMPT,
            "max_tokens": self.max_tokens,
            "warmup_tokens": 32,
            "parallel_sequences": 1,
            "num_paths": 2,
            "drop_steps": list(self.drop_steps),
            "repeats": self.repeats,
            "selected_cases": list(self.selected_cases),
            "recovery_replay": "chunk",
            "failure_model": "deterministic_active_branch_teardown",
            "standby_latency_ms": self.standby_latency_ms,
            "temperature": 0.0,
            "ignore_eos": True,
            "decoder_backend": "marlin",
            "cuda_graph": True,
            "topologies": {
                "ordinary": {
                    "paths": [
                        "Agent -> alternating fast Dealer on GPU 0",
                        "Agent -> surviving Dealer on GPU 1",
                    ],
                    "recovery_scope": "agent",
                    "failure_plan": self.failure_plan("ordinary"),
                },
                "backbone": {
                    "paths": [
                        "Agent -> direct Dealer on GPU 0",
                        "Agent -> Backbone -> alternating inner Dealer on GPU 1",
                    ],
                    "recovery_scope": "backbone",
                    "failure_plan": self.failure_plan("backbone"),
                },
            },
        }

    def initialize(self) -> None:
        if not self.python.is_file():
            raise ExperimentError(f"Python interpreter not found: {self.python}")
        if self.result_root.exists() or self.log_root.exists() or self.environment_path.exists():
            raise ExperimentError(f"refusing to overwrite run id {self.run_id!r}")
        if not port_available(self.port):
            raise ExperimentError(f"port {self.port} is already in use")
        self.result_root.mkdir(parents=True)
        self.log_root.mkdir(parents=True)
        self.environment_path.parent.mkdir(parents=True, exist_ok=True)
        environment = {
            "schema_version": 1,
            "run_id": self.run_id,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(REPO_ROOT),
            "python": str(self.python),
            "port": self.port,
            "parameters": self.parameters,
            "probes": {
                "git_commit": capture(("git", "rev-parse", "HEAD")),
                "git_status": capture(("git", "status", "--short", "--untracked-files=all")),
                "nvidia_smi": capture((
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version,memory.total",
                    "--format=csv,noheader",
                )),
            },
        }
        write_json(self.environment_path, environment)
        self.flush_manifest()

    def flush_manifest(self) -> None:
        write_json(self.result_root / "manifest.json", {
            "schema_version": 1,
            "run_id": self.run_id,
            "parameters": self.parameters,
            "networks": {
                topology: str(path.relative_to(SUITE_ROOT))
                for topology, path in self.network_paths.items()
            },
            "commands": self.commands,
            "runs": self.runs,
        })

    def record_command(
        self,
        purpose: str,
        command: Sequence[str],
        log_path: Path | None = None,
        returncode: int | None = None,
    ) -> dict[str, Any]:
        record = {
            "purpose": purpose,
            "command": command_text(command),
            "log": None if log_path is None else str(log_path.relative_to(SUITE_ROOT)),
            "returncode": returncode,
        }
        self.commands.append(record)
        self.flush_manifest()
        return record

    def generate_network(self, topology: str) -> None:
        output = self.network_paths[topology]
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.python), "-m", "scripts.online.benchmark.generate_network",
            "--name", self.network_names[topology],
            "--model", MODEL,
            "--decoder-group-sizes", "8", "8", "8", "8",
        ]
        for dealer in self.dealers[topology]:
            command.extend(("--dealer", f"{dealer.name}={dealer.groups}"))
        command.extend((
            "--decoder-backend", "marlin",
            "--max-seq-len", "512",
            "--max-concurrency", "1",
            "--cuda-graph",
            "--failure-injection",
            "--memory-policy", "static",
            "--major-reserved-mib", "8192",
            "--major-percentage", "0.8",
            "--unmanaged-reserved-mib", "32",
            "--minor-min-mib", "0",
            "--minor-max-mib", "4096",
            "--operate-reserved-mib", "1024",
            "--graph-budget-mib", "1024",
            "--output", str(output),
        ))
        result = capture(command)
        self.record_command(
            f"generate-network:{topology}",
            command,
            returncode=result["returncode"],
        )
        if result["returncode"] != 0:
            raise ExperimentError(result["output"])

    def network_command(self, topology: str) -> list[str]:
        command = [
            "uclampset", "-m", "1024", "--",
            str(self.python), str(EXP_ROOT / "launch_network.py"),
            "--topology", topology,
            "--network", str(self.network_paths[topology]),
            "--port", str(self.port),
            "--advertise-host", "127.0.0.1",
            "--async-backend", "golang",
            "--busy-wait",
        ]
        for dealer in self.dealers[topology]:
            command.extend((
                "--place", dealer.name, str(dealer.process), "0", dealer.device,
            ))
        return command

    def num_paths(self, topology: str) -> int:
        return 2

    def benchmark_command(
        self,
        raw_dir: Path,
        host_id: str,
        retrieve: bool,
        topology: str,
    ) -> list[str]:
        command = [
            "uclampset", "-m", "1024", "--",
            str(self.python), "-m", "scripts.online.benchmark.benchmark_infer",
            "--measurement", "request",
            "--orbit", f"127.0.0.1:{self.port}",
            "--host-id", host_id,
            "--advertise-host", "127.0.0.1",
            "--model", MODEL,
            "--output", str(raw_dir),
            "--prompt", PROMPT,
            "--max-tokens", str(self.max_tokens),
            "--warmup-tokens", "32",
            "--seqs", "1",
            "--num-paths", str(self.num_paths(topology)),
            "--logprobs", "0",
            "--disable-gc",
            "--async-backend", "golang",
            "--busy-wait",
        ]
        if retrieve:
            command.append("--retrieve")
        return command

    def extra_environment(self, case: Case, repeat: int) -> dict[str, str]:
        return {}

    def run_case(self, case: Case, repeat: int) -> None:
        relative_run_dir = Path(case.name) / f"repeat-{repeat}"
        run_dir = self.result_root / relative_run_dir
        raw_dir = run_dir / "raw"
        metrics_path = run_dir / "runtime_metrics.jsonl"
        run_dir.mkdir(parents=True)
        label = f"{case.name}-repeat-{repeat}"
        network_log = self.log_root / f"{label}-network.log"
        benchmark_log = self.log_root / f"{label}-benchmark.log"
        failure_plan = self.failure_plan(case.topology) if case.failure else {}
        host_id = f"recover-multi-{self.run_id}-{case.name}-r{repeat}"
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1",
            "DROFELL_METRICS_PATH": str(metrics_path.resolve()),
            "RECOVER_MULTI_DROP_PLAN": json.dumps(failure_plan, separators=(",", ":")),
            "RECOVER_MULTI_TOPOLOGY": case.topology,
            "RECOVER_MULTI_AGENT_NAME": host_id,
            "RECOVER_MULTI_INNER_DEALERS": json.dumps(self.inner_dealers(case.topology)),
            "RECOVER_MULTI_STANDBY_DEALERS": json.dumps(self.standby_dealers(case.topology)),
            "RECOVER_MULTI_STANDBY_LATENCY_MS": f"{self.standby_latency_ms:.6f}",
            "DROFELL_FORCE_INFER_PATH_PROBE": "1",
            "RECOVERY_REPLAY": "chunk",
        })
        env.update(self.extra_environment(case, repeat))
        if not port_available(self.port):
            raise ExperimentError(f"port {self.port} is not free before {label}")

        network_command = self.network_command(case.topology)
        with network_log.open("wb") as file:
            network_process = subprocess.Popen(
                network_command,
                cwd=REPO_ROOT,
                env=env,
                stdout=file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        network_record = self.record_command(
            f"network:{label}", network_command, network_log
        )
        benchmark_returncode: int | None = None
        error: str | None = None
        try:
            wait_for_network(
                network_process,
                network_log,
                self.network_names[case.topology],
                300.0,
            )
            benchmark_command = self.benchmark_command(
                raw_dir, host_id, case.retrieve, case.topology
            )
            with benchmark_log.open("wb") as file:
                benchmark = subprocess.run(
                    benchmark_command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=file,
                    stderr=subprocess.STDOUT,
                    timeout=1200.0,
                    check=False,
                )
            benchmark_returncode = benchmark.returncode
            self.record_command(
                f"benchmark:{label}",
                benchmark_command,
                benchmark_log,
                benchmark_returncode,
            )
            if benchmark_returncode != 0:
                raise ExperimentError(
                    f"benchmark {label} failed ({benchmark_returncode}):\n"
                    f"{read_text(benchmark_log)[-16000:]}"
                )
            required = (
                raw_dir / "result.json",
                raw_dir / "tokens.json",
                raw_dir / "phases.json",
                raw_dir / "decode_graph.json",
                metrics_path,
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise ExperimentError(f"run {label} missing outputs: {missing}")
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            stop_process_group(network_process)
            network_record["returncode"] = network_process.returncode
            network_record["stopped_by_runner"] = True
            self.runs.append({
                "case": case.name,
                "topology": case.topology,
                "method": case.method,
                "repeat": repeat,
                "failure": case.failure,
                "retrieve": case.retrieve,
                "failure_plan": failure_plan,
                "agent_name": host_id,
                "run_dir": str(relative_run_dir),
                "network_log": str(network_log.relative_to(SUITE_ROOT)),
                "benchmark_log": str(benchmark_log.relative_to(SUITE_ROOT)),
                "benchmark_returncode": benchmark_returncode,
                "error": error,
            })
            self.flush_manifest()
            deadline = time.monotonic() + 30.0
            while not port_available(self.port) and time.monotonic() < deadline:
                time.sleep(0.25)
            time.sleep(0.5)

    def analyze(self) -> None:
        command = [
            str(self.python), str(EXP_ROOT / "analyze.py"), str(self.result_root)
        ]
        log_path = self.log_root / "analysis.log"
        with log_path.open("wb") as file:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        self.record_command("analyze", command, log_path, completed.returncode)
        if completed.returncode != 0:
            raise ExperimentError(
                f"analysis failed:\n{read_text(log_path)[-16000:]}"
            )

    def execute(self) -> None:
        self.initialize()
        topologies = {
            case.topology for case in self.cases if case.name in self.selected_cases
        }
        for topology in sorted(topologies):
            self.generate_network(topology)
        by_name = {case.name: case for case in self.cases}
        for repeat in range(self.repeats):
            for case_name in self.selected_cases:
                self.run_case(by_name[case_name], repeat)
        self.analyze()
        environment = json.loads(self.environment_path.read_text(encoding="utf-8"))
        environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(self.environment_path, environment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run-id", default=f"multipath-delayed-{utc_run_id()}")
    parser.add_argument("--port", type=int, default=7311)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--drop-steps", nargs="+", type=int, default=(40, 100, 160, 220))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--standby-latency-ms", type=float, default=3.0)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.name for case in CASES),
        dest="cases",
    )
    args = parser.parse_args()
    args.drop_steps = tuple(args.drop_steps)
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if (
        any(step < 1 for step in args.drop_steps)
        or tuple(sorted(args.drop_steps)) != args.drop_steps
        or len(set(args.drop_steps)) != len(args.drop_steps)
    ):
        parser.error("--drop-steps must be unique increasing positive integers")
    if args.max_tokens <= args.drop_steps[-1] + 8:
        parser.error("--max-tokens must leave at least 8 outputs after the final drop")
    if args.standby_latency_ms < 0:
        parser.error("--standby-latency-ms must be non-negative")
    if args.cases is None:
        args.cases = [case.name for case in CASES]
    if len(set(args.cases)) != len(args.cases):
        parser.error("--case values must be unique")
    return args


def main() -> None:
    args = parse_args()
    experiment = Experiment(
        python=args.python,
        run_id=args.run_id,
        port=args.port,
        repeats=args.repeats,
        drop_steps=args.drop_steps,
        max_tokens=args.max_tokens,
        selected_cases=tuple(args.cases),
        standby_latency_ms=args.standby_latency_ms,
    )
    experiment.execute()
    print(f"completed: {experiment.result_root}")


if __name__ == "__main__":
    main()
