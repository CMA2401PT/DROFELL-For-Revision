#!/usr/bin/env python3
"""Run the reference-shaped basic inference recovery experiment."""

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
    "ordinary": "recover-analysis-basic-ordinary",
    "split": "recover-analysis-basic-split",
}
@dataclass(frozen=True)
class Case:
    name: str
    failure: bool
    retrieve: bool
    topology: str


CASES = (
    Case("normal-reprefill", False, False, "ordinary"),
    Case("recover-reprefill", True, False, "ordinary"),
    Case("recover-split-reprefill", True, False, "split"),
    Case("normal-mix", False, True, "ordinary"),
    Case("recover-mix", True, True, "ordinary"),
    Case("recover-split-mix", True, True, "split"),
)

TOPOLOGY_DEALERS = {
    "ordinary": (
        ("T0-ordinary-a", "0,1,2,3", 0, "cuda:0"),
        ("T0-ordinary-b", "0,1,2,3", 1, "cuda:1"),
        ("T0-ordinary-c", "0,1,2,3", 2, "cuda:0"),
    ),
    "split": (
        ("T0-split-active", "0,1,2,3", 0, "cuda:0"),
        ("T1-split-left", "0,1", 1, "cuda:0"),
        ("T1-split-right", "2,3", 2, "cuda:1"),
    ),
}


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
        inter_dealer_latency_ms: float,
        agent_to_dealer_latency_ms: float,
        dealer_to_agent_latency_ms: float,
        recovery_replay: str,
    ) -> None:
        self.python = python
        self.run_id = run_id
        self.port = port
        self.repeats = repeats
        self.drop_steps = drop_steps
        self.max_tokens = max_tokens
        self.selected_cases = selected_cases
        self.inter_dealer_latency_ms = inter_dealer_latency_ms
        self.agent_to_dealer_latency_ms = agent_to_dealer_latency_ms
        self.dealer_to_agent_latency_ms = dealer_to_agent_latency_ms
        self.recovery_replay = recovery_replay
        self.network_paths = {
            topology: EXP_ROOT / "networks" / f"{topology}-8x4.json"
            for topology in TOPOLOGY_DEALERS
        }
        self.result_root = RESULTS_ROOT / run_id
        self.log_root = EXP_ROOT / "logs" / run_id
        self.environment_path = EXP_ROOT / "environment" / f"{run_id}.json"
        self.commands: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []

    def failure_plan(self, topology: str) -> dict[str, list[int]]:
        if topology == "ordinary":
            # Only the currently active full provider sees each normal-trace
            # task, so this causes A/B/A (or B/A/B) compute-node teardown.
            return {
                "T0-ordinary-a": list(self.drop_steps),
                "T0-ordinary-b": list(self.drop_steps),
                "T0-ordinary-c": list(self.drop_steps),
            }
        if topology == "split":
            return {
                # The first failure turns the full path into left/right
                # halves.  After split-left fails and its range returns to
                # T0-split-active, fail that active left half once more.  A
                # downstream T1-split-right+tail teardown currently cannot be
                # replayed from its shadow state and would hang the request.
                "T0-split-active": [self.drop_steps[0], self.drop_steps[2]],
                "T1-split-left": [self.drop_steps[1]],
            }
        raise ValueError(f"unknown topology: {topology}")

    @property
    def parameters(self) -> dict[str, Any]:
        selected_topologies = {
            case.topology
            for case in CASES
            if case.name in self.selected_cases
        }
        return {
            "model": MODEL,
            "prompt": PROMPT,
            "max_tokens": self.max_tokens,
            "warmup_tokens": 32,
            "parallel_sequences": 1,
            "num_paths": 1,
            "drop_steps": list(self.drop_steps),
            "failure_plans": {
                topology: self.failure_plan(topology)
                for topology in selected_topologies
            },
            "repeats": self.repeats,
            "selected_cases": list(self.selected_cases),
            "methods": ["Re-Prefill", "Mix"],
            "topologies": sorted(selected_topologies),
            "recovery_replay": self.recovery_replay,
            "failure_model": "deterministic_compute_node_teardown",
            "decoder_groups": [8, 8, 8, 8],
            "ordinary_replacement": {
                "initial": "one complete 32-layer provider",
                "after_each_failure": (
                    "one of two surviving complete providers selected by a "
                    "measured path probe"
                ),
            },
            "split_replacement": {
                "initial": "T0-split-active: head + 32 layers + tail",
                "after_first_failure": [
                    "T1-split-left: head + layers 0..15",
                    "T1-split-right: layers 16..31 + tail",
                ],
                "later_failures": (
                    "delete T1-split-left, then delete the T0-split-active left "
                    "replacement"
                ),
            },
            "decoder_backend": "marlin",
            "cuda_graph": True,
            "graph_max_seq_len": 512,
            "max_concurrency": 1,
            "temperature": 0.0,
            "ignore_eos": True,
            "logprobs": 0,
            "disable_gc": True,
            "async_backend": "golang",
            "busy_wait": True,
            "uclamp_min": 1024,
            "peft_in_scope": False,
            "inter_dealer_latency_ms": self.inter_dealer_latency_ms,
            "agent_to_dealer_latency_ms": self.agent_to_dealer_latency_ms,
            "dealer_to_agent_latency_ms": self.dealer_to_agent_latency_ms,
            "network_latency_scope": {
                "inter_dealer": "direct Dealer-to-Dealer links only",
                "agent_to_dealer": "experiment Agent to any Dealer only",
                "dealer_to_agent": "any Dealer to experiment Agent only",
            },
        }

    def initialize(self) -> None:
        if not self.python.is_file():
            raise ExperimentError(f"Python interpreter not found: {self.python}")
        if (
            self.result_root.exists()
            or self.log_root.exists()
            or self.environment_path.exists()
        ):
            raise ExperimentError(
                f"refusing to overwrite run id {self.run_id!r}"
            )
        if not port_available(self.port):
            raise ExperimentError(f"port {self.port} is already in use")
        self.result_root.mkdir(parents=True)
        self.log_root.mkdir(parents=True)
        self.environment_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.environment_path, self.collect_environment())
        self.flush_manifest()

    def collect_environment(self) -> dict[str, Any]:
        probes = {
            "git_commit": ("git", "rev-parse", "HEAD"),
            "git_status": (
                "git", "status", "--short", "--untracked-files=all"
            ),
            "nvidia_smi": (
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.used,memory.total",
                "--format=csv,noheader",
            ),
            "python_torch": (
                str(self.python),
                "-c",
                "import platform,torch; print(platform.python_version()); "
                "print(torch.__version__); print(torch.version.cuda); "
                "print(torch.cuda.device_count())",
            ),
            "lscpu": ("lscpu",),
            "uclampset": ("uclampset", "--version"),
        }
        return {
            "schema_version": 2,
            "run_id": self.run_id,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(REPO_ROOT),
            "python": str(self.python),
            "port": self.port,
            "parameters": self.parameters,
            "probes": {
                name: capture(command) for name, command in probes.items()
            },
        }

    def flush_manifest(self) -> None:
        write_json(
            self.result_root / "manifest.json",
            {
                "schema_version": 2,
                "run_id": self.run_id,
                "parameters": self.parameters,
                "networks": {
                    topology: str(path.relative_to(SUITE_ROOT))
                    for topology, path in self.network_paths.items()
                },
                "commands": self.commands,
                "runs": self.runs,
            },
        )

    def record_command(
        self,
        *,
        purpose: str,
        command: Sequence[str],
        log_path: Path | None = None,
        returncode: int | None = None,
    ) -> dict[str, Any]:
        record = {
            "purpose": purpose,
            "command": command_text(command),
            "log": (
                None
                if log_path is None
                else str(log_path.relative_to(SUITE_ROOT))
            ),
            "returncode": returncode,
        }
        self.commands.append(record)
        self.flush_manifest()
        return record

    def generate_network(self, topology: str) -> None:
        output = self.network_paths[topology]
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.python),
            "-m", "scripts.online.benchmark.generate_network",
            "--name", NETWORK_NAMES[topology],
            "--model", MODEL,
            "--decoder-group-sizes", "8", "8", "8", "8",
        ]
        for name, groups, _, _ in TOPOLOGY_DEALERS[topology]:
            command.extend(("--dealer", f"{name}={groups}"))
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
            purpose=f"generate-network:{topology}",
            command=command,
            returncode=result["returncode"],
        )
        if result["returncode"] != 0:
            raise ExperimentError(
                f"{topology} network generation failed:\n{result['output']}"
            )

    def network_command(self, topology: str) -> list[str]:
        command = [
            "uclampset", "-m", "1024", "--",
            str(self.python),
            str(EXP_ROOT / "launch_network.py"),
            "local",
            "--network", str(self.network_paths[topology]),
            "--port", str(self.port),
            "--advertise-host", "127.0.0.1",
            "--async-backend", "golang",
            "--busy-wait",
        ]
        for name, _, process, device in TOPOLOGY_DEALERS[topology]:
            command.extend((
                "--place", name, str(process), "0", device,
            ))
        return command

    def benchmark_command(
        self,
        output_dir: Path,
        host_id: str,
        retrieve: bool,
    ) -> list[str]:
        command = [
            "uclampset", "-m", "1024", "--",
            str(self.python),
            "-m", "scripts.online.benchmark.benchmark_infer",
            "--measurement", "request",
            "--orbit", f"127.0.0.1:{self.port}",
            "--host-id", host_id,
            "--advertise-host", "127.0.0.1",
            "--model", MODEL,
            "--output", str(output_dir),
            "--prompt", PROMPT,
            "--max-tokens", str(self.max_tokens),
            "--warmup-tokens", "32",
            "--seqs", "1",
            "--num-paths", "1",
            "--logprobs", "0",
            "--disable-gc",
            "--async-backend", "golang",
            "--busy-wait",
        ]
        if retrieve:
            command.append("--retrieve")
        return command

    def run_case(self, case: Case, repeat: int) -> None:
        label = f"{case.name}-repeat-{repeat}"
        relative_run_dir = Path(case.name) / f"repeat-{repeat}"
        run_dir = self.result_root / relative_run_dir
        raw_dir = run_dir / "raw"
        metrics_path = run_dir / "runtime_metrics.jsonl"
        run_dir.mkdir(parents=True)
        network_log = self.log_root / f"{label}-network.log"
        benchmark_log = self.log_root / f"{label}-benchmark.log"
        failure_plan = self.failure_plan(case.topology) if case.failure else {}
        host_id = f"recover-basic-{self.run_id}-{case.name}-r{repeat}"
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(REPO_ROOT),
                "PYTHONUNBUFFERED": "1",
                "DROFELL_METRICS_PATH": str(metrics_path.resolve()),
                "RECOVER_ANALYSIS_DROP_PLAN": json.dumps(
                    failure_plan, separators=(",", ":")
                ),
                "RECOVERY_REPLAY": self.recovery_replay,
                "RECOVER_ANALYSIS_DEALERS": json.dumps(
                    [item[0] for item in TOPOLOGY_DEALERS[case.topology]],
                    separators=(",", ":"),
                ),
                "RECOVER_ANALYSIS_INTER_DEALER_LATENCY_MS": (
                    f"{self.inter_dealer_latency_ms:.6f}"
                ),
                "RECOVER_ANALYSIS_AGENT_NAME": host_id,
                "RECOVER_ANALYSIS_AGENT_TO_DEALER_LATENCY_MS": (
                    f"{self.agent_to_dealer_latency_ms:.6f}"
                ),
                "RECOVER_ANALYSIS_DEALER_TO_AGENT_LATENCY_MS": (
                    f"{self.dealer_to_agent_latency_ms:.6f}"
                ),
                "DROFELL_FORCE_INFER_PATH_PROBE": "1",
            }
        )

        if not port_available(self.port):
            raise ExperimentError(f"port {self.port} is not free before {label}")
        network_command = self.network_command(case.topology)
        with network_log.open("wb") as network_file:
            network_process = subprocess.Popen(
                network_command,
                cwd=REPO_ROOT,
                env=env,
                stdout=network_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        network_record = self.record_command(
            purpose=f"network:{label}",
            command=network_command,
            log_path=network_log,
        )
        benchmark_returncode: int | None = None
        error: str | None = None
        try:
            wait_for_network(
                network_process,
                network_log,
                NETWORK_NAMES[case.topology],
                timeout=240.0,
            )
            benchmark_command = self.benchmark_command(
                raw_dir,
                host_id=host_id,
                retrieve=case.retrieve,
            )
            with benchmark_log.open("wb") as benchmark_file:
                benchmark = subprocess.run(
                    benchmark_command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=benchmark_file,
                    stderr=subprocess.STDOUT,
                    timeout=900.0,
                    check=False,
                )
            benchmark_returncode = benchmark.returncode
            self.record_command(
                purpose=f"benchmark:{label}",
                command=benchmark_command,
                log_path=benchmark_log,
                returncode=benchmark_returncode,
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
            self.runs.append(
                {
                    "case": case.name,
                    "repeat": repeat,
                    "failure": case.failure,
                    "retrieve": case.retrieve,
                    "topology": case.topology,
                    "failure_plan": failure_plan,
                    "run_dir": str(relative_run_dir),
                    "network_log": str(network_log.relative_to(SUITE_ROOT)),
                    "benchmark_log": str(benchmark_log.relative_to(SUITE_ROOT)),
                    "benchmark_returncode": benchmark_returncode,
                    "error": error,
                }
            )
            self.flush_manifest()
            deadline = time.monotonic() + 30.0
            while not port_available(self.port) and time.monotonic() < deadline:
                time.sleep(0.25)
            time.sleep(0.5)

    def analyze(self) -> None:
        command = [
            str(self.python),
            str(EXP_ROOT / "analyze.py"),
            str(self.result_root),
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
        self.record_command(
            purpose="analyze",
            command=command,
            log_path=log_path,
            returncode=completed.returncode,
        )
        if completed.returncode != 0:
            raise ExperimentError(
                f"analysis failed:\n{read_text(log_path)[-16000:]}"
            )

    def execute(self) -> None:
        self.initialize()
        selected_topologies = {
            case.topology
            for case in CASES
            if case.name in self.selected_cases
        }
        for topology in sorted(selected_topologies):
            self.generate_network(topology)
        case_by_name = {case.name: case for case in CASES}
        for repeat in range(self.repeats):
            for case_name in self.selected_cases:
                self.run_case(case_by_name[case_name], repeat)
        self.analyze()
        environment = json.loads(
            self.environment_path.read_text(encoding="utf-8")
        )
        environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
        environment["post_run_nvidia_smi"] = capture(
            (
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            )
        )
        write_json(self.environment_path, environment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run-id", default=f"basic-{utc_run_id()}")
    parser.add_argument("--port", type=int, default=7265)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--drop-steps",
        nargs="+",
        type=int,
        default=(40, 100, 160),
        metavar="STEP",
        help=(
            "Strictly increasing deterministic decode-step failures. Ordinary "
            "topologies accept any positive count; split topologies require "
            "exactly three steps."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--inter-dealer-latency-ms",
        type=float,
        default=0.0,
        help=(
            "One-way latency added only to direct Dealer-to-Dealer links; "
            "Agent, Request, Aux, and Orbit links remain unshaped."
        ),
    )
    parser.add_argument(
        "--agent-to-dealer-latency-ms",
        type=float,
        default=0.0,
        help="One-way latency added only from this experiment's Agent to Dealers.",
    )
    parser.add_argument(
        "--dealer-to-agent-latency-ms",
        type=float,
        default=0.0,
        help="One-way latency added only from Dealers to this experiment's Agent.",
    )
    parser.add_argument(
        "--recovery-replay",
        choices=("chunk", "sequential"),
        default="chunk",
        help="Diagnostic override; the deliverable configuration is chunk.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.name for case in CASES),
        dest="cases",
        help="Run only selected cases; repeat the option as needed.",
    )
    args = parser.parse_args()
    args.drop_steps = tuple(args.drop_steps)
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if any(step < 1 for step in args.drop_steps):
        parser.error("--drop-steps must be positive")
    if tuple(sorted(args.drop_steps)) != args.drop_steps:
        parser.error("--drop-steps must be strictly increasing")
    if len(set(args.drop_steps)) != len(args.drop_steps):
        parser.error("--drop-steps must be unique")
    if args.max_tokens <= args.drop_steps[-1] + 8:
        parser.error(
            "--max-tokens must leave at least 8 outputs after the last drop"
        )
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.inter_dealer_latency_ms < 0:
        parser.error("--inter-dealer-latency-ms must be non-negative")
    if args.agent_to_dealer_latency_ms < 0:
        parser.error("--agent-to-dealer-latency-ms must be non-negative")
    if args.dealer_to_agent_latency_ms < 0:
        parser.error("--dealer-to-agent-latency-ms must be non-negative")
    if args.cases is None:
        args.cases = [case.name for case in CASES]
    if len(set(args.cases)) != len(args.cases):
        parser.error("--case values must be unique")
    selected = [case for case in CASES if case.name in args.cases]
    if any(case.topology == "split" for case in selected) and len(args.drop_steps) != 3:
        parser.error("split topology cases require exactly three --drop-steps")
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
        inter_dealer_latency_ms=args.inter_dealer_latency_ms,
        agent_to_dealer_latency_ms=args.agent_to_dealer_latency_ms,
        dealer_to_agent_latency_ms=args.dealer_to_agent_latency_ms,
        recovery_replay=args.recovery_replay,
    )
    experiment.execute()
    print(f"completed: {experiment.result_root}")


if __name__ == "__main__":
    main()
