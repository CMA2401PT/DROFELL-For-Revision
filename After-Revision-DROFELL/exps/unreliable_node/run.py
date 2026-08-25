#!/usr/bin/env python3
"""Run the paper-matrix unreliable-node reproduction experiment.

The script intentionally uses the public online benchmark CLIs.  It records
every expanded command and keeps each topology/deployment in a fresh network
process so recovery state cannot leak between comparisons.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path(
    "/root/miniconda3/bin/python"
)
MODEL = "teared-Llama-2-7B-GPTQ"
PROMPT = "once upon a time"
AGENT_NAME = "unreliable-persistent-agent"
FAILURE_LOG_MARKER = "trigger compute-node drop"
INSPECTION_POSTPONE = 0.3
FORMAL_MAX_TOKENS = 1028
WARMUP_TOKENS = 32
LOGPROBS = 0
# The 8-GPU extension uses the same probability grid as the paper and PETALS
# so every reported cell has a directly comparable baseline.
DEFAULT_REPEATS = 10
PAPER_PROBABILITIES = (0.0, 1e-4, 1e-3, 1e-2, 5e-2)
REPEATS = 5
FAILURE_PROBABILITIES = PAPER_PROBABILITIES
FORMAL_STALL_TIMEOUT = 600.0
FORMAL_TIMEOUT = 3600.0
SOURCE_FINGERPRINT_ROOTS = (
    "node",
    "scripts/online",
    "operators",
    "dealers",
    "utils",
    "orbit",
    "cuda_extends",
)
SOURCE_FINGERPRINT_SUFFIXES = {".py", ".go", ".cu", ".cpp", ".h"}


@dataclass(frozen=True)
class Topology:
    name: str
    cards: int
    nodes: int
    redundancy: int
    group_sizes: tuple[int, ...]
    dealers: tuple[tuple[str, tuple[int, ...]], ...]
    placements: tuple[tuple[str, int, int, str], ...]
    probabilities: tuple[float, ...]
    num_paths: int
    agent_anchor: str


TOPOLOGIES = (
    Topology(
        name="1_1_1",
        cards=1,
        nodes=1,
        redundancy=1,
        group_sizes=(32,),
        dealers=(("dealer-g0", (0,)),),
        placements=(("dealer-g0", 0, 0, "cuda:0"),),
        probabilities=PAPER_PROBABILITIES,
        num_paths=1,
        agent_anchor="dealer-g0",
    ),
    Topology(
        name="1_2_1",
        cards=1,
        nodes=2,
        redundancy=1,
        group_sizes=(16, 16),
        dealers=(("dealer-g0", (0,)), ("dealer-g1", (1,))),
        placements=(
            ("dealer-g0", 0, 0, "cuda:0"),
            ("dealer-g1", 0, 0, "cuda:0"),
        ),
        probabilities=PAPER_PROBABILITIES,
        num_paths=1,
        agent_anchor="dealer-g0",
    ),
    Topology(
        name="1_4_1",
        cards=1,
        nodes=4,
        redundancy=1,
        group_sizes=(8, 8, 8, 8),
        dealers=tuple((f"dealer-g{group}", (group,)) for group in range(4)),
        placements=tuple(
            (f"dealer-g{group}", 0, 0, "cuda:0") for group in range(4)
        ),
        probabilities=PAPER_PROBABILITIES,
        num_paths=1,
        agent_anchor="dealer-g0",
    ),
    Topology(
        name="2_2_1",
        cards=2,
        nodes=2,
        redundancy=1,
        group_sizes=(16, 16),
        dealers=(("dealer-g0", (0,)), ("dealer-g1", (1,))),
        placements=(
            ("dealer-g0", 0, 0, "cuda:0"),
            ("dealer-g1", 1, 0, "cuda:1"),
        ),
        probabilities=PAPER_PROBABILITIES,
        num_paths=1,
        agent_anchor="dealer-g0",
    ),
    Topology(
        name="2_4_1",
        cards=2,
        nodes=4,
        redundancy=1,
        group_sizes=(8, 8, 8, 8),
        dealers=tuple((f"dealer-g{group}", (group,)) for group in range(4)),
        placements=(
            ("dealer-g0", 0, 0, "cuda:0"),
            ("dealer-g1", 0, 0, "cuda:0"),
            ("dealer-g2", 1, 0, "cuda:1"),
            ("dealer-g3", 1, 0, "cuda:1"),
        ),
        # MERGE: 2_4_1 是论文 tab:failure_rate 的格 ⇒ 用 PAPER_PROBABILITIES。
        # 8 卡清单里它只作保留基线，不需要粗栅格。
        probabilities=PAPER_PROBABILITIES,
        num_paths=1,
        agent_anchor="dealer-g0",
    ),
    Topology(
        name="2_1_2",
        cards=2,
        nodes=1,
        redundancy=2,
        group_sizes=(32,),
        dealers=(
            ("dealer-g0-r0", (0,)),
            ("dealer-g0-r1", (0,)),
        ),
        placements=(
            ("dealer-g0-r0", 0, 0, "cuda:0"),
            ("dealer-g0-r1", 1, 0, "cuda:1"),
        ),
        probabilities=PAPER_PROBABILITIES,
        num_paths=2,
        agent_anchor="dealer-g0-r0",
    ),
    Topology(
        name="2_2_2",
        cards=2,
        nodes=2,
        redundancy=2,
        group_sizes=(16, 16),
        dealers=tuple(
            (f"dealer-g{group}-r{replica}", (group,))
            for group in range(2)
            for replica in range(2)
        ),
        placements=tuple(
            (f"dealer-g{group}-r{replica}", replica, 0, f"cuda:{replica}")
            for group in range(2)
            for replica in range(2)
        ),
        probabilities=PAPER_PROBABILITIES,
        num_paths=2,
        agent_anchor="dealer-g0-r0",
    ),
    Topology(
        name="2_4_2",
        cards=2,
        nodes=4,
        redundancy=2,
        group_sizes=(8, 8, 8, 8),
        dealers=tuple(
            (f"dealer-g{group}-r{replica}", (group,))
            for group in range(4)
            for replica in range(2)
        ),
        placements=tuple(
            (f"dealer-g{group}-r{replica}", replica, 0, f"cuda:{replica}")
            for group in range(4)
            for replica in range(2)
        ),
        # MERGE: 2_4_2 同样是论文格 ⇒ PAPER_PROBABILITIES。
        probabilities=PAPER_PROBABILITIES,
        num_paths=2,
        agent_anchor="dealer-g0-r0",
    ),
    # MERGE: 1_4_2 只在 8 卡侧(autodl)存在，是单卡 4 节点 2 副本的诊断拓扑
    # （副本同卡 ⇒ 属"假冗余"，只跑 p=0 作对照，不能用来主张容错）。
    # 论文表里没有这一格，保留但不进论文矩阵。
    Topology(
        name="1_4_2",
        cards=1,
        nodes=4,
        redundancy=2,
        group_sizes=(8, 8, 8, 8),
        dealers=tuple(
            (f"dealer-g{group}-r{replica}", (group,))
            for group in range(4)
            for replica in range(2)
        ),
        placements=tuple(
            (f"dealer-g{group}-r{replica}", replica, 0, "cuda:0")
            for group in range(4)
            for replica in range(2)
        ),
        probabilities=(0.0,),
        num_paths=2,
        agent_anchor="dealer-g0-r0",
    ),
    Topology(
        name="4_4_1",
        cards=4,
        nodes=4,
        redundancy=1,
        group_sizes=(8, 8, 8, 8),
        dealers=tuple((f"dealer-g{group}", (group,)) for group in range(4)),
        placements=tuple(
            (f"dealer-g{group}", group, 0, f"cuda:{group}")
            for group in range(4)
        ),
        probabilities=FAILURE_PROBABILITIES,
        num_paths=1,
        agent_anchor="dealer-g0",
    ),
    Topology(
        name="8_4_2",
        cards=8,
        nodes=4,
        redundancy=2,
        group_sizes=(8, 8, 8, 8),
        dealers=tuple(
            (f"dealer-g{group}-r{replica}", (group,))
            for group in range(4)
            for replica in range(2)
        ),
        # Give every replica its own physical GPU.  In particular, the two
        # replicas of one decoder group can never share a card.
        placements=tuple(
            (
                f"dealer-g{group}-r{replica}",
                group * 2 + replica,
                0,
                f"cuda:{group * 2 + replica}",
            )
            for group in range(4)
            for replica in range(2)
        ),
        probabilities=FAILURE_PROBABILITIES,
        num_paths=2,
        agent_anchor="dealer-g0-r0",
    ),
    Topology(
        name="8_8_1",
        cards=8,
        nodes=8,
        redundancy=1,
        group_sizes=(4, 4, 4, 4, 4, 4, 4, 4),
        dealers=tuple((f"dealer-g{group}", (group,)) for group in range(8)),
        placements=tuple(
            (f"dealer-g{group}", group, 0, f"cuda:{group}")
            for group in range(8)
        ),
        probabilities=FAILURE_PROBABILITIES,
        num_paths=1,
        agent_anchor="dealer-g0",
    ),
    Topology(
        name="8_4_4",
        cards=8,
        nodes=4,
        redundancy=4,
        group_sizes=(8, 8, 8, 8),
        dealers=tuple(
            (f"dealer-g{group}-r{replica}", (group,))
            for group in range(4)
            for replica in range(4)
        ),
        # Each group has four replicas on four distinct cards.  The cyclic
        # placement also uses all eight GPUs and balances two 8-layer shards
        # per card across the complete topology.
        placements=tuple(
            (
                f"dealer-g{group}-r{replica}",
                (group + 2 * replica) % 8,
                0,
                f"cuda:{(group + 2 * replica) % 8}",
            )
            for group in range(4)
            for replica in range(4)
        ),
        probabilities=FAILURE_PROBABILITIES,
        num_paths=4,
        agent_anchor="dealer-g0-r0",
    ),
)


def experiment_parameters(
    *,
    repeats: int,
    topologies: Sequence[Topology],
    include_persistent_p0: bool,
    capture_metrics: bool,
) -> dict[str, Any]:
    return {
        "model": MODEL,
        "decoder_backend": "marlin",
        "cuda_graph": True,
        "failure_injection": True,
        "max_seq_len": 2048,
        "max_concurrency": 1,
        "prompt": PROMPT,
        "formal_max_tokens": FORMAL_MAX_TOKENS,
        "warmup_tokens": WARMUP_TOKENS,
        "sequences": 1,
        "repeats": repeats,
        "inspection_postpone_seconds": {
            "combined-request-agent,p=0": 0.0,
            "combined-request-agent,p!=0": INSPECTION_POSTPONE,
            "persistent-agent": 0.0,
        },
        "temperature": 0.0,
        "ignore_eos": True,
        "logprobs": LOGPROBS,
        "retrieve": False,
        "detail_trace": False,
        "disable_gc": True,
        "async_backend": "golang",
        "busy_wait": True,
        "uclamp_min": 1024,
        "drofell_metrics_path": (
            "results/<run-id>/runtime_metrics.jsonl" if capture_metrics else None
        ),
        "memory": {
            "policy": "static",
            "major_reserved_mib": 8192,
            "major_percentage": 0.8,
            "unmanaged_reserved_mib": 32,
            "minor_min_mib": 0,
            "minor_max_mib": 4096,
            "operate_reserved_mib": 1024,
            "graph_budget_mib": 1024,
        },
        "topologies": [asdict(topology) for topology in topologies],
        "deployment_probabilities": {
            # MERGE: 取 3090 侧（按 topologies[0] 实际概率 + 受
            # --include-persistent-p0 控制）。不要退回写死 FAILURE_PROBABILITIES：
            # 那样在跑论文栅格时会把环境快照记成错的概率列表。
            # persistent-agent 只有 p=0 可比 —— p!=0 的持久 Agent 数据已作废
            # （0.3s InspectPostpone 靠 benchmark 进程内重绑定，不能跨进程传播）。
            "combined-request-agent": list(topologies[0].probabilities),
            "persistent-agent": (
                [0.0]
                if include_persistent_p0
                and 0.0 in topologies[0].probabilities
                else []
            ),
        },
    }


class ExperimentError(RuntimeError):
    pass


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def probability_label(value: float) -> str:
    return format(value, ".8g").replace(".", "p")


def deployment_probabilities(
    topology: Topology,
    deployment: str,
) -> tuple[float, ...]:
    if deployment == "combined-request-agent":
        return topology.probabilities
    if deployment == "persistent-agent":
        return tuple(
            probability
            for probability in topology.probabilities
            if probability == 0.0
        )
    raise ExperimentError(f"unknown deployment: {deployment}")


def command_text(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def run_capture(command: Sequence[str], *, check: bool = False) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and completed.returncode != 0:
        raise ExperimentError(
            f"command failed ({completed.returncode}): {command_text(command)}\n"
            f"{completed.stdout}"
        )
    return {
        "command": command_text(command),
        "returncode": completed.returncode,
        "output": completed.stdout,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint() -> dict[str, Any]:
    paths = {
        Path(__file__).resolve(),
        Path(__file__).with_name("run_cell.py").resolve(),
    }
    for root_name in SOURCE_FINGERPRINT_ROOTS:
        root = REPO_ROOT / root_name
        if not root.is_dir():
            raise ExperimentError(f"missing source root: {root}")
        paths.update(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix in SOURCE_FINGERPRINT_SUFFIXES
        )
    files = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in sorted(paths)
    }
    payload = json.dumps(
        files, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "algorithm": "sha256",
        "aggregate_sha256": hashlib.sha256(payload).hexdigest(),
        "files": files,
    }


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def wait_for_text(
    process: subprocess.Popen[bytes],
    log_path: Path,
    needle: str,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ExperimentError(
                f"network exited before ready (code={process.returncode}):\n"
                f"{read_text(log_path)[-12000:]}"
            )
        if needle in read_text(log_path):
            return
        time.sleep(0.5)
    raise ExperimentError(
        f"timed out waiting for {needle!r}:\n{read_text(log_path)[-12000:]}"
    )


def process_uclamp(pid: int) -> str:
    last = ""
    for _ in range(40):
        result = run_capture(("uclampset", "-p", str(pid)))
        last = result["output"]
        if "min: 1024" in last:
            return last.strip()
        time.sleep(0.05)
    return last.strip()


def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    for sig, timeout in (
        (signal.SIGINT, 30.0),
        (signal.SIGTERM, 15.0),
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
    def __init__(
        self,
        *,
        python: Path,
        run_id: str,
        port: int,
        repeats: int,
        topologies: tuple[Topology, ...],
        include_persistent_p0: bool,
        capture_metrics: bool,
        skip_smoke: bool,
    ) -> None:
        self.python = python
        self.run_id = run_id
        self.port = port
        self.repeats = repeats
        self.topologies = topologies
        self.include_persistent_p0 = include_persistent_p0
        self.capture_metrics = capture_metrics
        self.skip_smoke = skip_smoke
        self.network_dir = EXP_ROOT / "networks"
        self.log_dir = EXP_ROOT / "logs" / run_id
        self.result_dir = EXP_ROOT / "results" / run_id
        self.environment_path = EXP_ROOT / "environment" / f"{run_id}.json"
        # MERGE 2026-08-17: 这里原先被合并成两次赋值 —— 先赋真实路径，
        # 隔几行又被 autodl 侧的 `= None` 覆盖，导致 --capture-metrics 时
        # collect_environment() 里 self.metrics_path.relative_to() 直接
        # AttributeError。路径恒定，是否真的写指标由 self.capture_metrics 决定，
        # 所以只保留这一次赋值，不要再加 `metrics_path = None` 的哨兵。
        self.metrics_path = self.result_dir / "runtime_metrics.jsonl"
        self.commands: list[dict[str, Any]] = []
        self.network_runs: list[dict[str, Any]] = []
        self.smoke_results: list[dict[str, Any]] = []
        self.environment: dict[str, Any] = {}

    def initialize(self) -> None:
        if not self.python.is_file():
            raise ExperimentError(f"Python interpreter does not exist: {self.python}")
        if not port_available(self.port):
            raise ExperimentError(f"port {self.port} is already in use")
        for path in (self.log_dir, self.result_dir, self.environment_path):
            if path.exists():
                raise ExperimentError(f"refusing to overwrite existing path: {path}")
        self.network_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True)
        self.result_dir.mkdir(parents=True)
        self.environment_path.parent.mkdir(parents=True, exist_ok=True)
        self.environment = self.collect_environment()
        self.flush_environment()

    def collect_environment(self) -> dict[str, Any]:
        probes = {
            "git_commit": ("git", "rev-parse", "HEAD"),
            "git_status": (
                "git", "status", "--short", "--untracked-files=all"
            ),
            "git_diff_stat": ("git", "diff", "--stat"),
            "lscpu": ("lscpu",),
            "nvidia_smi": (
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,"
                "power.draw,clocks.current.sm,clocks.max.sm,memory.used,memory.total",
                "--format=csv,noheader",
            ),
            "python_torch": (
                str(self.python),
                "-c",
                "import platform,torch; print(platform.python_version()); "
                "print(torch.__version__); print(torch.version.cuda); "
                "print(torch.cuda.device_count())",
            ),
            "uclampset": ("uclampset", "--version"),
        }
        result = {
            "schema_version": 1,
            "run_id": self.run_id,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(REPO_ROOT),
            "python": str(self.python),
            "port": self.port,
            "parameters": experiment_parameters(
                repeats=self.repeats,
                topologies=self.topologies,
                include_persistent_p0=self.include_persistent_p0,
                capture_metrics=self.capture_metrics,
            ),
            "drofell_metrics_path_present": "DROFELL_METRICS_PATH" in os.environ,
            "capture_metrics": self.capture_metrics,
            "skip_smoke": self.skip_smoke,
            "source_fingerprint_start": source_fingerprint(),
            "metrics_path": (
                str(self.metrics_path.relative_to(EXP_ROOT))
                if self.capture_metrics else None
            ),
            "probes": {name: run_capture(command) for name, command in probes.items()},
            "cpu_frequency": {},
            "commands": self.commands,
            "network_runs": self.network_runs,
            "smoke_results": self.smoke_results,
        }
        for name in (
            "scaling_driver",
            "scaling_governor",
            "scaling_available_governors",
            "energy_performance_preference",
        ):
            path = Path(f"/sys/devices/system/cpu/cpu0/cpufreq/{name}")
            result["cpu_frequency"][name] = read_text(path).strip()
        return result

    def flush_environment(self) -> None:
        self.environment["commands"] = self.commands
        self.environment["network_runs"] = self.network_runs
        self.environment["smoke_results"] = self.smoke_results
        write_json(self.environment_path, self.environment)

    def record_command(
        self,
        *,
        purpose: str,
        command: Sequence[str],
        log_path: Path | None = None,
        pid: int | None = None,
        uclamp: str = "",
        returncode: int | None = None,
    ) -> dict[str, Any]:
        record = {
            "purpose": purpose,
            "command": command_text(command),
            "log": None if log_path is None else str(log_path.relative_to(EXP_ROOT)),
            "pid": pid,
            "uclamp": uclamp,
            "returncode": returncode,
        }
        self.commands.append(record)
        self.flush_environment()
        return record

    def generate_networks(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        snapshot_dir = self.result_dir / "network_specs"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for topology in self.topologies:
            output = self.network_dir / f"{topology.name}-graph-failure.json"
            command = [
                str(self.python),
                "-m", "scripts.online.benchmark.generate_network",
                "--name", f"unreliable-{topology.name}-graph-failure",
                "--model", MODEL,
                "--decoder-group-sizes", *(str(size) for size in topology.group_sizes),
            ]
            for dealer, groups in topology.dealers:
                command.extend(("--dealer", f"{dealer}={','.join(map(str, groups))}"))
            command.extend((
                "--decoder-backend", "marlin",
                "--max-seq-len", "2048",
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
            result = run_capture(command, check=True)
            self.record_command(
                purpose=f"generate-network:{topology.name}",
                command=command,
                returncode=result["returncode"],
            )
            shutil.copy2(output, snapshot_dir / output.name)
            paths[topology.name] = output
        return paths

    def network_command(
        self,
        topology: Topology,
        network: Path,
        deployment: str,
    ) -> list[str]:
        command = [
            "uclampset", "-m", "1024", "--",
            str(self.python),
            "-m", "scripts.online.benchmark.launch_network",
            "local",
            "--network", str(network),
            "--port", str(self.port),
            "--advertise-host", "127.0.0.1",
            "--async-backend", "golang",
            "--busy-wait",
        ]
        for dealer, process, thread, device in topology.placements:
            command.extend((
                "--place", dealer, str(process), str(thread), device,
            ))
        if deployment == "persistent-agent":
            command.extend((
                "--agent", AGENT_NAME, topology.agent_anchor,
                "--agent-path-search-redundancy",
                str(max(1, topology.redundancy - 1)),
            ))
        return command

    def benchmark_command(
        self,
        *,
        topology: Topology,
        deployment: str,
        probability: float,
        output: Path,
        max_tokens: int,
        repeats: int,
        host_suffix: str,
    ) -> list[str]:
        inspection_postpone = (
            INSPECTION_POSTPONE
            if deployment == "combined-request-agent" and probability != 0.0
            else 0.0
        )
        command = [
            "uclampset", "-m", "1024", "--",
            str(self.python),
            "-m", "scripts.online.benchmark.benchmark_infer",
            "--measurement", "failure",
            "--orbit", f"127.0.0.1:{self.port}",
            "--host-id", f"unreliable-{self.run_id}-{host_suffix}",
            "--advertise-host", "127.0.0.1",
            "--model", MODEL,
            "--output", str(output),
            "--prompt", PROMPT,
            "--max-tokens", str(max_tokens),
            "--warmup-tokens", str(WARMUP_TOKENS),
            "--seqs", "1",
            "--drop-probabilities", format(probability, ".8g"),
            "--repeats", str(repeats),
            "--inspection-postpone", str(inspection_postpone),
            "--num-paths", str(topology.num_paths),
            "--path-search-redundancy",
            str(max(1, topology.redundancy - 1)),
            "--logprobs", str(LOGPROBS),
            "--disable-gc",
            "--async-backend", "golang",
            "--busy-wait",
        ]
        if deployment == "persistent-agent":
            command.extend(("--agent", AGENT_NAME))
        return command

    def start_network(
        self,
        *,
        topology: Topology,
        network: Path,
        deployment: str,
        label: str,
    ) -> tuple[subprocess.Popen[bytes], Path, dict[str, Any]]:
        if not port_available(self.port):
            raise ExperimentError(f"port {self.port} is not free before {label}")
        log_path = self.log_dir / f"{label}-network.log"
        command = self.network_command(topology, network, deployment)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        # MERGE 2026-08-17: 这里原是 autodl 侧的 `metrics_path is None` 哨兵。
        # 去掉重复赋值后 metrics_path 恒非 None ⇒ 旧写法会变成"永远设置
        # DROFELL_METRICS_PATH"，让每个吞吐格都带上热路径 JSONL I/O。
        # 必须与 run_benchmark_process 一样改用 self.capture_metrics。
        if self.capture_metrics:
            env["DROFELL_METRICS_PATH"] = str(self.metrics_path.resolve())
        else:
            env.pop("DROFELL_METRICS_PATH", None)
        log_file = log_path.open("wb")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_file.close()
        record = self.record_command(
            purpose=f"network:{label}",
            command=command,
            log_path=log_path,
            pid=process.pid,
            uclamp=process_uclamp(process.pid),
        )
        try:
            wait_for_text(process, log_path, "ready at", timeout=600.0)
        except Exception:
            stop_process_group(process)
            record["returncode"] = process.returncode
            self.flush_environment()
            raise
        return process, log_path, record

    def run_benchmark_process(
        self,
        *,
        command: Sequence[str],
        purpose: str,
        log_path: Path,
        timeout: float,
        stall_timeout: float | None = None,
    ) -> dict[str, Any]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        # MERGE: 取 3090 侧的 capture_metrics 显式开关（autodl 侧用
        # `metrics_path is None` 当哨兵，是旧写法）。两者语义等价，但显式开关
        # 已被 12 处调用点串起来，且能在 environment 快照里记录该次是否采指标。
        # 重要：吞吐格必须关掉它 —— 热路径 JSONL I/O 会污染吞吐数字。
        if self.capture_metrics:
            env["DROFELL_METRICS_PATH"] = str(self.metrics_path.resolve())
        else:
            env.pop("DROFELL_METRICS_PATH", None)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(
                list(command),
                cwd=REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record = self.record_command(
                purpose=purpose,
                command=command,
                log_path=log_path,
                pid=process.pid,
                uclamp=process_uclamp(process.pid),
            )
            deadline = time.monotonic() + timeout
            last_progress = time.monotonic()
            last_size = log_path.stat().st_size
            failure_reason: str | None = None
            while process.poll() is None:
                now = time.monotonic()
                size = log_path.stat().st_size
                if size != last_size:
                    last_size = size
                    last_progress = now
                if now >= deadline:
                    failure_reason = f"benchmark timed out: {purpose}"
                    break
                if (
                    stall_timeout is not None
                    and now - last_progress >= stall_timeout
                ):
                    failure_reason = (
                        f"benchmark made no log progress for "
                        f"{stall_timeout:.0f}s: {purpose}"
                    )
                    break
                time.sleep(min(1.0, max(0.0, deadline - now)))
            if failure_reason is not None:
                stop_process_group(process)
                record["returncode"] = process.returncode
                self.flush_environment()
                raise ExperimentError(
                    f"{failure_reason}\n{read_text(log_path)[-12000:]}"
                )
            returncode = int(process.returncode)
        record["returncode"] = returncode
        self.flush_environment()
        if returncode != 0:
            raise ExperimentError(
                f"benchmark failed ({returncode}): {purpose}\n"
                f"{read_text(log_path)[-12000:]}"
            )
        return record

    def run_smoke(
        self,
        *,
        topology: Topology,
        network: Path,
        deployment: str,
        probability: float,
        max_tokens: int,
        label: str,
    ) -> None:
        process, network_log, network_record = self.start_network(
            topology=topology,
            network=network,
            deployment=deployment,
            label=label,
        )
        output = self.result_dir / "smoke" / label
        benchmark_log = self.log_dir / f"{label}-benchmark.log"
        command = self.benchmark_command(
            topology=topology,
            deployment=deployment,
            probability=probability,
            output=output,
            max_tokens=max_tokens,
            repeats=1,
            host_suffix=label,
        )
        error = ""
        try:
            self.run_benchmark_process(
                command=command,
                purpose=f"smoke:{label}",
                log_path=benchmark_log,
                timeout=900.0,
            )
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
            raise
        finally:
            stop_process_group(process)
            network_record["returncode"] = process.returncode
            smoke = {
                "label": label,
                "deployment": deployment,
                "topology": topology.name,
                "probability": probability,
                "max_tokens": max_tokens,
                "network_log": str(network_log.relative_to(EXP_ROOT)),
                "benchmark_log": str(benchmark_log.relative_to(EXP_ROOT)),
                "injected_failures": read_text(network_log).count(FAILURE_LOG_MARKER),
                "success": not error,
                "error": error,
            }
            self.smoke_results.append(smoke)
            self.flush_environment()

    def run_formal_network(
        self,
        *,
        topology: Topology,
        network: Path,
        deployment: str,
    ) -> None:
        label = f"formal-{deployment}-{topology.name}"
        process, network_log, network_record = self.start_network(
            topology=topology,
            network=network,
            deployment=deployment,
            label=label,
        )
        probability_stats: dict[str, Any] = {}
        try:
            for probability in deployment_probabilities(topology, deployment):
                prob_label = probability_label(probability)
                before = len(read_text(network_log))
                output = self.result_dir / deployment / topology.name
                benchmark_log = self.log_dir / f"{label}-p{prob_label}-benchmark.log"
                command = self.benchmark_command(
                    topology=topology,
                    deployment=deployment,
                    probability=probability,
                    output=output,
                    max_tokens=FORMAL_MAX_TOKENS,
                    repeats=self.repeats,
                    host_suffix=f"{deployment}-{topology.name}-p{prob_label}",
                )
                self.run_benchmark_process(
                    command=command,
                    purpose=f"formal:{deployment}:{topology.name}:p={probability}",
                    log_path=benchmark_log,
                    timeout=FORMAL_TIMEOUT,
                    stall_timeout=FORMAL_STALL_TIMEOUT,
                )
                time.sleep(0.5)
                appended = read_text(network_log)[before:]
                recovery_text = (
                    appended
                    if deployment == "persistent-agent"
                    else read_text(benchmark_log)
                )
                probability_stats[format(probability, ".8g")] = {
                    "injected_failures": appended.count(FAILURE_LOG_MARKER),
                    "benchmark_log": str(benchmark_log.relative_to(EXP_ROOT)),
                    "recovery_latency": recovery_latency_summary(recovery_text),
                }
        finally:
            stop_process_group(process)
            network_record["returncode"] = process.returncode
            network_run = {
                "deployment": deployment,
                "topology": topology.name,
                "network_log": str(network_log.relative_to(EXP_ROOT)),
                "probabilities": probability_stats,
                "total_injected_failures": read_text(network_log).count(
                    FAILURE_LOG_MARKER
                ),
            }
            self.network_runs.append(network_run)
            self.flush_environment()

    def execute(self) -> None:
        networks = self.generate_networks()
        by_name = {topology.name: topology for topology in self.topologies}
        smoke_topology = self.topologies[0]

        if not self.skip_smoke:
            self.run_smoke(
                topology=smoke_topology,
                network=networks[smoke_topology.name],
                deployment="combined-request-agent",
                probability=0.0,
                max_tokens=64,
                label=f"smoke-combined-{smoke_topology.name}-p0",
            )

        persistent_available = self.include_persistent_p0
        persistent_smokes = (
            (
                smoke_topology,
                0.0,
                64,
                f"smoke-persistent-{smoke_topology.name}-p0",
            ),
        )
        for topology, probability, max_tokens, label in (
            persistent_smokes
            if self.include_persistent_p0 and not self.skip_smoke else ()
        ):
            try:
                self.run_smoke(
                    topology=topology,
                    network=networks[topology.name],
                    deployment="persistent-agent",
                    probability=probability,
                    max_tokens=max_tokens,
                    label=label,
                )
            except Exception as error:
                persistent_available = False
                self.environment["persistent_agent_abandoned"] = {
                    "label": label,
                    "error": f"{type(error).__name__}: {error}",
                }
                self.flush_environment()
                break

        deployments = ["combined-request-agent"]
        if persistent_available:
            deployments.append("persistent-agent")
        for topology in self.topologies:
            for deployment in deployments:
                self.run_formal_network(
                    topology=topology,
                    network=networks[topology.name],
                    deployment=deployment,
                )

        self.environment["source_fingerprint_end"] = source_fingerprint()
        self.flush_environment()
        start_fingerprint = self.environment["source_fingerprint_start"]
        end_fingerprint = self.environment["source_fingerprint_end"]
        if start_fingerprint != end_fingerprint:
            raise ExperimentError(
                "runtime source changed during the experiment; refusing to "
                "generate a formal summary"
            )

        summary = summarize_run(
            self.result_dir,
            self.network_runs,
            repeats=self.repeats,
            topologies=self.topologies,
            include_persistent_p0=self.include_persistent_p0,
            capture_metrics=self.capture_metrics,
        )
        write_json(self.result_dir / "summary.json", summary)
        write_summary_csv(self.result_dir / "summary.csv", summary["runs"])
        self.environment["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self.environment["summary"] = str(
            (self.result_dir / "summary.json").relative_to(EXP_ROOT)
        )
        self.environment["post_run_nvidia_smi"] = run_capture((
            "nvidia-smi",
            "--query-gpu=index,pstate,temperature.gpu,power.draw,clocks.current.sm,"
            "memory.used,memory.total",
            "--format=csv,noheader",
        ))
        self.flush_environment()


def legacy_throughput(record: list[list[int]], length: int) -> float:
    sequence_zero = [int(timestamp) for seq, timestamp in record if int(seq) == 0]
    if len(sequence_zero) < length:
        raise ExperimentError(
            f"result contains {len(sequence_zero)} timestamps, needs {length}"
        )
    elapsed = (sequence_zero[length - 1] - sequence_zero[0]) / 1e9
    if elapsed <= 0:
        raise ExperimentError("result timestamps are not increasing")
    return length / elapsed


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_FLOAT = re.compile(r"\b\d+\.\d+\b")
_RECOVERY_START = re.compile(r"\s(\S+) @ \S+ suicide, self healing")
_RECOVERY_END = re.compile(
    r"find replacement of handle of infer_node=\((.+?)@[^)]+\)"
)


def recovery_latency_summary(text: str) -> dict[str, Any]:
    starts: dict[str, list[float]] = {}
    latencies: list[float] = []
    unmatched_ends = 0
    for raw_line in _ANSI_ESCAPE.sub("", text).splitlines():
        times = _FLOAT.findall(raw_line)
        if not times:
            continue
        timestamp = float(times[-1])
        start = _RECOVERY_START.search(raw_line)
        if start is not None:
            starts.setdefault(start.group(1), []).append(timestamp)
            continue
        end = _RECOVERY_END.search(raw_line)
        if end is None:
            continue
        pending = starts.get(end.group(1))
        if not pending:
            unmatched_ends += 1
            continue
        latency = timestamp - pending.pop(0)
        if latency >= 0:
            latencies.append(latency)
    unmatched_starts = sum(len(values) for values in starts.values())
    if not latencies:
        return {
            "count": 0,
            "mean_seconds": None,
            "median_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
            "unmatched_starts": unmatched_starts,
            "unmatched_ends": unmatched_ends,
        }
    return {
        "count": len(latencies),
        "mean_seconds": statistics.mean(latencies),
        "median_seconds": statistics.median(latencies),
        "min_seconds": min(latencies),
        "max_seconds": max(latencies),
        "unmatched_starts": unmatched_starts,
        "unmatched_ends": unmatched_ends,
    }


def count_fallbacks(stats: Any) -> int:
    if isinstance(stats, dict):
        total = 0
        fallbacks = stats.get("fallbacks")
        if isinstance(fallbacks, dict):
            total += sum(int(value) for value in fallbacks.values())
        for key, value in stats.items():
            if key != "fallbacks":
                total += count_fallbacks(value)
        return total
    if isinstance(stats, list):
        return sum(count_fallbacks(item) for item in stats)
    return 0


def fallback_reasons(stats: Any) -> dict[str, int]:
    reasons: dict[str, int] = {}
    if isinstance(stats, dict):
        fallbacks = stats.get("fallbacks")
        if isinstance(fallbacks, dict):
            for reason, count in fallbacks.items():
                reasons[str(reason)] = reasons.get(str(reason), 0) + int(count)
        for key, value in stats.items():
            if key == "fallbacks":
                continue
            for reason, count in fallback_reasons(value).items():
                reasons[reason] = reasons.get(reason, 0) + count
    elif isinstance(stats, list):
        for item in stats:
            for reason, count in fallback_reasons(item).items():
                reasons[reason] = reasons.get(reason, 0) + count
    return reasons


def graph_summary(stats: dict[str, Any]) -> dict[str, Any]:
    decoder_enabled = all(bool(value.get("enabled")) for value in stats.values())
    decoder_captured = all(bool(value.get("captured")) for value in stats.values())
    decoder_replays = sum(int(value.get("replays", 0)) for value in stats.values())
    reasons = fallback_reasons(stats)
    return {
        "dealer_count": len(stats),
        "decoder_enabled": decoder_enabled,
        "decoder_captured": decoder_captured,
        "decoder_replays": decoder_replays,
        "fallbacks": count_fallbacks(stats),
        "fallback_reasons": reasons,
        "non_sampling_fallbacks": sum(
            count for reason, count in reasons.items() if reason != "sampling"
        ),
    }


def validate_keypoints(
    keypoints: Iterable[Sequence[Any]],
    topology: Topology,
) -> list[int]:
    validate_topology(topology)
    expected_keypoints: list[tuple[str, int, int]] = []
    for dealer, raw_groups in topology.dealers:
        groups = sorted(int(group) for group in raw_groups)
        positions = [group + 1 for group in groups]
        if groups[0] == 0:
            positions.insert(0, 0)
        if groups[-1] == topology.nodes - 1:
            positions.append(topology.nodes + 1)
        expected_keypoints.append((dealer, positions[0], len(positions)))

    actual_keypoints: list[tuple[str, int, int]] = []
    coverage = [0 for _ in range(topology.nodes + 2)]
    for item in keypoints:
        if len(item) != 3:
            raise ExperimentError(
                f"{topology.name} malformed keypoint: {item!r}"
            )
        dealer, raw_start, raw_count = item
        start = int(raw_start)
        count = int(raw_count)
        if start < 0 or count <= 0 or start + count > len(coverage):
            raise ExperimentError(
                f"{topology.name} out-of-range keypoint: {item!r}"
            )
        actual_keypoints.append((str(dealer), start, count))
        for index in range(start, start + count):
            coverage[index] += 1
    if sorted(actual_keypoints) != sorted(expected_keypoints):
        raise ExperimentError(
            f"{topology.name} keypoints {sorted(actual_keypoints)}, expected "
            f"{sorted(expected_keypoints)}"
        )
    expected = [topology.num_paths] * (topology.nodes + 2)
    if coverage != expected:
        raise ExperimentError(
            f"{topology.name} keypoint coverage {coverage}, expected "
            f"{expected}"
        )
    return coverage


def validate_topology(topology: Topology) -> None:
    if topology.cards <= 0 or topology.nodes <= 0:
        raise ExperimentError(f"invalid topology dimensions: {topology.name}")
    if len(topology.group_sizes) != topology.nodes:
        raise ExperimentError(
            f"{topology.name} has {len(topology.group_sizes)} groups, expected "
            f"{topology.nodes}"
        )
    if sum(topology.group_sizes) != 32 or any(
        size <= 0 for size in topology.group_sizes
    ):
        raise ExperimentError(
            f"{topology.name} decoder groups must cover exactly 32 layers"
        )
    if topology.redundancy != topology.num_paths:
        raise ExperimentError(
            f"{topology.name} redundancy/path mismatch: "
            f"{topology.redundancy} != {topology.num_paths}"
        )

    dealer_names = [dealer for dealer, _groups in topology.dealers]
    if len(set(dealer_names)) != len(dealer_names):
        raise ExperimentError(f"{topology.name} has duplicate dealer names")
    group_dealers: dict[int, list[str]] = {
        group: [] for group in range(topology.nodes)
    }
    for dealer, raw_groups in topology.dealers:
        groups = tuple(int(group) for group in raw_groups)
        if not groups or len(set(groups)) != len(groups):
            raise ExperimentError(
                f"{topology.name} dealer {dealer} has invalid groups {groups}"
            )
        if tuple(sorted(groups)) != tuple(range(min(groups), max(groups) + 1)):
            raise ExperimentError(
                f"{topology.name} dealer {dealer} groups are not contiguous"
            )
        for group in groups:
            if group not in group_dealers:
                raise ExperimentError(
                    f"{topology.name} dealer {dealer} has out-of-range group {group}"
                )
            group_dealers[group].append(dealer)
    if any(
        len(dealers) != topology.redundancy
        for dealers in group_dealers.values()
    ):
        raise ExperimentError(
            f"{topology.name} group replica counts do not match redundancy "
            f"{topology.redundancy}"
        )

    placement_by_dealer: dict[str, int] = {}
    for dealer, _process, _thread, device in topology.placements:
        if dealer in placement_by_dealer:
            raise ExperimentError(
                f"{topology.name} has duplicate placement for {dealer}"
            )
        match = re.fullmatch(r"cuda:(\d+)", device)
        if match is None:
            raise ExperimentError(
                f"{topology.name} placement for {dealer} is not CUDA: {device}"
            )
        gpu = int(match.group(1))
        if gpu >= topology.cards:
            raise ExperimentError(
                f"{topology.name} placement for {dealer} exceeds card count: "
                f"{device}"
            )
        placement_by_dealer[dealer] = gpu
    if set(placement_by_dealer) != set(dealer_names):
        raise ExperimentError(
            f"{topology.name} placement/dealer names do not match"
        )
    if topology.cards > 1 and topology.redundancy > 1:
        for group, dealers in group_dealers.items():
            devices = {placement_by_dealer[dealer] for dealer in dealers}
            if len(devices) != topology.redundancy:
                raise ExperimentError(
                    f"{topology.name} group {group} replicas share a physical GPU"
                )


def summarize_run(
    result_root: Path,
    network_runs: list[dict[str, Any]],
    *,
    repeats: int,
    topologies: Sequence[Topology],
    include_persistent_p0: bool,
    capture_metrics: bool,
) -> dict[str, Any]:
    topology_by_name = {topology.name: topology for topology in topologies}
    injections = {
        (
            item["deployment"],
            item["topology"],
            probability,
        ): int(details["injected_failures"])
        for item in network_runs
        for probability, details in item["probabilities"].items()
    }
    recoveries = {
        (
            item["deployment"],
            item["topology"],
            probability,
        ): details["recovery_latency"]
        for item in network_runs
        for probability, details in item["probabilities"].items()
    }
    run_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    deployments = [
        path.name
        for path in result_root.iterdir()
        if path.is_dir() and path.name != "smoke"
    ]
    for deployment in sorted(deployments):
        for topology_name in (topology.name for topology in topologies):
            topology = topology_by_name[topology_name]
            topology_root = result_root / deployment / topology_name
            if not topology_root.exists():
                continue
            for probability in deployment_probabilities(topology, deployment):
                label = probability_label(probability)
                per_probability: list[dict[str, Any]] = []
                for repeat in range(repeats):
                    case_root = topology_root / f"drop-{label}" / f"repeat-{repeat}"
                    result_path = case_root / "result.json"
                    graph_path = case_root / "decode_graph.json"
                    phases_path = case_root / "phases.json"
                    record = json.loads(result_path.read_text(encoding="utf-8"))
                    graph = json.loads(graph_path.read_text(encoding="utf-8"))
                    phases = json.loads(phases_path.read_text(encoding="utf-8"))
                    parameters = phases.get("parameters", {})
                    expected_parameters = {
                        "model_name": MODEL,
                        "max_tokens": FORMAL_MAX_TOKENS,
                        "warmup_tokens": WARMUP_TOKENS,
                        "parallel_seqs": 1,
                        "num_paths": topology.num_paths,
                        "logprobs": LOGPROBS,
                    }
                    actual_parameters = {
                        key: parameters.get(key) for key in expected_parameters
                    }
                    if actual_parameters != expected_parameters:
                        raise ExperimentError(
                            f"unexpected request parameters in {phases_path}: "
                            f"{actual_parameters}, expected {expected_parameters}"
                        )
                    if phases.get("deployment") != deployment:
                        raise ExperimentError(
                            f"unexpected deployment in {phases_path}: "
                            f"{phases.get('deployment')!r}, expected "
                            f"{deployment!r}"
                        )
                    keypoint_coverage = validate_keypoints(
                        phases["keypoints"], topology
                    )
                    graph_data = graph_summary(graph)
                    if not graph_data["decoder_enabled"]:
                        raise ExperimentError(f"decoder graph disabled: {graph_path}")
                    if not graph_data["decoder_captured"]:
                        raise ExperimentError(f"decoder graph not captured: {graph_path}")
                    if graph_data["decoder_replays"] <= 0:
                        raise ExperimentError(f"decoder graph did not replay: {graph_path}")
                    if graph_data["fallbacks"] != 0:
                        raise ExperimentError(
                            f"unexpected graph fallback: {graph_path}"
                        )
                    row = {
                        "deployment": deployment,
                        "topology": topology_name,
                        "probability": probability,
                        "repeat": repeat,
                        "timestamps": len(record),
                        "throughput_128": legacy_throughput(record, 128),
                        "throughput_1024": legacy_throughput(record, 1024),
                        "graph_dealers": graph_data["dealer_count"],
                        "graph_decoder_replays": graph_data["decoder_replays"],
                        "graph_fallback_records": graph_data["fallbacks"],
                        "graph_sampling_fallback_records": graph_data[
                            "fallback_reasons"
                        ].get("sampling", 0),
                        "graph_non_sampling_fallback_records": graph_data[
                            "non_sampling_fallbacks"
                        ],
                        "keypoint_coverage": keypoint_coverage,
                        "result": str(result_path.relative_to(EXP_ROOT)),
                    }
                    run_rows.append(row)
                    per_probability.append(row)
                rates_128 = [row["throughput_128"] for row in per_probability]
                rates_1024 = [row["throughput_1024"] for row in per_probability]
                recovery = recoveries.get((
                    deployment, topology_name, format(probability, ".8g")
                ), {})
                aggregate_rows.append({
                    "deployment": deployment,
                    "topology": topology_name,
                    "probability": probability,
                    "repeat_count": len(per_probability),
                    "throughput_128_mean": statistics.mean(rates_128),
                    "throughput_128_min": min(rates_128),
                    "throughput_128_max": max(rates_128),
                    "throughput_1024_mean": statistics.mean(rates_1024),
                    "throughput_1024_min": min(rates_1024),
                    "throughput_1024_max": max(rates_1024),
                    "injected_failures_total": injections.get((
                        deployment, topology_name, format(probability, ".8g")
                    ), 0),
                    "recovery_count": recovery.get("count", 0),
                    "recovery_latency_mean_seconds": recovery.get(
                        "mean_seconds"
                    ),
                    "recovery_latency_median_seconds": recovery.get(
                        "median_seconds"
                    ),
                    "recovery_latency_min_seconds": recovery.get(
                        "min_seconds"
                    ),
                    "recovery_latency_max_seconds": recovery.get(
                        "max_seconds"
                    ),
                    "recovery_unmatched_starts": recovery.get(
                        "unmatched_starts", 0
                    ),
                    "recovery_unmatched_ends": recovery.get(
                        "unmatched_ends", 0
                    ),
                    # Graph counters are Dealer-lifetime cumulative snapshots;
                    # the final (maximum) value is meaningful, summing repeat
                    # snapshots would double count earlier executions.
                    "graph_decoder_replays_final": max(
                        row["graph_decoder_replays"]
                        for row in per_probability
                    ),
                    "graph_sampling_fallback_records_final": max(
                        row["graph_sampling_fallback_records"]
                        for row in per_probability
                    ),
                    "graph_non_sampling_fallback_records": max(
                        row["graph_non_sampling_fallback_records"]
                        for row in per_probability
                    ),
                })
    return {
        "schema_version": 2,
        "metric": "paper-compatible r/(t_r-t_1)",
        "parameters": experiment_parameters(
            repeats=repeats,
            topologies=topologies,
            include_persistent_p0=include_persistent_p0,
            capture_metrics=capture_metrics,
        ),
        "runs": run_rows,
        "aggregates": aggregate_rows,
        "network_runs": network_runs,
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ExperimentError("cannot write an empty summary")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["keypoint_coverage"] = json.dumps(row["keypoint_coverage"])
            writer.writerow(encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper-matrix DROFELL unreliable-node experiment"
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--port", type=int, default=7255)
    # MERGE: 默认 10 次（论文口径）。8 卡扩展清单的下限是 5 次，
    # 跑那批时显式传 --repeats 5，不要改这里的默认值。
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=(
            "formal repeats per probability; default 10 matches the paper, "
            "the 8-card expansion list allows a minimum of 5"
        ),
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        choices=tuple(topology.name for topology in TOPOLOGIES),
        default=tuple(topology.name for topology in TOPOLOGIES),
        help="Topology subset; defaults to all eight paper topologies.",
    )
    parser.add_argument(
        "--probabilities",
        type=float,
        nargs="+",
        default=PAPER_PROBABILITIES,
        help=(
            "Failure-probability subset from the paper grid; intended for "
            "isolated retry runs after a failed cell."
        ),
    )
    parser.add_argument(
        "--include-persistent-p0",
        action="store_true",
        help="Also run the extra persistent-Agent p=0 observations.",
    )
    parser.add_argument(
        "--capture-metrics",
        action="store_true",
        help=(
            "Write DROFELL JSONL metrics for a dedicated instrumentation run; "
            "do not use this for throughput-table cells."
        ),
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help=(
            "Skip the built-in p=0 protocol smoke. Use only for isolated "
            "replicates after the same topology has already passed smoke."
        ),
    )
    args = parser.parse_args()
    if not args.run_id or "/" in args.run_id or args.run_id in {".", ".."}:
        parser.error("--run-id must be one non-empty path component")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    allowed_probabilities = set(PAPER_PROBABILITIES)
    invalid_probabilities = [
        value for value in args.probabilities
        if value not in allowed_probabilities
    ]
    if invalid_probabilities:
        parser.error(
            "--probabilities contains values outside the shared DROFELL/PETALS "
            f"grid {PAPER_PROBABILITIES}: {invalid_probabilities}"
        )
    args.probabilities = tuple(dict.fromkeys(args.probabilities))
    return args


def main() -> None:
    global REPEATS
    args = parse_args()
    REPEATS = args.repeats
    python = args.python.expanduser()
    if not python.is_absolute():
        python = REPO_ROOT / python
    experiment = Experiment(
        # Do not resolve the venv's python symlink: executing its resolved
        # Conda target directly would drop the venv site-packages.
        python=python,
        run_id=args.run_id,
        port=args.port,
        repeats=args.repeats,
        topologies=tuple(
            replace(topology, probabilities=args.probabilities)
            for topology in TOPOLOGIES
            if topology.name in set(args.topologies)
        ),
        include_persistent_p0=args.include_persistent_p0,
        capture_metrics=args.capture_metrics,
        skip_smoke=args.skip_smoke,
    )
    experiment.initialize()
    print(f"unreliable-node run id: {experiment.run_id}", flush=True)
    print(f"artifacts: {experiment.result_dir}", flush=True)
    experiment.execute()
    print(f"summary: {experiment.result_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except ExperimentError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
