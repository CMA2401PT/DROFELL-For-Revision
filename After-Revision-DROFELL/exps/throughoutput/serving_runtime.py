#!/usr/bin/env python3
"""Runtime primitives for the split serving entrypoint.

This module intentionally contains only network-spec generation and persistent
Orbit/dealer lifecycle management; benchmark execution belongs to
``run_benchmark.py``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from typing import Sequence

from session import (
    AGENT,
    BRIDGE,
    EXP_ROOT,
    model_spec,
    NSPREFIX,
    PORT,
    REPO_ROOT,
    SUBNET,
)


NET_CLUSTER = REPO_ROOT / "scripts" / "online" / "benchmark" / "net_cluster"
TORCH_EXTENSIONS_DIR = Path("/root/.cache/torch_extensions")


class ExperimentError(RuntimeError):
    """A serving setup or lifecycle command failed."""


def command_text(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ServingRuntime:
    """Generate one network spec and keep its Orbit/dealers alive."""

    def __init__(
        self,
        *,
        python: Path,
        run_id: str,
        model_size: str,
        cards: int,
        workload: str,
        deployment: str,
        bandwidth_mbit: int,
        target_rtt_ms: float,
        no_netns: bool = False,
        train_backend: str = "exllama_fuse",
    ) -> None:
        if not python.is_file():
            raise ExperimentError(f"Python does not exist: {python}")
        if workload not in {"inference", "finetuning"}:
            raise ExperimentError(f"unsupported workload: {workload}")
        if cards not in {2, 4, 8}:
            raise ExperimentError(f"unsupported card count: {cards}")
        self.python = python
        self.run_id = run_id
        self.model_size = model_size
        self.model = str(model_spec(model_size)["name"])
        self.model_decoder_layers = int(model_spec(model_size)["decoder_layers"])
        self.cards = cards
        self.workload = workload
        self.deployment = deployment
        self.train_backend = train_backend
        self.no_netns = no_netns
        self.bandwidth = f"{bandwidth_mbit}mbit"
        self.one_way_latency = f"{target_rtt_ms / 2:g}ms"
        self.result_root = EXP_ROOT / "results" / run_id
        self.log_root = EXP_ROOT / "logs" / run_id
        self.processes: list[subprocess.Popen[bytes]] = []
        self.base_env = os.environ.copy()
        allocator_config = self.base_env.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:" not in allocator_config:
            allocator_config = ",".join(filter(None, (
                allocator_config, "expandable_segments:True"
            )))
        self.base_env.update({
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONUNBUFFERED": "1",
            "TORCH_EXTENSIONS_DIR": str(TORCH_EXTENSIONS_DIR),
            "PYTORCH_CUDA_ALLOC_CONF": allocator_config,
            "BRIDGE": BRIDGE,
            "SUBNET": SUBNET,
            "NSPREFIX": NSPREFIX,
            "LAT": self.one_way_latency,
            "BW": self.bandwidth,
            "REUSE_RECONSTRUCT_CACHE_WHEN_GRAD": "1",
        })

    @staticmethod
    def dealer_name(index: int) -> str:
        return f"dealer-{chr(ord('a') + index)}"

    @staticmethod
    def card_root(root: Path, cards: int) -> Path:
        return root / f"cards-{cards}"

    def node_host(self, node_id: int) -> str:
        return "127.0.0.1" if self.no_netns else f"{SUBNET}.{node_id}"

    @property
    def orbit_addr(self) -> str:
        return f"{self.node_host(100)}:{PORT}"

    def infer_network(self) -> Path:
        suffix = "" if self.cards == 2 else f"-{self.cards}gpu"
        return EXP_ROOT / "networks" / f"llama2-{self.model_size}{suffix}-graph.json"

    def train_network(self) -> Path:
        suffix = "" if self.cards == 2 else f"-{self.cards}gpu"
        return EXP_ROOT / "networks" / f"llama2-{self.model_size}{suffix}-train-{self.train_backend}.json"

    def network(self) -> Path:
        return self.infer_network() if self.workload == "inference" else self.train_network()

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
        if check and completed.returncode != 0:
            output = completed.stdout or (
                log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
                if log_path else ""
            )
            raise ExperimentError(
                f"command failed ({completed.returncode}): "
                f"{command_text(command)}\n{output}"
            )
        return completed

    def uclamped_python(self, *args: str) -> tuple[str, ...]:
        return ("/usr/bin/uclampset", "-m", "1024", "--", str(self.python), *args)

    def cluster_command(self, *args: str) -> tuple[str, ...]:
        return (str(NET_CLUSTER), *args)

    def setup_network_transport(self) -> None:
        if self.no_netns:
            return
        self.run(self.cluster_command("down"), check=False)
        self.run(self.cluster_command("up"))

    def snapshot_network_spec(self, source: Path) -> None:
        target = self.result_root / "network_specs" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ExperimentError(f"archived network spec differs: {target}")
        if not target.exists():
            target.write_bytes(source.read_bytes())

    def generate_network(self) -> Path:
        cards = self.cards
        if self.model_decoder_layers % cards:
            raise ExperimentError(
                f"{self.model_decoder_layers} decoder layers are not divisible by {cards} cards"
            )
        layers_per_card = self.model_decoder_layers // cards
        common = [
            "--model", self.model,
            "--decoder-group-sizes",
            *(str(layers_per_card) for _ in range(cards)),
        ]
        for index in range(cards):
            common += ["--dealer", f"{self.dealer_name(index)}={index}"]
        common += [
            "--max-concurrency", "16",
            "--memory-policy", "percentage",
            "--major-percentage", "0.5",
        ]
        output = self.network()
        if self.workload == "inference":
            name = f"llama2-{self.model_size}-throughput-{cards}gpu-graph"
            args = [
                "--name", name, *common,
                "--decoder-backend", "marlin",
                "--max-seq-len", "4096", "--cuda-graph",
                "--unmanaged-reserved-mib", "10",
                "--minor-min-mib", "0", "--minor-max-mib", "262144",
                "--operate-reserved-mib", "100", "--graph-budget-mib", "100",
            ]
        else:
            name = f"llama2-{self.model_size}-throughput-{cards}gpu-train-{self.train_backend}"
            args = [
                "--name", name, *common,
                "--decoder-backend", self.train_backend,
                "--max-seq-len", "256",
                "--unmanaged-reserved-mib", "500",
                "--minor-min-mib", "512", "--minor-max-mib", "1024",
                "--operate-reserved-mib", "3072",
            ]
        self.run(self.uclamped_python(
            "-m", "scripts.online.benchmark.generate_network", *args,
            "--output", str(output),
        ))
        self.snapshot_network_spec(output)
        return output

    def start_in_namespace(
        self, node_id: int, command: Sequence[str], log_path: Path,
    ) -> subprocess.Popen[bytes]:
        env = self.base_env.copy()
        if self.no_netns:
            cmd = tuple(command)
        else:
            env["ID"] = str(node_id)
            cmd = self.cluster_command("run", *command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("wb")
        process = subprocess.Popen(
            list(cmd), cwd=REPO_ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        log.close()
        self.processes.append(process)
        return process

    @staticmethod
    def wait_for_text(process: subprocess.Popen[bytes], log_path: Path, needle: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                raise ExperimentError(f"process exited before {needle!r}:\n{text[-12000:]}")
            if needle in log_path.read_text(encoding="utf-8", errors="replace"):
                return
            time.sleep(0.5)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        raise ExperimentError(f"timeout waiting for {needle!r}:\n{text[-12000:]}")

    def launch_network(self, network: Path) -> None:
        self.setup_network_transport()
        phase_logs = self.card_root(self.log_root, self.cards) / self.workload / self.deployment
        orbit_log = phase_logs / "orbit.log"
        orbit = self.start_in_namespace(100, self.uclamped_python(
            "-m", "scripts.online.benchmark.launch_network", "orbit",
            "--network", str(network), "--port", str(PORT),
            "--advertise-host", self.node_host(100),
            "--async-backend", "golang", "--busy-wait",
        ), orbit_log)
        self.wait_for_text(orbit, orbit_log, "network coordinator", 120)
        dealers: list[tuple[subprocess.Popen[bytes], Path]] = []
        for index in range(self.cards):
            dealer_name = self.dealer_name(index)
            args = [
                "-m", "scripts.online.benchmark.launch_network", "satellite",
                "--network", str(network), "--orbit", self.orbit_addr,
                "--host-id", f"{self.workload}-{self.cards}gpu-{self.deployment}-gpu-{index}",
                "--advertise-host", self.node_host(index + 1),
                "--async-backend", "golang", "--busy-wait",
                "--place", dealer_name, "0", "0", f"cuda:{index}",
            ]
            if self.deployment == "persistent-agent" and index == 0:
                args += ["--agent", AGENT, dealer_name]
            log_path = phase_logs / f"{dealer_name}.log"
            dealer = self.start_in_namespace(index + 1, self.uclamped_python(*args), log_path)
            dealers.append((dealer, log_path))
        for dealer, log_path in dealers:
            self.wait_for_text(dealer, log_path, "load done, announcing", 900)

    def stop_network(self) -> None:
        for process in reversed(self.processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 20
        for process in reversed(self.processes):
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        self.processes.clear()
        if not self.no_netns:
            self.run(self.cluster_command("down"), check=False)
