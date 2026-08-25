#!/usr/bin/env python3
"""Orchestrate the 30-minute-per-condition PEFT convergence comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
from typing import Any, Sequence


EXP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXP_ROOT.parents[1]
DEFAULT_PYTHON = Path("/home/dai/projects/LLM/dflash_related/.venv/bin/python")
ONLINE_RUNNER = REPO_ROOT / "exps/recover-analysis-peft/basic/run.py"


class ExperimentError(RuntimeError):
    pass


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("peft-correctness-%Y%m%dT%H%M%SZ")


def command_text(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dataset_size(path: Path) -> int:
    with path.open("rb") as source:
        return sum(1 for line in source if line.strip())


class Experiment:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # Keep the virtual-environment executable path intact.  Resolving its
        # symlink would invoke the base interpreter without the venv packages.
        self.python = args.python.absolute()
        self.result_root = (EXP_ROOT / "results" / args.run_id).resolve()
        self.log_root = (EXP_ROOT / "logs" / args.run_id).resolve()
        self.manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": args.run_id,
            "status": "planned",
            "started_utc": None,
            "finished_utc": None,
            "parameters": {},
            "commands": [],
        }

    @property
    def minibatches(self) -> int:
        return round(self.args.duration_s * 1000.0 / self.args.submission_interval_ms)

    @property
    def samples(self) -> int:
        return self.minibatches * self.args.minibatch_size

    @property
    def updates(self) -> int:
        return self.minibatches // self.args.grad_accum_steps

    def initialize(self) -> None:
        if not self.python.is_file():
            raise ExperimentError(f"Python interpreter not found: {self.python}")
        if self.result_root.exists() or self.log_root.exists():
            raise ExperimentError(f"refusing to overwrite run {self.args.run_id!r}")
        self.result_root.mkdir(parents=True)
        self.log_root.mkdir(parents=True)
        source = REPO_ROOT / "data/AdvertiseGen/train.json"
        count = dataset_size(source)
        indices = list(range(count))
        random.Random(self.args.seed).shuffle(indices)
        indices = indices[:self.samples]
        write_json(self.result_root / "sample_indices.json", indices)
        self.manifest["started_utc"] = datetime.now(timezone.utc).isoformat()
        self.manifest["parameters"] = {
            "model": "teared-Llama-2-7B-GPTQ",
            "adapters": list(self.args.adapters),
            "conditions": ["reference", "offline", "online-static", "online-recovery"],
            "planned_submission_window_s": self.args.duration_s,
            "minibatches": self.minibatches,
            "samples": self.samples,
            "optimizer_updates": self.updates,
            "minibatch_size": self.args.minibatch_size,
            "grad_accum_steps": self.args.grad_accum_steps,
            "submission_interval_ms": self.args.submission_interval_ms,
            "learning_rate": self.args.learning_rate,
            "optimizer_eps": self.args.optimizer_eps,
            "optimizer": "AdamW",
            "scheduler": None,
            "precision": "fp16",
            "decoder_backend": "exllama_fuse",
            "adapter_dropout": 0.0,
            "ptuning_prefix_len": self.args.ptuning_prefix_len,
            "lora_r_q": -1,
            "lora_r_k": 0,
            "lora_r_v": -1,
            "lora_len_limit": self.args.lora_len_limit,
            "max_query_tokens": self.args.max_query_tokens,
            "max_response_tokens": self.args.max_response_tokens,
            "force_query_tokens": 0,
            "force_response_tokens": 0,
            "max_seq_len": self.args.max_seq_len,
            "checkpoint_period": self.args.checkpoint_period,
            "recovery_failure_times_s": list(self.args.failure_times_s),
            "seed": self.args.seed,
            "sample_index_file": "sample_indices.json",
            "initialization": "independent deterministic reinit per condition",
        }
        self.flush()

    def flush(self) -> None:
        write_json(self.result_root / "manifest.json", self.manifest)

    def common_train_args(self, adapter: str, output: Path) -> list[str]:
        return [
            "--adapter", adapter,
            "--output", str(output),
            "--sample-indices", str(self.result_root / "sample_indices.json"),
            "--seed", str(self.args.seed),
            "--minibatches", str(self.minibatches),
            "--minibatch-size", str(self.args.minibatch_size),
            "--grad-accum-steps", str(self.args.grad_accum_steps),
            "--submission-interval-ms", str(self.args.submission_interval_ms),
            "--max-query-tokens", str(self.args.max_query_tokens),
            "--max-response-tokens", str(self.args.max_response_tokens),
            "--ptuning-prefix-len", str(self.args.ptuning_prefix_len),
            "--lora-len-limit", str(self.args.lora_len_limit),
            "--learning-rate", str(self.args.learning_rate),
            "--optimizer-eps", str(self.args.optimizer_eps),
            "--async-backend", "golang", "--busy-wait",
        ]

    def commands(self) -> list[tuple[str, list[str], Path]]:
        commands: list[tuple[str, list[str], Path]] = []
        for adapter_index, adapter in enumerate(self.args.adapters):
            adapter_root = self.result_root / adapter
            reference_output = adapter_root / "reference" / "repeat-0" / "raw"
            offline_output = adapter_root / "offline" / "repeat-0" / "raw"
            commands.append((
                f"{adapter}:reference",
                [str(self.python), str(EXP_ROOT / "reference_train.py"),
                 *self.common_train_args(adapter, reference_output)],
                self.log_root / adapter / "reference.log",
            ))
            commands.append((
                f"{adapter}:offline",
                [str(self.python), str(EXP_ROOT / "offline_train.py"),
                 *self.common_train_args(adapter, offline_output),
                 "--groups", "4", "--checkpoint-period", str(self.args.checkpoint_period),
                 "--max-seq-len", str(self.args.max_seq_len)],
                self.log_root / adapter / "offline.log",
            ))
            online_root = adapter_root / "online"
            online_log_root = self.log_root / adapter / "online"
            port = self.args.port + adapter_index
            online_command = [
                str(self.python), str(ONLINE_RUNNER),
                "--python", str(self.python),
                "--run-id", f"{self.args.run_id}-{adapter}-online",
                "--experiment-profile", "peft-correctness",
                "--result-root", str(online_root),
                "--log-root", str(online_log_root),
                "--port", str(port), "--repeats", "1",
                "--adapter", adapter,
                "--learning-rate", str(self.args.learning_rate),
                "--optimizer-eps", str(self.args.optimizer_eps),
                "--prefix-len", str(self.args.ptuning_prefix_len),
                "--lora-len-limit", str(self.args.lora_len_limit),
                "--samples", str(self.samples),
                "--minibatch-size", str(self.args.minibatch_size),
                "--grad-accum-steps", str(self.args.grad_accum_steps),
                "--submission-interval-ms", str(self.args.submission_interval_ms),
                "--dataset-query-tokens", "0",
                "--dataset-response-tokens", "0",
                "--max-seq-len", str(self.args.max_seq_len),
                "--ckpt-period", str(self.args.checkpoint_period),
                "--seed", str(self.args.seed),
                "--drop-versions",
                "--drop-times-s", *(str(value) for value in self.args.failure_times_s),
                "--recovery-latencies-ms", "0",
                "--case", "normal", "--case", "recover",
            ]
            commands.append((f"{adapter}:online-static+recovery", online_command,
                             self.log_root / adapter / "online-runner.log"))
        return commands

    def run(self) -> None:
        self.initialize()
        commands = self.commands()
        if self.args.dry_run:
            self.manifest["status"] = "dry-run"
            self.manifest["commands"] = [
                {"purpose": purpose, "command": command_text(command), "log": str(log)}
                for purpose, command, log in commands
            ]
            self.flush()
            for purpose, command, _ in commands:
                print(f"[{purpose}] {command_text(command)}")
            return

        self.manifest["status"] = "running"
        self.flush()
        env = os.environ.copy()
        env.update({"PYTHONPATH": str(REPO_ROOT), "PYTHONUNBUFFERED": "1"})
        for purpose, command, log in commands:
            log.parent.mkdir(parents=True, exist_ok=True)
            record = {"purpose": purpose, "command": command_text(command),
                      "log": str(log.relative_to(EXP_ROOT)), "returncode": None}
            self.manifest["commands"].append(record)
            self.flush()
            print(f"starting {purpose}; log={log}", flush=True)
            try:
                with log.open("wb") as output:
                    completed = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=self.args.command_timeout_s,
                    )
            except Exception:
                self.manifest["status"] = "failed"
                self.flush()
                raise
            record["returncode"] = completed.returncode
            self.flush()
            if completed.returncode != 0:
                self.manifest["status"] = "failed"
                self.flush()
                tail = log.read_text(encoding="utf-8", errors="replace")[-12000:]
                raise ExperimentError(f"{purpose} failed ({completed.returncode}):\n{tail}")
            print(f"completed {purpose}", flush=True)

        analysis_command = [str(self.python), str(EXP_ROOT / "analyze.py"), str(self.result_root)]
        analysis_log = self.log_root / "analysis.log"
        with analysis_log.open("wb") as output:
            completed = subprocess.run(
                analysis_command, cwd=REPO_ROOT, env=env,
                stdout=output, stderr=subprocess.STDOUT, check=False,
            )
        self.manifest["commands"].append({
            "purpose": "analyze", "command": command_text(analysis_command),
            "log": str(analysis_log.relative_to(EXP_ROOT)),
            "returncode": completed.returncode,
        })
        self.manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self.manifest["status"] = "completed" if completed.returncode == 0 else "analysis-failed"
        self.flush()
        if completed.returncode != 0:
            raise ExperimentError(analysis_log.read_text(encoding="utf-8", errors="replace")[-12000:])
        print(f"completed: {self.result_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--adapter", action="append", choices=("ptuning", "lora"), dest="adapters")
    parser.add_argument("--duration-s", type=float, default=1800.0)
    parser.add_argument("--submission-interval-ms", type=float, default=500.0)
    parser.add_argument("--minibatch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--optimizer-eps", type=float, default=1e-4)
    parser.add_argument("--max-query-tokens", type=int, default=-1)
    parser.add_argument("--max-response-tokens", type=int, default=-1)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--ptuning-prefix-len", type=int, default=128)
    parser.add_argument("--lora-len-limit", type=int, default=128)
    parser.add_argument("--checkpoint-period", type=int, default=10)
    parser.add_argument(
        "--failure-times-s", nargs="+", type=float,
        default=(450.0, 900.0, 1350.0),
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--port", type=int, default=7360)
    parser.add_argument("--command-timeout-s", type=float, default=10800.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.adapters = tuple(args.adapters or ("ptuning", "lora"))
    args.failure_times_s = tuple(args.failure_times_s)
    if min(args.duration_s, args.submission_interval_ms, args.minibatch_size,
           args.grad_accum_steps, args.learning_rate, args.optimizer_eps,
           args.max_seq_len) <= 0:
        parser.error("duration, batch, and optimizer values must be positive")
    if args.max_query_tokens != -1 or args.max_response_tokens != -1:
        parser.error("correctness experiment requires untruncated query/response tokens (-1/-1)")
    minibatches = round(args.duration_s * 1000.0 / args.submission_interval_ms)
    if minibatches % args.grad_accum_steps != 0:
        parser.error("planned minibatches must be divisible by grad-accum-steps")
    if args.failure_times_s[-1] >= args.duration_s:
        parser.error("failure times must fall inside the submission window")
    if (tuple(sorted(args.failure_times_s)) != args.failure_times_s
            or len(set(args.failure_times_s)) != len(args.failure_times_s)
            or any(value <= 0 for value in args.failure_times_s)):
        parser.error("failure times must be positive, unique, and increasing")
    if not 1 <= args.port < 65535 - len(args.adapters):
        parser.error("port range is invalid for the selected adapters")
    if len(set(args.adapters)) != len(args.adapters):
        parser.error("adapter values must be unique")
    return args


if __name__ == "__main__":
    Experiment(parse_args()).run()
