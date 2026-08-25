#!/usr/bin/env python3
"""Run independent PEFT optimizer/training determinism probes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any


EXP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXP_ROOT.parents[2]
DEFAULT_PYTHON = Path("/home/dai/projects/LLM/dflash_related/.venv/bin/python")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("determinism-lr1e-4-q64-r64-%Y%m%dT%H%M%SZ"))
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--minibatches", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--optimizer-eps", type=float, default=1e-4)
    parser.add_argument("--query-tokens", type=int, default=64)
    parser.add_argument("--response-tokens", type=int, default=64)
    parser.add_argument("--prefix-len", type=int, default=32)
    parser.add_argument(
        "--decoder-backend", choices=("exllama_fuse", "marlin"),
        default="exllama_fuse",
    )
    parser.add_argument(
        "--attention-backend", choices=("flash", "torch-math"),
        default="flash",
    )
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--skip-gpu1", action="store_true")
    args = parser.parse_args()
    result_root = EXP_ROOT / "results" / args.run_id
    log_root = EXP_ROOT / "logs" / args.run_id
    if result_root.exists() or log_root.exists():
        parser.error(f"refusing to overwrite {args.run_id}")
    result_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    parameters = {
        "learning_rate": args.learning_rate,
        "optimizer_eps": args.optimizer_eps,
        "query_tokens": args.query_tokens,
        "response_tokens": args.response_tokens,
        "prefix_len": args.prefix_len,
        "decoder_backend": args.decoder_backend,
        "attention_backend": args.attention_backend,
        "minibatches": args.minibatches,
        "grad_accum_steps": args.grad_accum_steps,
        "seed": args.seed,
        "fault_injection": False,
        "online_network": False,
        "deterministic_algorithms": args.deterministic_algorithms,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "parameters": parameters,
        "runs": [],
    }
    write_json(result_root / "manifest.json", manifest)
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": str(args.seed),
    })

    def execute(name: str, script: str, device: str) -> None:
        relative = Path(name)
        output = result_root / relative
        command = [
            str(args.python), str(EXP_ROOT / script),
            "--output", str(output), "--device", device,
            "--seed", str(args.seed),
            "--learning-rate", str(args.learning_rate),
            "--optimizer-eps", str(args.optimizer_eps),
        ]
        if script == "train_probe.py":
            command.extend([
                "--minibatches", str(args.minibatches),
                "--grad-accum-steps", str(args.grad_accum_steps),
                "--query-tokens", str(args.query_tokens),
                "--response-tokens", str(args.response_tokens),
                "--prefix-len", str(args.prefix_len),
                "--decoder-backend", args.decoder_backend,
                "--attention-backend", args.attention_backend,
            ])
            if args.deterministic_algorithms:
                command.append("--deterministic-algorithms")
        log = log_root / f"{name}.log"
        record = {
            "name": name,
            "relative_path": str(relative),
            "command": shlex.join(command),
            "log": str(log.relative_to(EXP_ROOT)),
            "returncode": None,
        }
        manifest["runs"].append(record)
        write_json(result_root / "manifest.json", manifest)
        with log.open("wb") as destination:
            completed = subprocess.run(
                command, cwd=REPO_ROOT, env=env,
                stdout=destination, stderr=subprocess.STDOUT,
                check=False, timeout=1800,
            )
        record["returncode"] = completed.returncode
        write_json(result_root / "manifest.json", manifest)
        if completed.returncode:
            tail = log.read_text(encoding="utf-8", errors="replace")[-12000:]
            raise RuntimeError(f"{name} failed ({completed.returncode}):\n{tail}")

    for repeat in range(2):
        execute(f"optimizer-gpu0-r{repeat}", "optimizer_probe.py", "cuda:0")
    if not args.skip_gpu1:
        execute("optimizer-gpu1-r0", "optimizer_probe.py", "cuda:1")
    for repeat in range(2):
        execute(f"train-gpu0-r{repeat}", "train_probe.py", "cuda:0")
    if not args.skip_gpu1:
        execute("train-gpu1-r0", "train_probe.py", "cuda:1")

    analysis_command = [str(args.python), str(EXP_ROOT / "analyze.py"), str(result_root)]
    analysis_log = log_root / "analysis.log"
    with analysis_log.open("wb") as destination:
        completed = subprocess.run(
            analysis_command, cwd=REPO_ROOT, env=env,
            stdout=destination, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode:
        raise RuntimeError(analysis_log.read_text(encoding="utf-8", errors="replace")[-12000:])
    print(f"completed: {result_root}")


if __name__ == "__main__":
    main()
