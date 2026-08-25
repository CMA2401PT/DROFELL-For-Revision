#!/usr/bin/env python3
"""Run the paper's single-GPU vLLM upper-bound measurements."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/home/dai/projects/LLM/dflash_related/.venv/bin/python")
DEFAULT_MODEL = Path(
    "/home/dai/projects/LLM/dflash_related/models/Llama-2-7B-GPTQ"
)
TOKEN_POSITIONS = (128, 1024)


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=utc_run_id())
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.runs <= 0 or args.warmup_requests < 0:
        parser.error("--runs must be positive and --warmup-requests non-negative")
    return args


def command_text(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def capture(command: Sequence[str], env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        list(command), cwd=REPO_ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    return completed.stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    # Do not resolve the venv's python symlink: executing the resolved system
    # interpreter loses pyvenv.cfg discovery and therefore the venv packages.
    python = args.python.absolute()
    model = args.model.resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    if not (model / "model.safetensors").is_file():
        raise FileNotFoundError(f"incomplete model directory: {model}")

    run_dir = EXP_ROOT / "results" / args.run_id
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"run directory already exists: {run_dir}")
    raw_dir = run_dir / "raw"
    log_dir = run_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=args.resume)
    log_dir.mkdir(exist_ok=args.resume)

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONUNBUFFERED": "1",
    })
    commands: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for tokens in TOKEN_POSITIONS:
        output_path = raw_dir / f"seq{tokens}.json"
        log_path = log_dir / f"seq{tokens}.log"
        command = (
            "/usr/bin/uclampset", "-m", "1024", "--", str(python),
            "scripts/vllm/decode_benchmark.py",
            "--model", str(model),
            "--num-seqs", "1",
            "--runs", str(args.runs),
            "--warmup-requests", str(args.warmup_requests),
            # Warm the engine with separate requests, but do not skip positions
            # inside a measured trajectory.  This matches DROFELL's timing from
            # the first post-prefill decode token.
            "--warmup-decode-tokens", "0",
            "--decode-tokens", str(tokens),
            "--max-model-len", "1100",
            "--cuda-graph",
            "--output", str(output_path),
        )
        if args.resume and output_path.is_file():
            commands.append({"command": command_text(command), "skipped": True})
        else:
            if output_path.exists() or log_path.exists():
                raise RuntimeError(f"partial case exists for seq={tokens}")
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    list(command), cwd=REPO_ROOT, env=env, text=True,
                    stdout=log, stderr=subprocess.STDOUT, check=False,
                )
            commands.append({
                "command": command_text(command),
                "returncode": completed.returncode,
                "log": str(log_path),
            })
            if completed.returncode != 0:
                tail = log_path.read_text(
                    encoding="utf-8", errors="replace"
                )[-16000:]
                raise RuntimeError(
                    f"vLLM seq={tokens} failed ({completed.returncode}):\n{tail}"
                )

        report = json.loads(output_path.read_text(encoding="utf-8"))
        if report["quantization"] != "gptq_marlin":
            raise RuntimeError(f"unexpected quantization for seq={tokens}")
        if report["execution"] != "graph":
            raise RuntimeError(f"CUDA graph was not active for seq={tokens}")
        case = report["cases"][0]
        rates = [float(run["tok_s"]) for run in case["runs"]]
        if len(rates) != args.runs:
            raise RuntimeError(f"incomplete repeats for seq={tokens}")
        rows.append({
            "token_position": tokens,
            "repeats": len(rates),
            "mean_tok_s": statistics.mean(rates),
            "stdev_tok_s": statistics.stdev(rates) if len(rates) > 1 else 0.0,
            "median_tok_s": statistics.median(rates),
            "min_tok_s": min(rates),
            "max_tok_s": max(rates),
            "raw": str(output_path),
        })

    metadata = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": capture(("git", "rev-parse", "HEAD")),
        "git_status": capture(("git", "status", "--short")),
        "platform": platform.platform(),
        "python": str(python),
        "python_version": capture((str(python), "--version"), env),
        "model": str(model),
        "model_sha256": capture(("sha256sum", str(model / "model.safetensors"))),
        "gpu": capture((
            "nvidia-smi", f"--id={args.gpu}",
            "--query-gpu=name,memory.total,driver_version,pstate,clocks.sm",
            "--format=csv,noheader",
        )),
        "parameters": {
            "token_positions": list(TOKEN_POSITIONS),
            "num_seqs": 1,
            "runs": args.runs,
            "warmup_requests": args.warmup_requests,
            "warmup_decode_tokens": 0,
            "cuda_graph": True,
            "gpu": args.gpu,
            "timing_scope": "post-prefill decode from first decode token",
        },
        "commands": commands,
    }
    write_json(run_dir / "summary.json", {"rows": rows})
    write_json(run_dir / "environment.json", metadata)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"completed: {run_dir}")


if __name__ == "__main__":
    main()
