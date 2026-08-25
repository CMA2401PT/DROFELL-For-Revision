"""Run the short decoder time/memory trade-off experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import time
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/home/dai/projects/LLM/dflash_related/.venv/bin/python")
BACKENDS = ("exllama_fuse", "marlin")
STRATEGIES = ("each_decoder", "first.2pass", "first.grad", "grad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-reconstruct-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def capture(command: Sequence[str]) -> str:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout.strip()


def command_text(command: Sequence[str]) -> str:
    return shlex.join(str(value) for value in command)


def main() -> None:
    args = parse_args()
    python = args.python.absolute()
    if not python.is_file():
        raise FileNotFoundError(python)
    run_dir = EXPERIMENT_DIR / "results" / args.run_id
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"run directory already exists: {run_dir}")
    raw_dir = run_dir / "raw"
    log_dir = run_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=args.resume)
    log_dir.mkdir(exist_ok=args.resume)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment["PYTHONUNBUFFERED"] = "1"
    extension_dir = Path("/tmp") / f"drofell-torch-extensions-{os.getuid()}"
    extension_dir.mkdir(parents=True, exist_ok=True)
    environment["TORCH_EXTENSIONS_DIR"] = str(extension_dir)

    commands: list[str] = []
    for backend in args.backends:
        for strategy in args.strategies:
            case_dir = raw_dir / backend / strategy
            case_dir.mkdir(parents=True, exist_ok=args.resume)
            log_file = log_dir / f"{backend}-{strategy}.log"
            output_file = case_dir / "record.pkl"
            sidecar_file = case_dir / "record.json"
            command = [
                "/usr/bin/uclampset",
                "-m",
                "1024",
                "--",
                str(python),
                "scripts/module/decoder/test.py",
                "--output",
                str(output_file),
                "--backend",
                backend,
                "--strategy",
                strategy,
                "--layers",
                "8",
                "--batch-size",
                "4",
                "--seq-len",
                "512",
                "--prompt-seq-len",
                "256",
                "--warmup",
                str(args.warmup),
                "--repeats",
                str(args.repeats),
                "--seed",
                str(args.seed),
                "--swap",
                (
                    "--reuse-reconstruct-cache"
                    if args.reuse_reconstruct_cache
                    else "--no-reuse-reconstruct-cache"
                ),
            ]
            commands.append(command_text(command))
            if args.resume and output_file.is_file() and sidecar_file.is_file():
                print(f"skipping completed {backend}/{strategy}", flush=True)
                continue
            if output_file.exists() or sidecar_file.exists():
                raise RuntimeError(
                    f"partial case output exists for {backend}/{strategy}: "
                    f"{case_dir}"
                )
            print(f"running {backend}/{strategy}", flush=True)
            with log_file.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"case {backend}/{strategy} failed with code "
                    f"{completed.returncode}; see {log_file}"
                )

    analyze_command = [
        str(python),
        "exps/time_memory_trade/analyze.py",
        "--run-dir",
        str(run_dir),
    ]
    commands.append(command_text(analyze_command))
    subprocess.run(
        analyze_command, cwd=REPO_ROOT, env=environment, check=True
    )

    metadata = {
        "run_id": args.run_id,
        "created_at_unix": time.time(),
        "repo_root": str(REPO_ROOT),
        "git_commit": capture(("git", "rev-parse", "HEAD")),
        "git_status": capture(("git", "status", "--short")),
        "python_executable": str(python),
        "python_version": capture((str(python), "--version")),
        "platform": platform.platform(),
        "gpu": capture(
            (
                "nvidia-smi",
                f"--id={args.gpu}",
                "--query-gpu=name,memory.total,driver_version,pstate,clocks.sm",
                "--format=csv,noheader",
            )
        ),
        "torch_extensions_dir": str(extension_dir),
        "uclampset_version": capture(("uclampset", "--version")),
        "parameters": {
            "backends": args.backends,
            "strategies": args.strategies,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "gpu": args.gpu,
            "layers": 8,
            "batch_size": 4,
            "seq_len": 512,
            "prompt_seq_len": 256,
            "dtype": "float16",
            "swap": True,
            "user_model_checkpoint": "seed",
            "reuse_reconstruct_cache_when_grad": args.reuse_reconstruct_cache,
            "resumed": args.resume,
        },
        "commands": commands,
    }
    (run_dir / "environment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"completed: {run_dir}")


if __name__ == "__main__":
    main()
