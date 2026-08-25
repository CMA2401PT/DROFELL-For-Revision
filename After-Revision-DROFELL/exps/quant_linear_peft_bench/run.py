"""Run isolated short validations for quantized-linear PEFT backends."""

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
DEFAULT_MODEL = REPO_ROOT / "split_qllm_models" / "teared-Llama-2-7B-GPTQ"
DEFAULT_METHODS = (
    "our_no_cache",
    "our_cache",
    "marlin",
    "marlin_dense_no_cache",
    "marlin_dense_cache",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--projection", default="q_proj")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument(
        "--seq-lens", nargs="+", type=int, default=[1, 16, 128, 512, 2048]
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def capture(command: Sequence[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def command_text(command: Sequence[str]) -> str:
    return shlex.join(str(value) for value in command)


def main() -> None:
    args = parse_args()
    # Keep the venv entry point itself. Resolving its symlink to the base Conda
    # interpreter drops the venv site-packages (including torch in this setup).
    python = args.python.absolute()
    if not python.is_file():
        raise FileNotFoundError(python)
    run_dir = EXPERIMENT_DIR / "results" / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    raw_dir = run_dir / "raw"
    log_dir = run_dir / "logs"
    input_dir = run_dir / "inputs"
    raw_dir.mkdir(parents=True)
    log_dir.mkdir()

    commands: list[str] = []
    extract_command = [
        str(python),
        "scripts/converts/qlinear.py",
        "--model-dir",
        str(args.model_dir),
        "--layer",
        "0",
        "--output-dir",
        str(input_dir),
        "--projections",
        args.projection,
    ]
    commands.append(command_text(extract_command))
    subprocess.run(extract_command, cwd=REPO_ROOT, check=True)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    extension_dir = Path("/tmp") / f"drofell-torch-extensions-{os.getuid()}"
    extension_dir.mkdir(parents=True, exist_ok=True)
    environment["TORCH_EXTENSIONS_DIR"] = str(extension_dir)

    for method in args.methods:
        output_file = raw_dir / f"{method}.json"
        benchmark_command = [
            "/usr/bin/uclampset",
            "-m",
            "1024",
            "--",
            str(python),
            "scripts/module/qlinear/test.py",
            "--input-dir",
            str(input_dir),
            "--output",
            str(output_file),
            "--method",
            method,
            "--projection",
            args.projection,
            "--seq-lens",
            *(str(value) for value in args.seq_lens),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
        ]
        commands.append(command_text(benchmark_command))
        print(f"running {method}", flush=True)
        with (log_dir / f"{method}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                benchmark_command,
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{method} failed with code {completed.returncode}; "
                f"see {log_dir / f'{method}.log'}"
            )

    analyze_command = [
        str(python),
        "exps/quant_linear_peft_bench/analyze.py",
        "--run-dir",
        str(run_dir),
    ]
    commands.append(command_text(analyze_command))
    subprocess.run(analyze_command, cwd=REPO_ROOT, check=True)

    metadata = {
        "run_id": args.run_id,
        "created_at_unix": time.time(),
        "repo_root": str(REPO_ROOT),
        "git_commit": capture(("git", "rev-parse", "HEAD")),
        "git_status": capture(("git", "status", "--short")),
        "python_executable": str(python),
        "python_version": capture((str(python), "--version")),
        "platform": platform.platform(),
        "nvidia_smi": capture(("nvidia-smi", "-q", "-i", args.gpu)),
        "uclampset_version": capture(("uclampset", "--version")),
        "torch_extensions_dir": str(extension_dir),
        "parameters": {
            "model_dir": str(args.model_dir.resolve()),
            "source_layer": 0,
            "projection": args.projection,
            "methods": args.methods,
            "seq_lens": args.seq_lens,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "dtype": "float16",
            "gpu": args.gpu,
        },
        "commands": commands,
    }
    (run_dir / "environment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"completed: {run_dir}")


if __name__ == "__main__":
    main()
