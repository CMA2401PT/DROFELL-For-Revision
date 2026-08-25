"""Extract individual GPTQ projections for module-level benchmarks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open  # type: ignore
from safetensors.torch import save_file as save_safetensors  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "split_qllm_models" / "teared-Llama-2-7B-GPTQ"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "qlinear_data"
PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selected projections from one split GPTQ decoder."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--projections",
        nargs="+",
        choices=PROJECTIONS,
        default=["q_proj"],
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace existing config/projection files.",
    )
    return parser.parse_args()


def copy_file(source: Path, target: Path, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {target}; pass --overwrite")
    shutil.copy2(source, target)


def extract(
    model_dir: Path,
    layer: int,
    output_dir: Path,
    projections: list[str],
    overwrite: bool,
) -> dict[str, object]:
    model_dir = model_dir.resolve()
    output_dir = output_dir.resolve()
    config_file = model_dir / "config.json"
    layer_file = model_dir / "split_model" / f"layer_{layer}.safetensors"
    if not config_file.is_file():
        raise FileNotFoundError(config_file)
    if not layer_file.is_file():
        raise FileNotFoundError(layer_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / "config.json", output_dir / "manifest.json"] + [
        output_dir / f"{projection}.safetensors" for projection in projections
    ]
    if not overwrite:
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite existing files; pass --overwrite: "
                + ", ".join(existing)
            )

    copy_file(config_file, output_dir / "config.json", overwrite=True)
    extracted: Dict[str, Dict[str, torch.Tensor]] = {
        projection: {} for projection in projections
    }
    with safe_open(layer_file, framework="pt") as weights:
        for key in weights.keys():
            parts = key.split(".")
            if len(parts) != 3:
                continue
            _, projection, quant_part = parts
            if projection in extracted:
                extracted[projection][quant_part] = weights.get_tensor(key)

    required_parts = {"qweight", "qzeros", "scales", "g_idx"}
    written: list[dict[str, object]] = []
    for projection in projections:
        tensors = extracted[projection]
        missing = sorted(required_parts - tensors.keys())
        if missing:
            raise ValueError(
                f"{layer_file} projection {projection} misses {missing}"
            )
        target = output_dir / f"{projection}.safetensors"
        save_safetensors(tensors, target)
        written.append(
            {
                "projection": projection,
                "path": str(target),
                "bytes": target.stat().st_size,
                "keys": sorted(tensors),
            }
        )

    manifest: dict[str, object] = {
        "model_dir": str(model_dir),
        "source_layer": layer,
        "source_file": str(layer_file),
        "config_file": str(output_dir / "config.json"),
        "projections": written,
    }
    manifest_file = output_dir / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = extract(
        model_dir=args.model_dir,
        layer=args.layer,
        output_dir=args.output_dir,
        projections=args.projections,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
