import os
import shutil
from typing import Dict, Set
import torch
from safetensors import safe_open  # type:ignore
from safetensors.torch import save_file as save_torch_safetensors  # type:ignore

orig_model_dir = './Llama-2-7B-GPTQ'
target_dir = os.path.join(os.path.dirname(__file__), "convert_output")


def convert_weights(orig_weight_file: str, target_dir: str):
    assert os.path.exists(orig_weight_file), orig_weight_file
    os.makedirs(target_dir, exist_ok=True)

    fp = safe_open(orig_weight_file, framework="pt")
    all_keys: Set[str] = set([k for k in fp.keys()])

    encoders_prefix = "model.layers."
    non_encoders = [k for k in all_keys if not k.startswith(encoders_prefix)]
    assert ['lm_head.weight', 'model.embed_tokens.weight', 'model.norm.weight'] \
        == sorted(non_encoders)

    def save_part(part_name: str, data: Dict[str, torch.Tensor]):
        actual_data: Dict[str, torch.Tensor] = {}
        print(f"save to: {part_name}")
        for k, v in data.items():
            print(f"\t- {k}")
            actual_data[k] = v.clone()
        save_torch_safetensors(
            actual_data,
            os.path.join(target_dir, f'{part_name}.safetensors')
        )

    save_part("lm_head", {
        'weight': fp.get_tensor("lm_head.weight")
    })
    save_part("embed_tokens", {
        'weight': fp.get_tensor("model.embed_tokens.weight")
    })
    save_part("norm", {
        'weight': fp.get_tensor("model.norm.weight")
    })

    encoders: Dict[int, Dict[str, str]] = {}

    for orig_k in all_keys:
        if not orig_k.startswith(encoders_prefix):
            continue
        idx_with_obj_name = orig_k.removeprefix(encoders_prefix)
        idx = int(idx_with_obj_name.split(".", 1)[0])
        part_name = idx_with_obj_name.split(".", 1)[1]
        if not idx in encoders:
            encoders[idx] = {}
        encoders[idx][part_name] = orig_k

    for encoder_i, data in encoders.items():
        save_part(f'layer_{encoder_i}', {
            part_name: fp.get_tensor(orig_k) for part_name, orig_k in data.items()
        })


convert_weights(
    orig_weight_file=os.path.join(orig_model_dir, 'model.safetensors'),
    target_dir=os.path.join(target_dir, 'split_model')
)


def copy_tokenizer(orig_model_dir: str, target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    related_files = [
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
    ]
    for related_file in related_files:
        shutil.copy(
            os.path.join(orig_model_dir, related_file),
            os.path.join(target_dir, related_file)
        )


copy_tokenizer(
    orig_model_dir=orig_model_dir,
    target_dir=os.path.join(target_dir, 'tokenizer')
)

shutil.copy(
    os.path.join(orig_model_dir, "config.json"),
    os.path.join(target_dir, "config.json")
)
