import os
import shutil
from typing import Dict
from safetensors import safe_open  # type:ignore
from safetensors.torch import save_file as save_torch_safetensors  # type:ignore
import torch


base_model = "./split_qllm_models/teared-Llama-2-13B-GPTQ"
output_dir = os.path.join(os.path.dirname(__file__), "qlinear_data")

os.makedirs(output_dir, exist_ok=True)
config_file = os.path.join(base_model, "config.json")
encoder_0_file = os.path.join(
    base_model,
    "split_model",
    "layer_0.safetensors"
)

shutil.copy(config_file, os.path.join(output_dir, "config.json"))

necessary_data: Dict[str, Dict[str, torch.Tensor]] = {}
with safe_open(encoder_0_file, framework="pt") as fp:
    for k in fp.keys():
        if not "proj" in k:
            continue
        _, linear_name, q_part = k.split(".")
        if not linear_name in necessary_data:
            necessary_data[linear_name] = {}
        necessary_data[linear_name][q_part] = fp.get_tensor(k)
for linear_name, part_weight in necessary_data.items():
    save_torch_safetensors(
        necessary_data[linear_name],
        os.path.join(output_dir, f'{linear_name}.safetensors')
    )
