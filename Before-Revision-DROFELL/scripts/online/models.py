import os
import split_qllm_models
from node.aux_node.protocols import ExtraInfo, ExtraModelInfo
from operators.helpers.model_lib.common import ModelsLib
from dealers.common.base_types import group_model_with_split

lib_path = split_qllm_models.lib_path
models = ModelsLib(split_qllm_models.lib_path)
extra_info = ExtraInfo(models={})
possible_split = [int(v) for v in os.getenv("SPLITS", "10,8,6,4").split(",")]
print(f"splits set to: {possible_split}")
for model_name, info in models.models_storage.items():
    num_decoders = info.config.num_decoders
    split = 0
    for _split in possible_split:
        if num_decoders % _split == 0:
            split = _split
            break
    assert split != 0, f"cannot decide split of {num_decoders}"
    head, model_groups, tail = group_model_with_split(
        base_model_name=model_name,
        decoders=num_decoders,
        group_fn=lambda layers: [split for _ in range(layers//split)]
    )
    extra_info.models[model_name] = ExtraModelInfo(
        hidden_size=info.config.hidden_size,
        num_decoders=num_decoders,
        decoders_groups=model_groups,
        head=head,
        tail=tail,
    )
