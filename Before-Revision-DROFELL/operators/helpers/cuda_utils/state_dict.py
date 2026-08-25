from typing import Any, Dict
from .torch_cuda import new_stream
import torch


def state_dict_to(state_dict: Dict[Any, Any], device: torch.device, deepth: int = 0):
    new: Dict[Any, Any] = {}
    if deepth == 0:
        stream = new_stream(device)
        with torch.cuda.stream(stream):
            for key, param in state_dict.items():
                # Not sure there are any global tensors in the state dict
                if isinstance(param, torch.Tensor):
                    new[key] = param.to(device, non_blocking=True)
                elif isinstance(param, dict):
                    new[key] = state_dict_to(param, device, deepth + 1)  # type:ignore
                else:
                    new[key] = param
        stream.synchronize()
    else:
        for key, param in state_dict.items():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                new[key] = param.to(device, non_blocking=True)
            elif isinstance(param, dict):
                new[key] = state_dict_to(param, device, deepth + 1)  # type:ignore
            else:
                new[key] = param
    return new
