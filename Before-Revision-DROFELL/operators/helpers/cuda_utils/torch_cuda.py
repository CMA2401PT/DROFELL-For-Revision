from typing import Any, Dict
import torch


def new_event(blocking: bool, enable_timing: bool = False) -> torch.cuda.streams.Event:
    r""" 
        actual torch.cuda.Event, ignore type hint
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
    """
    ev = torch.cuda.streams.Event(enable_timing=enable_timing, blocking=blocking)
    return ev  # type: ignore


def new_stream(device: torch.device, priority: int = 0) -> torch.cuda.streams.Stream:
    r"""Wrapper around a CUDA stream.

    actual torch.cuda.Stream, ignore type hint

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream, should be 0 or
            negative, where negative numbers indicate higher priority. By default,
            streams have priority 0.

    """
    stream = torch.cuda.streams.Stream(device=device, priority=priority)
    return stream  # type: ignore


def state_dict_cpu_to_device(state_dict: Dict[Any, Any], gpu_device: torch.device, deepth: int = 0):
    new: Dict[Any, Any] = {}
    if deepth == 0:
        stream = new_stream(gpu_device)
        with torch.cuda.stream(stream):
            for key, param in state_dict.items():
                # Not sure there are any global tensors in the state dict
                if isinstance(param, torch.Tensor):
                    new[key] = param.to(gpu_device, non_blocking=True)
                elif isinstance(param, dict):
                    new[key] = state_dict_cpu_to_device(param, gpu_device, deepth + 1)  # type:ignore
                else:
                    new[key] = param
        stream.synchronize()
    else:
        for key, param in state_dict.items():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                new[key] = param.to(gpu_device, non_blocking=True)
            elif isinstance(param, dict):
                new[key] = state_dict_cpu_to_device(param, gpu_device, deepth + 1)  # type:ignore
            else:
                new[key] = param
    return new


def state_dict_device_to_cpu(state_dict: Dict[Any, Any], gpu_device: torch.device, deepth: int = 0):
    new: Dict[Any, Any] = {}
    if deepth == 0:
        stream = new_stream(gpu_device)
        with torch.cuda.stream(stream):
            for key, param in state_dict.items():
                # Not sure there are any global tensors in the state dict
                if isinstance(param, torch.Tensor):
                    new[key] = param.to("cpu", non_blocking=True)
                elif isinstance(param, dict):
                    new[key] = state_dict_device_to_cpu(param, gpu_device, deepth + 1)  # type:ignore
                else:
                    new[key] = param
        stream.synchronize()
    else:
        for key, param in state_dict.items():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                new[key] = param.to("cpu", non_blocking=True)
            elif isinstance(param, dict):
                new[key] = state_dict_device_to_cpu(param, gpu_device, deepth + 1)  # type:ignore
            else:
                new[key] = param
    return new
