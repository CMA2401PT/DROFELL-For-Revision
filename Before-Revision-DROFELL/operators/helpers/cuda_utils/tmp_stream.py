from typing import Dict, cast
import torch

_tmp_streams: Dict[str, torch.cuda.Stream] = {}


def create_or_reuse_tmp_stream(purpose: str, device: torch.device):
    dev_name = device.__str__()
    full_name = purpose+"-"+dev_name
    if full_name not in _tmp_streams:
        _tmp_stream = cast(torch.cuda.Stream, torch.cuda.Stream(device))
        _tmp_streams[full_name] = _tmp_stream
    _tmp_stream = _tmp_streams[full_name]
    return _tmp_stream
