from typing import List, Tuple, Union
import torch
from .prebuilt_hack_extend import tensor_from_pointer
from .prebuilt_hack_extend import tensor_to_memory_view as _tensor_to_memory_view
from .prebuilt_hack_extend import tensor_from_memory_view as _tensor_from_memory_view


def tensor_as_leaf(non_leaf: torch.Tensor) -> torch.Tensor:
    int_address = non_leaf.data_ptr()
    sizes = [int(i) for i in list(non_leaf.shape)]
    new_tensor = tensor_from_pointer(int_address, non_leaf, sizes)
    return new_tensor


def tensor_to_memory_view(tensor: torch.Tensor) -> memoryview:
    mv: memoryview = _tensor_to_memory_view(tensor)
    return mv


meta_device = torch.empty((0,), device="meta").device


def tensor_from_memory_view(mv: memoryview, sizes: Union[List[int], torch.Size, Tuple[int, ...]], dtype: torch.dtype) -> torch.Tensor:
    template = torch.empty(sizes, dtype=dtype, device=meta_device)
    t = _tensor_from_memory_view(mv, template)
    return t
