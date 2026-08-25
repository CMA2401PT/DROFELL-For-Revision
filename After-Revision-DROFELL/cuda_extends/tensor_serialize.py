import os

import torch

try:
    from .prebuilt_serialize_helper import cpu_tensor_to_bytes as cpu_tensor_to_bytes
    from .prebuilt_serialize_helper import bytes_to_cpu_tensor as bytes_to_cpu_tensor
    from .prebuilt_serialize_helper import tensor_copy_async as tensor_copy_async
except Exception as e:
    from torch.utils.cpp_extension import load
    extension_name = "rebulit_serialize_helper"
    verbose = True
    library_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensor_serialize")
    load(
        name="load_"+extension_name,
        sources=[os.path.join(library_dir, "serialize_helper.cpp")],
        extra_include_paths=[],
        verbose=verbose,
        extra_ldflags=[],
        extra_cuda_cflags=["-lineinfo", "-std=c++17"],
        extra_cflags=["-O0", "-std=c++17"],
    )
    from rebulit_serialize_helper import cpu_tensor_to_bytes as cpu_tensor_to_bytes  # type:ignore
    from rebulit_serialize_helper import bytes_to_cpu_tensor as bytes_to_cpu_tensor  # type:ignore
    from rebulit_serialize_helper import tensor_copy_async as tensor_copy_async  # type:ignore


def copy_tensor_async(src: torch.Tensor, dst: torch.Tensor, src_is_gpu: bool):
    tensor_copy_async(src, dst, 0 if src_is_gpu else 1)


# def cpu_tensor_to_bytes(t: torch.Tensor) -> bytes:
#     return _cpu_tensor_to_bytes(t)


# def bytes_to_cpu_tensor(bs: bytes) -> torch.Tensor:
#     return _bytes_to_cpu_tensor(bs)


# if __name__ == "__main__":
#     # python serialize_helper.py build_ext - -inplace
#     # prebuilt_extension()
#     t = torch.ones(dtype=torch.float16, size=(3, 3))
#     for i in range(3):
#         for j in range(3):
#             t[i][j] = i*3+j
#     tensor_bytes = cpu_tensor_to_bytes(t)
#     print(tensor_bytes)
#     print(bytes_to_cpu_tensor(tensor_bytes))
#     print("abc")
