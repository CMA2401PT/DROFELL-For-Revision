import os
from setuptools import setup
import pybind11
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


this_dir = os.path.dirname(__file__)


def local_path(path: str):
    return os.path.abspath(os.path.join(this_dir, path))


setup(
    name="cuda_async_load",
    ext_modules=[
        CUDAExtension(
            "cuda_async_load",
            [local_path("export.cpp"), local_path("event_syncer.cpp"), local_path("memory_copy.cpp")],
            include_dirs=[pybind11.get_include(), local_path(".")],  # "." 包含 libgoadd.h
            extra_objects=[],  # 静态库路径
            language="c++"
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

# python setup.py build_ext --inplace
