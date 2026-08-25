import os
from setuptools import setup, Extension
import pybind11

this_dir = os.path.dirname(__file__)


def local_path(path: str):
    return os.path.abspath(os.path.join(this_dir, path))


setup(
    name="simple_load",
    ext_modules=[
        Extension(
            "simple_load",
            [local_path("load.cpp")],
            include_dirs=[pybind11.get_include(), local_path(".")],  # "." 包含 libgoadd.h
            extra_objects=[],  # 静态库路径
            language="c++"
        ),
    ],
    zip_safe=False,
)

# python setup.py build_ext --inplace
