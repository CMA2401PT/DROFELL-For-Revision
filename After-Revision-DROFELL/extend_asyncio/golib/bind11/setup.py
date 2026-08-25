import os
from setuptools import setup, Extension
import pybind11

this_dir = os.path.dirname(__file__)


def local_path(path: str):
    return os.path.abspath(os.path.join(this_dir, path))


setup(
    name="goselector",
    ext_modules=[
        Extension(
            "goselector",
            [local_path("goselector.cpp"), local_path("unblocking.cpp")],
            include_dirs=[pybind11.get_include(), local_path(".")],  # "." 包含 libgoadd.h
            extra_objects=[local_path("libgo.a")],  # 静态库路径
            depends=[local_path("libgo.a")],
            language="c++",
        ),
    ],
)
