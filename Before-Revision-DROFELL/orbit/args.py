from typing import Any, Dict, Tuple


class ArgsKwargs:
    def __init__(self, *args: Any, **kwargs: Any):
        self._args = args
        self._kwargs = kwargs

    def set(self, arg_pack: Tuple[Tuple[Any], Dict[str, Any]]):
        self._args = arg_pack[0]
        self._kwargs = arg_pack[1]

    def get(self, key: str, default: Any):
        if key in self._kwargs:
            return self._kwargs[key]
        return default

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    def __getitem__(self, index: int):
        return self._args[index]

    def __getattr__(self, key: str):
        if key in self._kwargs:
            return self._kwargs[key]
        raise AttributeError(f"No such attribute: {key}")

    def __repr__(self):
        return f"<ArgsKwargs args={self._args} kwargs={self._kwargs}>"
