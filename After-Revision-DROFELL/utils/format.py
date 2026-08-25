from typing import Optional, Union


def format_size(size: Union[int, float], digital: Optional[int] = None) -> str:
    units = ["KiB", "MiB", "GiB"]
    out = f"{size}B"
    for u in units:
        if -2**10 < size < 2**10:
            break
        size = size/(2**10)
        if digital is None:
            out = f"{size}{u}"
        else:
            out = format(size, f".{digital}f") + f"{u}"
    return out
