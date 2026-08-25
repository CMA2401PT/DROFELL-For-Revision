

from typing import Any, Literal
from uuid import uuid4


class IDMaker:
    def __init__(self, prefix: str = "", mode: Literal["uuid", ""] = "") -> None:
        self.incremental_idx: int = 0
        self.mode = mode
        self.prefix = prefix

    def __call__(self) -> Any:
        base = ""
        if self.mode == "uuid":
            base = uuid4().__str__()
        else:
            base = f"{self.incremental_idx}"
            self.incremental_idx += 1
        return f"{self.prefix}{base}"
