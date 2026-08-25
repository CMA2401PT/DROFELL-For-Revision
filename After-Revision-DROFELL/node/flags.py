# maximum time for infer task group trap window
import os
from typing import Literal


INTERCEPT_WINDOW_TIME = float(os.getenv("INTERCEPT_WINDOW_TIME", "0.001"))
INTERCEPT_MAX_SIZE = int(os.getenv("INTERCEPT_MAX_SIZE", "8"))
GROUP_INFERENCE = os.getenv("GROUP_INFERENCE", "1").strip().lower() not in {"0", "false", "no", "off"}
RECOVERY_REPLAY = os.getenv("RECOVERY_REPLAY", "chunk").strip().lower()
if RECOVERY_REPLAY not in {"sequential", "chunk"}:
    raise ValueError(
        "RECOVERY_REPLAY must be either 'sequential' or 'chunk'")


def SHOULD_INTERCEPT(step: int):
    # 1,4,16,64,256...
    return ((step & (step - 1)) == 0 and (step % 3 == 1))


async def InspectPostpone(kind: Literal[
        "infer",
        "train",
        "backbone.infer",
        "backbone.train"
]):
    pass
