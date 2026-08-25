# maximum time for infer task group trap window
from typing import Literal


INTERCEPT_WINDOW_TIME = 1/1000  # 1ms
INTERCEPT_MAX_SIZE = 8
GROUP_INFERENCE = True


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
