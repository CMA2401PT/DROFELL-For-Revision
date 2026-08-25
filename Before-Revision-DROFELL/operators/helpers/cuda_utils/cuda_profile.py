from typing import Callable, Dict, List, Optional, Set, cast
import torch
import torch.autograd.profiler as prof
from torch.autograd.profiler_util import FunctionEvent


class WrappedEvent(FunctionEvent):
    def __init__(self, ev: FunctionEvent):
        self.__dict__.update(ev.__dict__)

    def __str__(self) -> str:
        def get_name_chain(ev: Optional[FunctionEvent]) -> str:
            if ev is None:
                return ""
            else:
                return get_name_chain(ev.cpu_parent)+f"{ev.name}->"
        return f"{get_name_chain(self.cpu_parent)}{self.name}"

    def __repr__(self):
        return self.__str__()


def make_default_profile() -> torch.profiler.profile:
    return torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CUDA,
        torch.profiler.ProfilerActivity.CPU
    ],
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        record_shapes=False,
        with_stack=False
    )


profiler: Optional[torch.profiler.profile] = None


def start_profile(make_profile: Optional[Callable[[], torch.profiler.profile]]):
    global profiler
    assert profiler is None
    if make_profile is None:
        make_profile = make_default_profile
    profiler = make_profile()
    profiler.start()
    return profiler


def end_profile(not_exist_ok: bool = False) -> List[WrappedEvent]:
    global profiler
    if not_exist_ok:
        if profiler is None:
            return []
    assert profiler is not None
    profiler.stop()
    prof.KinetoStepTracker.erase_step_count("ProfilerStep")
    _events = profiler.events()
    events: List[WrappedEvent] = [
        WrappedEvent(e)
        for e in cast(List[FunctionEvent], _events)
        if e.cuda_time_total != 0
    ] if _events is not None else []
    ev_ids: Dict[int, Set[int]] = {}
    for e in events:
        ev_ids[e.id] = set()
        if e.cpu_parent is not None:
            ev_ids[e.cpu_parent.id].add(e.id)
    events = [e for e in events if len(ev_ids[e.id]) == 0]
    profiler = None
    return events
