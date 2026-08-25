from dataclasses import dataclass
import io
import pickle
import time
from typing import Any, Callable, Dict, List, Tuple, Union, cast
import torch

from ..real_basic.basic import BasicBrain

from ..common.base_types import DeviceTensorCarrier, TensorMeta

from ..common.brain import Brain, Operate
from operators.helpers.cuda_utils import memory

FunctionName = str
# ARGS = Tuple[Any]
KWARGS = Dict[str, Any]
FinishCallback = Callable[[Any], None]
CallRecorder = Callable[[FunctionName, Union[Operate[Any], KWARGS]], FinishCallback]


class NopRecorder:
    def __init__(self) -> None:
        pass

    def __call__(self, is_coro: bool, fn_name: str, fn_kwargs: KWARGS) -> FinishCallback:
        return lambda _: self._on_finish(fn_name)

    def _on_finish(self, fn_name: str) -> None:
        return None


@dataclass
class TimeRecord:
    idle_time: int = 0
    begin_time: int = 0
    delta_time: int = 0
    extra_async_delta_time: int = 0
    description: Tuple[str, ...] = ("",)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self.begin_time/10**6:.2f}ms +{self.delta_time/10**6:.2f}+{self.extra_async_delta_time/10**6:.2f}ms {self.busy_time:.2%}"


class TimeRecorder:
    def __init__(self, sync_cuda: bool = False, enabled: bool = True) -> None:
        self.last_operate_finish_time: int = 0
        self.total_busy_time: int = 0
        self.records: List[Tuple[str, TimeRecord]] = []
        self.sync_cuda = sync_cuda
        self.enabled = enabled

    def reset(self, enable: bool):
        self.last_operate_finish_time = 0
        self.total_busy_time = 0
        self.records = []
        self.enabled = enable

    def record(self, data: Tuple[str, TimeRecord]):
        self.records.append(data)

    def __call__(self, fn_name: str, op_or_args: Union[Operate[Any],  KWARGS]) -> FinishCallback:
        if not self.enabled:
            return lambda _: None
        start_time, idle_time = self.record_start_time_and_idle_time()
        description = (fn_name, "")
        if isinstance(op_or_args, Operate):
            description = op_or_args.desc()
        return lambda _: self.on_finish(fn_name, start_time, idle_time, description)

    def on_finish(self, fn_name: str, start_time: int, idle_time: int, description: Tuple[str, ...]) -> None:
        end_time = time.time_ns()
        delta_time = end_time-start_time
        extra_async_delta_time = 0
        if self.sync_cuda:
            start_time = end_time
            torch.cuda.synchronize()
            end_time = time.time_ns()
            extra_async_delta_time = end_time-start_time
        self.last_operate_finish_time = end_time
        self.total_busy_time += delta_time+extra_async_delta_time
        self.record((
            fn_name,
            TimeRecord(
                idle_time=idle_time,
                begin_time=start_time,
                delta_time=delta_time,
                extra_async_delta_time=extra_async_delta_time,
                description=description
            ))
        )

    def record_start_time_and_idle_time(self) -> Tuple[int, int]:
        start_time = time.time_ns()
        idle_time = start_time-self.last_operate_finish_time
        if self.last_operate_finish_time == 0:
            idle_time = 0
        return start_time, idle_time

    def dump_info(self):
        return [
            (f, r.__dict__.copy())
            for f, r in self.records
        ]


class MetaPickler(pickle.Pickler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, protocol=-1)

    def reducer_override(self, obj: object):  # type:ignore
        if isinstance(obj, DeviceTensorCarrier):
            meta = obj.meta
            return meta.__reduce__()
        if isinstance(obj, torch.Tensor):
            return TensorMeta, (id(obj), obj.data_ptr(), obj.dtype, obj.is_cpu, obj.shape)
        else:
            return NotImplemented

    @staticmethod
    def dumps(obj: Any):
        f = io.BytesIO()
        p = MetaPickler(f)
        p.dump(obj)
        return f.getvalue()


class ActualIORecorder:
    """
    record actual input/output of each operate
    """

    def __init__(self, brain: Brain, with_drop: bool = True) -> None:
        self.capturing = ""
        self.brain = brain
        self.device_idx = brain.device.index
        self.recorded: List[Tuple[int, str, bytes, memory.MemoryStatus, bytes, Dict[str, int]]] = []
        memory.record_start()
        self.actual_accumulated = 0
        self.with_drop = with_drop
        self.count = 0

    def __call__(self, fn_name: str, op_or_args: Union[Operate[Any],  KWARGS]) -> FinishCallback:
        # print(fn_name, "called")
        # could leek in node drop
        if not self.with_drop:
            status = memory.record_ckpt()[self.device_idx]
            assert status.delta.free_before == 0, status.delta
            assert status.delta.preserved_new_alloc == 0, status.delta
        assert self.capturing == "", f"still capturing {self.capturing} but want capture {fn_name}"
        self.capturing = fn_name
        sdata = MetaPickler.dumps(op_or_args)
        called = {"called": False}
        return lambda ret: self.on_finish(fn_name, sdata, ret, called)

    def on_finish(self, fn_name: str, serialized_args: bytes, ret: Any, called: Any) -> None:
        assert not called["called"]
        called["called"] = True
        self.count += 1
        count = self.count
        # print(fn_name, "done")
        self.capturing = ""
        status = memory.record_ckpt()[self.device_idx]
        actual_change = status.delta.total_change
        self.actual_accumulated += actual_change
        # assert self.actual_accumulated == status.accumulate.requested, f"{fn_name} {(self.actual_accumulated-status.accumulate.requested)/(2**20)}"
        memory_ret = {k: v for k, v in cast(BasicBrain, self.brain).memory_modeling.pools.items()}
        self.recorded.append((count, fn_name, serialized_args, status, MetaPickler.dumps(ret), memory_ret))
        # WARNING! for unknown reason, appended value could shuffle!
        return
        # @dataclass
        # class MemoryRecord:
        #     modeling_current: Dict[str, int] = field(default_factory=dict)
        #     modeling_delta: Dict[str, int] = field(default_factory=dict)
        #     actual_current: Optional[memory.BlockSize] = None
        #     actual_delta: Optional[memory.BlockSize] = None

        # class MemoryRecoder:
        #     def __init__(self, record_actual_cuda_memory: bool = False, device_idx: int = 0) -> None:
        #         self.record_actual_cuda_memory = record_actual_cuda_memory
        #         self.device_idx = device_idx
        #         self.records: List[Tuple[FunctionName, MemoryRecord]] = []
        #         self.get_pool: Callable[[], Dict[str, int]]
        #         self.record = self.records.append

        #     def reset(self):
        #         self.records = []

        #     def __call__(self, is_coro: bool, fn_name: str,  fn_kwargs: KWARGS) -> FinishCallback:
        #         rec = MemoryRecord({k: v for k, v in self.get_pool().items()})
        #         actual_memory_rec: Optional[memory.MemoryStatus] = None
        #         if self.record_actual_cuda_memory:
        #             actual_memory_rec = memory.MemoryStatus(self.device_idx).get_update()
        #         return lambda: self.on_finish(fn_name, actual_memory_rec, rec)

        #     def on_finish(self, fn_name: str, actual_memory: Optional[memory.MemoryStatus], rec: MemoryRecord) -> None:
        #         last = rec.modeling_current
        #         rec.modeling_delta = {k: v - last.get(k, 0) for k, v in self.get_pool().items() if (k == "all" or (v - last.get(k, 0) != 0))}
        #         rec.modeling_current = {k: v for k, v in self.get_pool().items()}
        #         if actual_memory is not None:
        #             current_memory_state = memory.MemoryStatus.from_last(actual_memory)
        #             rec.actual_current = current_memory_state.accumulate
        #             rec.actual_delta = current_memory_state.delta
        #         self.record((fn_name, rec))

        # def readable_unit(value: float) -> str:
        #     unit = ["", "KiB", "MiB", "GiB"]
        #     i = 0
        #     while value > 2**10 or value < -2**10:
        #         value = value/(2**10)
        #         i += 1
        #     if i == 0:
        #         return f"{value}"
        #     return f"{value:.1f}{unit[i]}"


# class CudaMemoryDeltaRecorder:
#     def __init__(self) -> None:
#         self.modeling_memory: Optional[Callable[[], int]] = None
#         self.last_cuda_total: int = 0
#         self.last_call: str = ""
#         self.id_maker = IDMaker()

#     def __call__(self, is_coro: bool, fn_name: str, fn_kwargs: KWARGS) -> FinishCallback:
#         detail = False
#         before_blocks = memory.cuda_memory_blocks(0)
#         before_modeling = 0 if self.modeling_memory is None else self.modeling_memory()
#         before_blocks_by_address: Dict[int, int] = {}
#         for b in before_blocks:
#             if b.state != "active_allocated":
#                 continue
#             before_blocks_by_address[b.address] = self.alloc_simu(b.requested_size)
#         assert sum(before_blocks_by_address.values()) == self.last_cuda_total
#         call_id = self.id_maker()
#         assert self.last_call == ""
#         self.last_call = call_id
#         return lambda: self.on_finish(fn_name, before_blocks_by_address, before_modeling, detail, call_id)

#     def alloc_simu(self, value: int) -> int:
#         if value < 512:
#             return 512
#         return value

#     def on_finish(self, fn_name: str, before_blocks_by_address: Dict[int, int], before_modeling: int, detail: bool, call_id: str) -> None:
#         assert self.last_call == call_id, "concurrent will broke correctness"
#         self.last_call = ""
#         after_blocks = memory.cuda_memory_blocks(0)
#         after_modeling = 0 if self.modeling_memory is None else self.modeling_memory()
#         alloc_frames: Dict[int, List[Dict[str, Any]]] = {}
#         modeling_change = after_modeling-before_modeling
#         after_blocks_by_address: Dict[int, int] = {}
#         for b in after_blocks:
#             if b.state != "active_allocated":
#                 continue
#             after_blocks_by_address[b.address] = self.alloc_simu(b.requested_size)
#         free_blocks: Dict[int, int] = {}
#         alloc_blocks: Dict[int, int] = {}
#         alloc_blocks_with_frame: Dict[int, List[Dict[str, Any]]] = {}
#         for addr, old_size in before_blocks_by_address.items():
#             if addr in after_blocks_by_address and old_size == after_blocks_by_address[addr]:
#                 continue
#             free_blocks[addr] = old_size
#         for addr, new_size in after_blocks_by_address.items():
#             if addr in before_blocks_by_address and new_size == before_blocks_by_address[addr]:
#                 continue
#             alloc_blocks[addr] = new_size
#             if addr in alloc_frames:
#                 alloc_blocks_with_frame[addr] = alloc_frames[addr]
#         alloc_total = 0
#         free_total = 0
#         error_total = 0
#         print(fn_name)
#         if len(alloc_blocks):
#             print("\talloc blocks:", end="\t")
#             for addr, size in alloc_blocks.items():
#                 if detail:
#                     print(readable_unit(size)+"@"+f"{addr}", end=", ")
#                 alloc_total += size
#             print("total:"+readable_unit(alloc_total))
#         if len(free_blocks):
#             print("\tfree blocks:", end="\t")
#             for addr, size in free_blocks.items():
#                 if detail:
#                     print(readable_unit(size)+"@"+f"{addr}", end=", ")
#                 free_total += size
#             print("total:"+readable_unit(free_total))
#         if alloc_total != 0 or free_total != 0 or modeling_change != 0:
#             error_total = alloc_total-free_total-modeling_change
#             print("\ttotal: ", readable_unit(value=alloc_total-free_total), " modeling: ", readable_unit(modeling_change), " error: ", readable_unit(error_total))
#             print("error accumulate: ", readable_unit(sum(after_blocks_by_address.values())-after_modeling))
#         self.last_cuda_total = sum(after_blocks_by_address.values())


# class MemoryAndTimeRecorder:
#     def __init__(self, sync_cuda: bool, record_actual_cuda_memory: bool = False, device_idx: int = 0) -> None:
#         self.memory_recorder = MemoryRecoder(record_actual_cuda_memory, device_idx)
#         self.time_recorder = TimeRecorder(sync_cuda)
#         self.records: List[Tuple[FunctionName, MemoryRecord, TimeRecord]] = []
#         self._last_time: TimeRecord
#         self._last_memory: MemoryRecord

#         def time_redirect(name_and_time: Tuple[FunctionName, TimeRecord]):
#             self._last_time = name_and_time[1]
#         self.time_recorder.record = time_redirect

#         def memory_redirect(name_and_memory: Tuple[FunctionName, MemoryRecord]):
#             self._last_memory = name_and_memory[1]
#         self.memory_recorder.record = memory_redirect

#     def __call__(self, is_coro: bool, fn_name: str,  fn_kwargs: KWARGS) -> FinishCallback:
#         memory_cb = self.memory_recorder(is_coro, fn_name, fn_kwargs)
#         time_cb = self.time_recorder(is_coro, fn_name, fn_kwargs)
#         return lambda: self.on_finish(fn_name, memory_cb, time_cb)

#     def reset(self):
#         self.time_recorder.reset()
#         self.memory_recorder.reset()
#         self.records = []

#     def on_finish(self, fn_name: str, memory_cb: FinishCallback, time_cb: FinishCallback) -> None:
#         time_cb()
#         memory_cb()
#         self.records.append((fn_name, self._last_memory, self._last_time))
