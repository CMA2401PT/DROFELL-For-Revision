import re
import numpy as np
import torch
from typing import Any, Dict, List, Literal, Set, Tuple, Union
from dataclasses import dataclass
from utils.format import format_size


@dataclass
class Block:
    address: int
    alloc_size: int
    requested_size: int
    state: Literal["active_allocated", "inactive"]

    def __repr__(self) -> str:
        return f"[{self.state},{self.requested_size}+{self.alloc_size-self.requested_size}@{self.address}]"


@dataclass
class BlockSize:
    requested: int
    allocated: int

    @property
    def approximate(self):
        if self.requested < 512 and self.requested > -512:
            assert self.allocated == 512 or self.allocated == -512
            return self.allocated
        else:
            return self.requested

    def __str__(self) -> str:
        return f"requested={self.to_size(self.requested)},allocated={self.to_size(self.allocated)},approximate={self.to_size(self.approximate)}"

    @staticmethod
    def to_size(v: int) -> str:
        if v < 1024:
            return f"{v}B"
        unit = ["B", "KiB", "MiB", "GiB"]
        unit_i = 0
        vf = float(v)
        while vf > 2**10:
            unit_i += 1
            vf /= 2**10
        return f"{vf:.2f}{unit[unit_i]}"


@dataclass
class MemoryAccumulateStatus:
    blocks: Dict[int, BlockSize]

    @property
    def allocated(self):
        return sum(b.allocated for b in self.blocks.values())

    @property
    def requested(self):
        return sum(b.requested for b in self.blocks.values())

    @property
    def approximate(self):
        return sum(b.approximate for b in self.blocks.values())

    @classmethod
    def from_segments(cls, segemnts: List[Dict[str, Any]], device_idx: int):
        blocks: Dict[int, BlockSize] = {}
        for segment in segemnts:
            _device = segment['device']
            if device_idx != _device:
                continue
            for block in segment['blocks']:
                if block['state'] == "active_allocated":
                    blocks[block['address']] = BlockSize(block['requested_size'], block['size'])
        return cls(blocks)

    def __str__(self) -> str:
        s = "Memory Accumulate:\n"
        s += f"  Req: {format_size(self.requested)}\n  Alloc:{format_size(self.allocated)}\n  Approx:{format_size(self.approximate)}"
        return s


# @dataclass
# class MemoryStatus:
#     device: int
#     blocks: Dict[int, BlockSize] = field(default_factory=dict)
#     changes: List[BlockSize] = field(default_factory=list)
#     accumulate: BlockSize = field(default_factory=lambda: BlockSize(0, 0))
#     delta: BlockSize = field(default_factory=lambda: BlockSize(0, 0))

#     def get_update(self):
#         return MemoryStatus.from_last(self)

#     @classmethod
#     def from_last(cls, last: "MemoryStatus") -> "MemoryStatus":
#         blocks = cuda_memory_blocks(last.device)
#         new_blocks: Dict[int, BlockSize] = {}
#         accumulate = BlockSize(0, 0)
#         for block in blocks:
#             if block.state == "active_allocated":
#                 assert not block.address in new_blocks
#                 new_blocks[block.address] = BlockSize(block.requested_size, block.alloc_size)
#                 accumulate = BlockSize(accumulate.requested + block.requested_size, accumulate.allocated + block.alloc_size)

#         changes: List[BlockSize] = []
#         for address, block in last.blocks.items():
#             new_block = new_blocks.get(address)
#             if new_block is None:
#                 # delete
#                 changes.append(BlockSize(-block.requested, -block.allocated))
#             elif new_block.requested != block.requested or new_block.allocated != block.allocated:
#                 # change
#                 changes.append(BlockSize(new_block.requested - block.requested, new_block.allocated - block.allocated))
#         for address, new_block in new_blocks.items():
#             if address not in last.blocks:
#                 # add
#                 changes.append(new_block)
#         new = cls(
#             device=last.device,
#             blocks=new_blocks,
#             changes=changes,
#             delta=BlockSize(
#                 accumulate.requested - last.accumulate.requested,
#                 accumulate.allocated - last.accumulate.allocated,
#             ),
#             accumulate=accumulate
#         )
#         # check
#         # 1. is delta match changes?
#         delta_check = BlockSize(0, 0)
#         for block in changes:
#             delta_check = BlockSize(
#                 delta_check.requested + block.requested,
#                 delta_check.allocated + block.allocated,
#             )
#         assert delta_check == new.delta, f"{delta_check},{new.delta}"

#         # 2. is accumulate match torch.cuda.memory_allocated?
#         allocated_check = torch.cuda.memory_allocated()
#         if allocated_check != new.accumulate.allocated:
#             print("warning: cuda memory changed during computing state")
#             return cls.from_last(last)
#         #     blocks = cls._cuda_memory_blocks()
#         #     total_allocated = 0
#         #     for block in blocks:
#         #         if block.state == "active_allocated":
#         #             total_allocated += block.size
#         # assert allocated_check == new.accumulate.allocated, f"{allocated_check,new.accumulate.allocated}"
#         return new

#     def __repr__(self) -> str:
#         return f"delta:[{self.delta}] all:[{self.accumulate}] "

class MemoryDeltaStatus:
    def __init__(self,
                 traces: List[Any],
                 frame_trim_rule: str = r'^(?!$)(?!\?\?$)(?!.*(/tmp|\.vscode|.cpp|site-packages|runpy.py|debug)).+$'
                 ) -> None:
        self.frame_trim_rule = None
        if frame_trim_rule != "":
            self.frame_trim_rule = re.compile(frame_trim_rule)

        # preserved_new_alloc requested
        self.free_before, self.preserved_new_alloc, self.history, self.preserved_alloc_frames, self.free_before_frames =\
            self.analyzed_trace(traces)
        self.data_block = self.gen_block_history(traces)
        self.peak_memory = max(self.history+[0])
        self.peak_step = int(np.argmax(self.history+[0]))

    @property
    def total_change(self):
        return self.preserved_new_alloc-self.free_before

    def _format_size(self, size: Union[int, float]) -> str:
        units = ["KiB", "MiB", "GiB"]
        out = f"{size}B"
        for u in units:
            if -2**10 < size < 2**10:
                break
            size = size/(2**10)
            out = f"{size}{u}"
        return out

    def __str__(self) -> str:
        def longest_common_prefix(strs: List[str]):
            prefix = ''
            for chars in zip(*strs):
                if len(set(chars)) == 1:
                    prefix += chars[0]
                else:
                    break
            return prefix
        pre = longest_common_prefix([
            frame["filename"]
            for (_, frames) in self.preserved_alloc_frames.values()
            for frame in frames
        ]+[
            frame["filename"]
            for (_, frames) in self.free_before_frames.values()
            for frame in frames
        ])
        if pre == "/":
            pre = ""

        s = "Memory Delta:\n"
        s += f"  Overall: {self._format_size(self.preserved_new_alloc-self.free_before)}\n"
        s += f"    New Preserved Alloc (Requested): {self._format_size(self.preserved_new_alloc)}\tFree Before: {self._format_size(self.free_before)}\n"
        if len(self.preserved_alloc_frames) > 0:
            s += f"  Preserved Alloc History:\n"
            for k, (size, frames) in self.preserved_alloc_frames.items():
                s += f"    memory @ {k} size: {self._format_size(size)}\n"
                for frame in frames[::-1]:
                    name = frame["name"]
                    file = frame["filename"].removeprefix(pre)
                    line = frame["line"]
                    s += f"      \"{file}\", line {line}\n"
        if len(self.free_before_frames) > 0:
            s += f"  Free Before History:\n"
            for k, (size, frames) in self.free_before_frames.items():
                s += f"    memory size: {self._format_size(-size)} @ {k}\n"
                for frame in frames[::-1]:
                    name = frame["name"]
                    file = frame["filename"].removeprefix(pre)
                    line = frame["line"]
                    s += f"      \"{file}\", line {line}\n"
        return s.strip()

    def print_current_frames(self):
        for k, (size, frames) in self.preserved_alloc_frames.items():
            print(f"memory @ {k} size: {self._format_size(size)}")
            for frame in frames:
                name = frame["name"]
                file = frame["filename"]
                line = frame["line"]
                print(f"\t{file}: {line}")
        if len(self.preserved_alloc_frames) == 0:
            print("all new allocated device memory are free")

    def trim_frame(self, frames: List[Dict[str, Any]]):
        return [
            frame
            for frame in frames
            if self.frame_trim_rule is None or self.frame_trim_rule.match(frame["filename"])
        ]

    def analyzed_trace(self, traces: List[Any]) -> Tuple[int, int, List[int], Dict[int, Tuple[int, Any]], Dict[int, Tuple[int, Any]]]:
        current: Dict[int, int] = {}
        alloc_frames: Dict[int, Tuple[int, Any]] = {}
        free_before_frames: Dict[int, Tuple[int, Any]] = {}
        free_before: Dict[int, int] = {}
        history: List[int] = []
        for trace in traces:
            action = trace["action"]
            addr = trace["addr"]
            size = trace["size"]
            if action == 'alloc':
                current[addr] = size
                history.append(sum(current.values()))
                alloc_frames[addr] = (size, self.trim_frame(trace["frames"]))
            if action == "free_requested":
                if not addr in current:
                    free_before_frames[addr] = (size, self.trim_frame(trace["frames"]))
            if action == "free_completed":
                if not addr in current:
                    free_before[addr] = size
                else:
                    del current[addr]
                    del alloc_frames[addr]
                    history.append(sum(current.values()))
            if action == "snapshot":
                continue
            if action == "segment_alloc":
                continue
            assert "frames" in trace, trace
            # assert len(self.trim_frame(trace["frames"])), trace
            # trace["frames"] = self.trim_frame(trace["frames"])
        return sum(free_before.values()), sum(current.values()), history, alloc_frames, free_before_frames

    @staticmethod
    def gen_block_history(traces: List[Any]):
        occoured: Set[int] = set()
        steps = 0
        for trace in traces:
            addr = trace["addr"]
            action = trace["action"]
            if action == 'alloc':
                occoured.add(addr)
                steps += 1
            elif action == "free_completed":
                steps += 1

        addrs = [a for a in occoured]
        addrs = sorted(addrs)
        addr_mapping = {a: i for i, a in enumerate(addrs)}
        data_block = np.zeros((len(addrs), steps), dtype=np.int64)

        current: Dict[int, int] = {}
        step_i = 0
        for (trace) in traces:
            action = trace["action"]
            addr = trace["addr"]
            size = trace["size"]
            if action not in ['alloc', 'free_completed']:
                continue

            if action == 'alloc':
                current[addr] = size
                for addr, size in current.items():
                    data_block[addr_mapping[addr], step_i] = size
            elif action == "free_completed":
                if addr in current:
                    del current[addr]

            for addr, size in current.items():
                data_block[addr_mapping[addr], step_i] = size
            step_i += 1
        return data_block

    def plot(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        _, steps = self.data_block.shape
        # set seaborn style
        sns.set_theme(context='paper', style='whitegrid')
        plt.rcParams["patch.force_edgecolor"] = False
        x = range(steps)
        y = self.data_block

        # Plot
        plt.stackplot(x, y)
        plt.xlim(-10, steps+10)


@dataclass
class MemoryStatus:
    delta: MemoryDeltaStatus
    accumulate: MemoryAccumulateStatus

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any], device_idx: int):
        traces = snapshot["device_traces"][device_idx]
        record = MemoryDeltaStatus(traces)
        segments = snapshot["segments"]
        status: MemoryAccumulateStatus = MemoryAccumulateStatus.from_segments(segments, device_idx)
        return cls(record, status)

    def __str__(self):
        return self.delta.__str__()+"\n"+self.accumulate.__str__()

    def __repr__(self) -> str:
        return self.__str__()


def record_start():
    torch.cuda.memory._record_memory_history(stacks="python")  # type:ignore


def record_ckpt():
    snapshot = torch.cuda.memory._snapshot()  # type:ignore
    devices = len(snapshot["device_traces"])
    info = {
        idx: MemoryStatus.from_snapshot(snapshot, idx)
        for idx in range(devices)
    }
    torch.cuda.memory._record_memory_history(enabled=None)  # type:ignore
    torch.cuda.memory._record_memory_history(stacks="python")  # type:ignore
    return info


def record_end():
    torch.cuda.memory._record_memory_history(enabled=None)  # type:ignore


def current_memory_status() -> Dict[int, MemoryAccumulateStatus]:
    snapshot = torch.cuda.memory_snapshot()
    all_devices: Dict[int, MemoryAccumulateStatus] = {}
    for segment in snapshot:
        _device = segment['device']
        all_devices[_device] = MemoryAccumulateStatus({})

    for segment in torch.cuda.memory_snapshot():
        _device = segment['device']
        device_info = all_devices[_device]

        for block in segment['blocks']:
            if block['state'] == "active_allocated":
                device_info.blocks[block['address']] = BlockSize(block['requested_size'], block['size'])
    return all_devices
