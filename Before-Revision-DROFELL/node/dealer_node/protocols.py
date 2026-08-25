from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple

from node.compute_node import TrainInit
from node.compute_node import InferInit
from ..compute_node.train_meta import ModelCKPT, TrainTask
from ..compute_node.infer_meta import InferTask, PutBlocks


@dataclass
class CreateTrainNode:
    name: str
    config: TrainInit
    ckpt_period: int


API_CreateTrainNode = "create_train_node"


@dataclass
class CreateInferNode:
    name: str
    config: InferInit
    block_retrieve: bool


API_CreateInferNode = "create_infer_node"


@dataclass
class RemoveNode:
    name: str


API_RemoveNode = "remove_node"


@dataclass
class MultiPathInferTask:
    task: InferTask
    resource_loc: Dict[int, List[str]]
    trace: List[str]
    trace_invalid: bool = False
    trace_override: Optional[List[str]] = None
    detail_trace: Optional[List[Tuple[str, int]]] = None

    @property
    def current(self):
        return self.task.navi.current

    @property
    def unique_id(self):
        return self.task.unique_id

    def shallow_copy(self):
        return MultiPathInferTask(
            task=self.task.shallow_copy(),
            resource_loc={k: v for k, v in self.resource_loc.items()},
            trace_invalid=self.trace_invalid,
            trace=self.trace.copy(),
            trace_override=self.trace_override.copy() if self.trace_override is not None else None,
            detail_trace=self.detail_trace.copy() if self.detail_trace is not None else None
        )

    def detail_trace_fork(self):
        if self.detail_trace is None:
            return self
        return MultiPathInferTask(
            task=self.task,
            resource_loc=self.resource_loc,
            trace=self.trace,
            trace_invalid=self.trace_invalid,
            trace_override=self.trace_override,
            detail_trace=self.detail_trace.copy()
        )

    def record_step(self, step_name: str):
        if self.detail_trace is not None:
            self.detail_trace.append((step_name, time.time_ns()))

    def extend_steps(self, steps: List[Tuple[str, int]]):
        if self.detail_trace is not None:
            self.detail_trace.extend(steps)

    def __str__(self) -> str:
        return f"{self.task}(trace={self.trace})"

    def __repr__(self) -> str:
        return self.__str__()


API_InferTask = "infer_task"


@dataclass
class GroupedInferTask:
    tasks: List[MultiPathInferTask]


API_GroupedInferTask = "grouped_infer_task"


@dataclass
class InferPutBlocks:
    infer_node_name: str
    blocks: PutBlocks

    def __str__(self) -> str:
        return f"{self.blocks.__str__()}@{self.infer_node_name}"


API_InferPutBlocks = "infer_put_blocks"


@dataclass
class MultiPathTrainTask:
    task: TrainTask
    resource_loc: Dict[int, List[str]]
    trace: List[str]
    trace_invalid: bool = False
    trace_override: Optional[List[str]] = None
    detail_trace: Optional[List[Tuple[str, int]]] = None

    def __str__(self) -> str:
        return f"{self.task}(trace={self.trace})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def current(self):
        return self.task.navi.current

    @property
    def unique_id(self):
        return self.task.minibatch_id

    def shallow_copy(self):
        return MultiPathTrainTask(
            task=self.task.shallow_copy(),
            resource_loc={k: v for k, v in self.resource_loc.items()},
            trace_invalid=self.trace_invalid,
            trace=self.trace.copy(),
            trace_override=self.trace_override.copy() if self.trace_override is not None else None,
            detail_trace=self.detail_trace.copy() if self.detail_trace is not None else None
        )

    def detail_trace_fork(self):
        if self.detail_trace is None:
            return self
        return MultiPathTrainTask(
            task=self.task,
            resource_loc=self.resource_loc,
            trace=self.trace,
            trace_invalid=self.trace_invalid,
            trace_override=self.trace_override,
            detail_trace=self.detail_trace.copy()
        )

    def record_step(self, step_name: str):
        if self.detail_trace is not None:
            self.detail_trace.append((step_name, time.time_ns()))

    def extend_steps(self, steps: List[Tuple[str, int]]):
        if self.detail_trace is not None:
            self.detail_trace.extend(steps)


API_TrainTask = "train_task"


@dataclass
class TrainModelCKPT:
    train_node_name: str
    ckpt: ModelCKPT

    def __str__(self) -> str:
        return f"{self.ckpt.__str__()}@{self.train_node_name}"


API_TrainModelCKPT = "train_ckpt"

API_NodeSuicide = "node_suicide"
