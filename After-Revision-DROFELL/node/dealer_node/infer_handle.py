from typing import Callable, List, Optional, Tuple

from ..shadow_node_state.infer_state_data import InferStateData
from ..compute_node.infer_meta import InferInit, PutBlocks

from .protocols import API_CreateInferNode, API_RemoveNode, CreateInferNode, MultiPathInferTask, RemoveNode
from orbit.defines import Packet
import extend_asyncio as asyncio
from utils.runtime_metrics import record_event

SendInferTask = Callable[[
    str, MultiPathInferTask,
    Optional[Callable[[], None]],
    Optional[Callable[[], None]],
    Optional[Callable[[], None]],
], None]


class InferNodeHandle:
    def __init__(
        self,
        send: Callable[[Packet, bool], None],
        send_infer_task: SendInferTask,
        dealer_node_name: str,
        retrieve: bool,
        init_config: InferInit,
        infer_node_name: str,
        recovery_id: Optional[str] = None,
        recovery_task_count: int = 0,
    ) -> None:
        self.send = send
        self.send_infer_task = send_infer_task
        self.init_config = init_config
        self.seq_id = self.init_config.seq_id
        # if infer_node_name == "":
        #     infer_node_name = f"infer-<{self.seq_id}>-<{init_config.models[0].split(':')[-1]}>-<{init_config.models[-1].split(':')[-1]}>"
        self.infer_node_name = infer_node_name
        self.retrieve = retrieve
        self.create_config = CreateInferNode(
            name=infer_node_name,
            config=init_config,
            block_retrieve=retrieve,
            recovery_id=recovery_id,
            recovery_task_count=recovery_task_count,
        )
        self.dealer_node_name = dealer_node_name
        if recovery_id is not None:
            record_event(
                "replacement_create_send",
                recovery_id=recovery_id,
                dealer_node=dealer_node_name,
                compute_node=infer_node_name,
                recovery_task_count=recovery_task_count,
            )
        self.send((self.dealer_node_name, API_CreateInferNode, self.create_config), False)
        if recovery_id is not None:
            record_event(
                "replacement_create_send_return",
                recovery_id=recovery_id,
                dealer_node=dealer_node_name,
                compute_node=infer_node_name,
                recovery_task_count=recovery_task_count,
                acceptance_semantics="send_returned_without_exception",
            )
        self.instance_data = InferStateData(self.init_config)
        self.recover_lock: asyncio.Lock = asyncio.Lock()
        self._dropped = False

    def __str__(self) -> str:
        return f"handle of infer_node=({self.infer_node_name}@{self.dealer_node_name})"

    def __repr__(self) -> str:
        return f"handle of infer_node=({self.infer_node_name}@{self.dealer_node_name})"

    @property
    def num_models(self):
        return self.init_config.num_models

    @property
    def models(self) -> List[Tuple[str, str]]:
        return self.init_config.models

    @property
    def base_models(self):
        return self.init_config.base_models

    @property
    def has_tail(self):
        return self.init_config.tail is not None

    @property
    def dropped(self):
        return self._dropped

    def update_result(self, infer_task: MultiPathInferTask):
        assert self.has_tail
        self.instance_data.update_by_output(infer_task.task)

    def update_recovery_result(self, infer_task: MultiPathInferTask):
        assert self.has_tail
        self.instance_data.update_by_recovery_output(infer_task.task)

    def dispatch(self, infer_task: MultiPathInferTask) -> bool:
        """
        return: dispatched
        """
        # this is possible since two recover could happen simultaneous
        # assert not self.dropped
        assert not self.dropped
        if infer_task.task.navi.finished:
            self.instance_data.update_by_output(infer_task.task)
            return False
        drop = self.instance_data.update_by_task(infer_task.task)
        if drop:
            return False
        # is ok if target is actual unreachable,
        # as recover will recover all dispatch status
        _start_trap: Optional[Callable[[], None]] = None
        _end_trap: Optional[Callable[[], None]] = None
        _send_cb: Optional[Callable[[], None]] = None

        if infer_task.detail_trace is not None:
            def start_trap():
                infer_task.record_step("start_trap")
            _start_trap = start_trap

            def end_trap():
                infer_task.record_step("end_trap")
            _end_trap = end_trap

            def send_cb():
                infer_task.record_step("leave")
            _send_cb = send_cb

        self.send_infer_task(
            self.dealer_node_name,
            infer_task,
            _start_trap,
            _end_trap,
            _send_cb
        )
        return True

    def put_blocks(self, blocks: PutBlocks):
        assert not self.dropped
        self.instance_data.update_by_block(blocks)

    def drop(self, drop_remote: bool):
        if self.dropped:
            return
        self._dropped = True
        # self.instance_data.clear()
        del self.instance_data
        if drop_remote:
            self.send((self.dealer_node_name, API_RemoveNode, RemoveNode(self.infer_node_name)), True)

    def __del__(self):
        if not self.dropped:
            self.drop(drop_remote=True)
