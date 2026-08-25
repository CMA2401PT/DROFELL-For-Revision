from typing import Callable, Optional
from uuid import uuid4

from ..shadow_node_state.train_state_data import TrainStateData

from ..compute_node.train_meta import TrainInit, ModelCKPT


from .protocols import API_CreateTrainNode, API_RemoveNode, API_TrainTask, CreateTrainNode, MultiPathTrainTask, RemoveNode
from orbit.defines import Packet
from utils.runtime_metrics import record_event


class TrainNodeHandle:
    def __init__(
        self,
        send: Callable[[Packet, bool], None],
        dealer_node_name: str,
        ckpt_period: int,
        init_config: TrainInit,
        train_node_name: str,
        control_id: Optional[str] = None,
        recovery_id: Optional[str] = None,
        recovery_task_count: int = 0,
    ) -> None:
        self.send = send
        self.init_config = init_config
        self.train_node_name = train_node_name
        self.ckpt_period = ckpt_period
        self.control_id = control_id
        self.recovery_id = recovery_id
        self.create_packet_id = uuid4().hex if recovery_id is not None else None
        self.create_config = CreateTrainNode(
            name=train_node_name,
            config=init_config,
            ckpt_period=ckpt_period,
            control_id=control_id,
            recovery_id=recovery_id,
            recovery_task_count=recovery_task_count,
            packet_id=self.create_packet_id,
        )
        self.dealer_node_name = dealer_node_name
        if recovery_id is not None:
            record_event(
                "recovery_packet_send_start",
                recovery_id=recovery_id,
                control_id=control_id,
                packet_id=self.create_packet_id,
                packet_kind="create_train_node",
                target_dealer=dealer_node_name,
                train_node_name=train_node_name,
            )
        self.send((self.dealer_node_name, API_CreateTrainNode, self.create_config), False)
        if recovery_id is not None:
            record_event(
                "recovery_packet_enqueue_accepted",
                recovery_id=recovery_id,
                control_id=control_id,
                packet_id=self.create_packet_id,
                packet_kind="create_train_node",
                target_dealer=dealer_node_name,
                train_node_name=train_node_name,
            )
        self.instance_data = TrainStateData(self.init_config)
        self._dropped = False

    def __str__(self) -> str:
        return f"handle of train_node=({self.train_node_name}@{self.dealer_node_name})"

    def __repr__(self) -> str:
        return f"handle of train_node=({self.train_node_name}@{self.dealer_node_name})"

    def num_steps(self, is_backward: bool):
        return self.init_config.num_steps(is_backward)

    @property
    def models(self):
        return self.init_config.models

    @property
    def base_models(self):
        return self.init_config.base_models

    @property
    def dropped(self):
        return self._dropped

    def dispatch(self, train_task: MultiPathTrainTask) -> bool:
        """
        return: dispatched
        """
        assert not self.dropped
        drop = self.instance_data.update_by_task(train_task.task)
        if drop:
            return False
        # is ok if target is actual unreachable,
        # as recover will recover all dispatch status

        if train_task.detail_trace is not None:
            train_task.record_step("leave")

        record_event(
            "train_packet_send_start",
            control_id=train_task.control_id,
            packet_id=train_task.packet_id,
            origin=train_task.origin,
            recovery_id=train_task.recovery_id,
            logical_handoff_id=train_task.logical_handoff_id,
            minibatch_id=train_task.task.minibatch_id,
            request_version=train_task.task.request_version,
            target_dealer=self.dealer_node_name,
            train_node_name=self.train_node_name,
        )
        self.send((self.dealer_node_name, API_TrainTask, train_task), True)
        record_event(
            "train_packet_enqueue_accepted",
            control_id=train_task.control_id,
            packet_id=train_task.packet_id,
            origin=train_task.origin,
            recovery_id=train_task.recovery_id,
            logical_handoff_id=train_task.logical_handoff_id,
            minibatch_id=train_task.task.minibatch_id,
            request_version=train_task.task.request_version,
            target_dealer=self.dealer_node_name,
            train_node_name=self.train_node_name,
        )
        return True

    def model_ckpt(self, ckpt: ModelCKPT):
        assert not self.dropped
        self.instance_data.update_by_ckpt(ckpt)

    def drop(self, drop_remote: bool):
        if self.dropped:
            return
        self._dropped = True
        # self.instance_data.clear()
        del self.instance_data
        if drop_remote:
            self.send((self.dealer_node_name, API_RemoveNode, RemoveNode(self.train_node_name)), True)

    def __del__(self):
        if not self.dropped:
            self.drop(drop_remote=True)
