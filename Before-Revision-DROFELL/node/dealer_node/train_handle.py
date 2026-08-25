from typing import Callable

from ..shadow_node_state.train_state_data import TrainStateData

from ..compute_node.train_meta import TrainInit, ModelCKPT


from .protocols import API_CreateTrainNode, API_RemoveNode, API_TrainTask, CreateTrainNode, MultiPathTrainTask, RemoveNode
from orbit.defines import Packet


class TrainNodeHandle:
    def __init__(
        self,
        send: Callable[[Packet, bool], None],
        dealer_node_name: str,
        ckpt_period: int,
        init_config: TrainInit,
        train_node_name: str
    ) -> None:
        self.send = send
        self.init_config = init_config
        self.train_node_name = train_node_name
        self.ckpt_period = ckpt_period
        self.create_config = CreateTrainNode(
            name=train_node_name,
            config=init_config,
            ckpt_period=ckpt_period,
        )
        self.dealer_node_name = dealer_node_name
        self.send((self.dealer_node_name, API_CreateTrainNode, self.create_config), False)
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

        self.send((self.dealer_node_name, API_TrainTask, train_task), True)
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
