from typing import Set
from .train_client import ClientTrainNode
from .infer_client import ClientInferNode
from .client_base import ClientBaseNode
from ..base import BaseNode
from ..dealer_node.protocols import API_GroupedInferTask, API_InferTask, API_NodeSuicide


from typing import Set

from .client_base import ClientBaseNode

from ..base import BaseNode
from ..dealer_node.protocols import (API_NodeSuicide, API_TrainTask)


class ClientNode:
    def __init__(
        self,
        base_node: BaseNode,
        path_search_redundancy: int = 1
    ) -> None:
        self.base_node = base_node
        self.base = ClientBaseNode(
            base_node=base_node,
            path_search_redundancy=path_search_redundancy
        )
        self.inspect_fastest_path = self.base.inspect_fastest_path
        self.inspect_fastest_n_path = self.base.inspect_fastest_n_path
        self.infer_impl = ClientInferNode(self.base)
        self.create_infer_control = self.infer_impl.create_infer_control
        self.remove_infer_control = self.base.remove_infer_control
        self.train_impl = ClientTrainNode(self.base)
        self.create_train_control = self.train_impl.create_train_control
        self.remove_train_control = self.base.remove_train_control
        self.base.base.apis[API_NodeSuicide] = self.on_suicide
        self.base.base.apis[API_InferTask] = self.infer_impl.on_infer_task
        self.base.base.apis[API_GroupedInferTask] = self.infer_impl.on_grouped_infer_task
        self.base.base.apis[API_TrainTask] = self.train_impl.on_train_task
        self.base.base.link_change_cbs.append(self.on_link_change)

    def on_suicide(self, node: str, node_name: str):
        target = self.base.train_controls.get_handle(node_name)
        if target is not None:
            self.base.logger.warn(f"{node_name} @ {node} suicide, self healing")
            train_control, handle = target
            self.train_impl.recover_handles_by_train_node(train_control=train_control, effected_handle=handle, report_dealer_node=node)
            return
        target = self.base.infer_controls.get_handle(node_name)
        if target is not None:
            self.base.logger.warn(f"{node_name} @ {node} suicide, self healing")
            infer_control, handle = target
            self.infer_impl.recover_handles_by_infer_node(infer_control=infer_control, effected_handle=handle, report_dealer_node=node)
            return

    def on_link_change(self, node: str, drop: bool, dsts: Set[str]):
        if not drop:
            return
        targets = self.base.train_controls.get_handle_by_dealer(node)
        if targets is not None:
            # self.base.logger.error("LINK CHANGED! HEALING")
            for train_control, handles in targets:
                # train_control.breaker_enable()
                self.base.logger.warn(f"{node} dropped, {train_control.ctrl_id} effected, self healing")
                self.train_impl.recover_handles_by_dealer_node(train_control=train_control, effected_handles=handles, dropped_node=node)

        targets = self.base.infer_controls.get_handle_by_dealer(node)
        if targets is not None:
            for infer_control, handles in targets:
                self.base.logger.warn(f"{node} dropped, {infer_control.ctrl_id} effected, self healing")
                self.infer_impl.recover_handles_by_dealer_node(infer_control=infer_control, effected_handles=handles, dropped_node=node)
