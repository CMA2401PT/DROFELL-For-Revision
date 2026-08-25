from typing import Set
from ..dealer_node.protocols import API_CreateInferNode, API_CreateTrainNode, API_GroupedInferTask, API_InferPutBlocks, API_InferTask, API_NodeSuicide, API_RemoveNode, API_TrainModelCKPT, API_TrainTask, RemoveNode
from ..aux_node.protocols import API_Announce
from ..base import BaseNode
from .backbone_base import BackboneBaseNode
from .infer_backbone import BackboneInferNode
from .train_backbone import BackboneTrainNode


class BackboneNode:
    def __init__(
        self,
        base_node: BaseNode,
        max_search_redundancy: int = 0,
        path_gen_redundancy: int = 0,
    ) -> None:
        self.base = BackboneBaseNode(
            base_node=base_node,
            max_search_redundancy=max_search_redundancy,
            path_gen_redundancy=path_gen_redundancy
        )
        self.train_impl = BackboneTrainNode(self.base)
        self.infer_impl = BackboneInferNode(self.base)
        self.base.base.apis[API_Announce] = self.base.on_announce

        # train
        self.base.base.apis[API_CreateTrainNode] = self.train_impl.on_create_train_node
        self.base.base.apis[API_TrainTask] = self.train_impl.on_train_task
        self.base.base.apis[API_TrainModelCKPT] = self.train_impl.on_model_ckpt

        # infer
        self.base.base.apis[API_CreateInferNode] = self.infer_impl.on_create_infer_node
        self.base.base.apis[API_InferTask] = self.infer_impl.on_infer_task
        self.base.base.apis[API_InferPutBlocks] = self.infer_impl.on_put_blocks
        self.base.base.apis[API_GroupedInferTask] = self.infer_impl.on_grouped_infer_task

        # node change
        self.base.base.apis[API_RemoveNode] = self.on_remove_node
        self.base.base.apis[API_NodeSuicide] = self.on_node_suicide
        self.base.base.link_change_cbs.append(self.on_link_change)

    def on_remove_node(self, src: str, data: RemoveNode):
        # called by upper control node
        name = data.name
        if name in self.train_impl.train_controls.patched_train_tail_nodes:
            self.train_impl.train_controls.on_remove_patch_train_node(src, data)
            return
        if self.train_impl.has_node(name):
            self.train_impl.remove_virtual_node(name)
            return
        if self.infer_impl.has_node(name):
            self.infer_impl.remove_virtual_node(name)
            return

    def on_node_suicide(self, dealer_node: str, node_name: str) -> None:
        target = self.train_impl.train_controls.get_handle(node_name)
        if target is not None:
            self.base.logger.warn(f"{node_name} @ {dealer_node} suicide, self healing")
            train_control, handle = target
            self.train_impl.recover_handles_by_train_node(train_control=train_control, effected_handle=handle, report_dealer_node=dealer_node)
            return
        target = self.infer_impl.infer_controls.get_handle(node_name)
        if target is not None:
            self.base.logger.warn(f"{node_name} @ {dealer_node} suicide, self healing")
            infer_control, handle = target
            self.infer_impl.recover_handles_by_infer_node(infer_control=infer_control, effected_handle=handle, report_dealer_node=dealer_node)
            return

    def on_link_change(self, node: str, drop: bool, dsts: Set[str]):
        if not drop:
            return
        self.infer_impl.on_control_node_drop(node)
        self.infer_impl.on_dealer_node_drop(node)
        self.train_impl.on_control_node_drop(node)
        self.train_impl.on_dealer_node_drop(node)
