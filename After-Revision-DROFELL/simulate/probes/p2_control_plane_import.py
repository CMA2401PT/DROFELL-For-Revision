import torch
print("cuda available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
import extend_asyncio as ea
ea.config_impl("py", False)
# control-plane-only imports: no Brain, no operators backend
from node.base import BaseNode
from node.agent import AgentNode
from node.dealer_node.dealer_node import DealerNode
from node.shadow_node_state.infer_state_data import InferStateData
from node.compute_node.infer_meta import InferTask, InferInit, Navigate
from node.path_utils import gen_lowest_energy_paths, paths_to_keypoints
from node.infer.infer_ctrl import InferControl
from dealers.common.reorder import InferTaskReorderAndDropCtrl
print("control-plane imports OK (no GPU touched)")
# does the STM automaton work standalone?
c = InferTaskReorderAndDropCtrl(0)
print("ctrl step:", c.step)
print("gen_lowest_energy_paths:", gen_lowest_energy_paths([["A","B"],["A","C"],["C"]]))
