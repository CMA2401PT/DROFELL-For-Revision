import pickle
from typing import Any, List, Optional, Tuple
import torch
from torch import nn

from . import LoraRenderConfig

from ...assemble.fwd_ctx.trainable import TrainableUserModel
from .common import LoraConfig
from ..swappable.real import CanSwapObject, NoSwapAny
from ..swappable.graph import HijackGraph


class Lora(nn.Module, TrainableUserModel):
    def __init__(self, config: LoraConfig):
        nn.Module.__init__(self)
        self.cfg = config
        self.lora_block = torch.nn.Parameter(
            torch.zeros(self.cfg.num_layers, *self.cfg.lora_block_shape))
        self.dropout = nn.Dropout(self.cfg.dropout_p)
        self.device = torch.zeros((0,)).device
        self.seed_inc = 0
        self.non_free_addrs = HijackGraph.get_module_address(self)

    @property
    def build_prompt_str(self):
        return self.cfg.build_prompt_str

    def get_inference_lora_blocks(self, dtype: torch.dtype) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        with torch.inference_mode():
            raw_blocks = [self.lora_block[i].to(dtype).detach() for i in range(self.cfg.num_layers)]
            return [
                self.cfg.index.single_block_to_saperate_blocks(b)
                for b in raw_blocks
            ]

    def to_per_layer_serialized(self) -> List[bytes]:
        state_dict = self.state_dict()
        assert len(state_dict) == 1
        weight = state_dict["lora_block"]
        flatten_size = weight.shape[1]
        # print(stream)
        split_weights = [
            {"lora_block": w.detach().to(device="cpu", non_blocking=True)}
            for w in weight.split(flatten_size // self.num_layers, 1)
        ]
        torch.cuda.current_stream().synchronize()
        return [pickle.dumps(w) for w in split_weights]

    def from_pre_layer_serialized(self, split_weights_serialized: List[bytes]):
        split_weights_read = [pickle.loads(s) for s in split_weights_serialized]
        split_weight_tensor = [w["lora_block"] for w in split_weights_read]
        split_weight_tensor = [
            w.detach().to(device=self.device, non_blocking=True)
            for w in split_weight_tensor
        ]
        state_dict = {"lora_block": torch.cat(split_weight_tensor, dim=1)}
        self.load_state_dict(state_dict)

    def train_forward(self, dtype: torch.dtype, reproduce_seed: Optional[int]) -> Tuple[CanSwapObject, List[torch.Tensor], int]:
        if reproduce_seed is None:
            self.seed_inc += 1
            reproduce_seed = self.seed_inc
        layers_lora = [self.lora_block[i].to(dtype) for i in range(self.cfg.num_layers)]
        return NoSwapAny((dtype, reproduce_seed)), layers_lora, reproduce_seed

    def train_backward(self, ctx: CanSwapObject, grad: List[torch.Tensor]) -> None:
        for i in range(self.cfg.num_layers):
            self.lora_block[i].backward(grad[i])
        # all_grad = torch.stack(grad)
        # self.lora_block.backward(gradient=all_grad)

    def load_from_file(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        assert self.cfg.build_prompt_str != ""

    def save_to_file(self, path: str) -> None:
        # os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path)

    @property
    def mode(self) -> Optional[LoraRenderConfig]:
        return self.cfg

    def reinit(self, **kwargs: Any):
        with torch.no_grad():
            ab = self.cfg.get_q(self.lora_block)
            if ab is not None:
                a, _ = ab
                a.copy_(torch.rand_like(a)*0.01)
            ab = self.cfg.get_k(self.lora_block)
            if ab is not None:
                a, _ = ab
                a.copy_(torch.rand_like(a)*0.01)
            ab = self.cfg.get_v(self.lora_block)
            if ab is not None:
                a, _ = ab
                a.copy_(torch.rand_like(a)*0.01)
