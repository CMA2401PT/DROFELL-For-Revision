import pickle
from typing import Any, List, Optional, Tuple
import torch
from torch import nn

from ..block.common import BlockInfo

from ...assemble.fwd_ctx.trainable import TrainableUserModel
from ... import flags
from .common import PtuningConfig
from ..swappable.real import CanSwapObject, NoSwapAny
from ..swappable.graph import HijackGraph


class Ptuning(nn.Module, TrainableUserModel):
    def __init__(self, config: PtuningConfig):
        nn.Module.__init__(self)
        self.cfg = config
        self.embedding = torch.nn.Embedding(
            self.cfg.prefix_len, self.cfg.combined_output_shape)
        self.dropout = nn.Dropout(self.cfg.dropout_p)
        self.device = torch.zeros((0,)).device
        self.seed_inc = 0
        self.prefix_tokens = torch.arange(self.cfg.prefix_len).long().unsqueeze(0).to(self.device)
        self.output_shape = (1, self.cfg.prefix_len, self.cfg.combined_output_shape)
        self.non_free_addrs = HijackGraph.get_module_address(self)

    def _get_combined_prompt(self, prefix: torch.Tensor):
        combined_ptuning_kv_cache = self.embedding(prefix)
        return combined_ptuning_kv_cache

    @property
    def build_prompt_str(self):
        return self.cfg.build_prompt_str

    def get_inference_ptuning_kv_cache(self, dtype: torch.dtype) -> List[torch.Tensor]:
        with torch.no_grad():
            ptuning_kv_cache = self._get_combined_prompt(
                self.prefix_tokens).to(dtype)
            ptuning_kv_cache = ptuning_kv_cache.view(
                self.cfg.prefix_len,  # 0
                self.cfg.num_layers * 2,  # 1
                self.cfg.num_key_value_heads,  # 2
                self.cfg.head_dim,  # 3
            )
            return ptuning_kv_cache.permute([1, 0, 2, 3]).split(2)

    def get_inference_ptuning_kv_cache_with_convert(self, dtype: torch.dtype) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        layer_kvs = self.get_inference_ptuning_kv_cache(dtype)
        block_info = BlockInfo(self.cfg.head_dim, num_key_value_heads=self.cfg.num_key_value_heads, dtype=dtype)
        ptuning_blocks: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
        for lk, lv in layer_kvs:
            lk, lv = block_info.tensor_kv_to_block_kv(
                key_states=lk, val_states=lv
            )
            ptuning_blocks.append([
                (k, v)
                for (k, v) in zip(lk, lv)
            ])
        return ptuning_blocks

    def to_per_layer_serialized(self) -> List[bytes]:
        state_dict = self.state_dict()
        assert len(state_dict) == 1
        weight = state_dict["embedding.weight"]
        flatten_size = weight.shape[1]
        # print(stream)
        split_weights = [
            {"embedding.weight": w.detach().to(device="cpu", non_blocking=True)}
            for w in weight.split(flatten_size // self.num_layers, 1)
        ]
        torch.cuda.current_stream().synchronize()
        return [pickle.dumps(w) for w in split_weights]

    def from_pre_layer_serialized(self, split_weights_serialized: List[bytes]):
        split_weights_read = [pickle.loads(s) for s in split_weights_serialized]
        split_weight_tensor = [w["embedding.weight"] for w in split_weights_read]
        split_weight_tensor = [
            w.detach().to(device=self.device, non_blocking=True)
            for w in split_weight_tensor
        ]
        state_dict = {"embedding.weight": torch.cat(split_weight_tensor, dim=1)}
        self.load_state_dict(state_dict)

    def train_forward(self, dtype: torch.dtype, reproduce_seed: Optional[int]) -> Tuple[CanSwapObject, List[torch.Tensor], int]:
        if flags.USER_MODEL_ENABLE_CHECKPOTIN:
            return self._train_forward_no_grad(dtype, reproduce_seed)
        else:
            return self._train_forward_with_grad(dtype, reproduce_seed)

    def train_backward(self, ctx: CanSwapObject, grad: List[torch.Tensor]) -> None:
        if flags.USER_MODEL_ENABLE_CHECKPOTIN:
            assert isinstance(ctx, NoSwapAny)
            return self._train_backward_no_grad(ctx, grad)  # type: ignore
        else:
            assert isinstance(ctx, HijackGraph)
            return self._train_backward_with_grad(ctx, grad)

    def _train_forward_with_grad(self, dtype: torch.dtype, reproduce_seed: Optional[int]) -> Tuple[HijackGraph, List[torch.Tensor], int]:
        if reproduce_seed is None:
            self.seed_inc += 1
            reproduce_seed = self.seed_inc

        graph = HijackGraph(self.non_free_addrs)
        with graph.hijack_tensor():
            ptuning_kv_cache = self._get_combined_prompt(self.prefix_tokens).to(dtype)
            torch.manual_seed(reproduce_seed)
            ptuning_kv_cache: torch.Tensor = self.dropout(ptuning_kv_cache)
            graph.set_output_t(ptuning_kv_cache)
        ptuning_kv_cache = ptuning_kv_cache.detach().requires_grad_(False)
        ptuning_kv_cache = ptuning_kv_cache.view(
            self.cfg.prefix_len,  # 0
            self.cfg.num_layers * 2,  # 1
            self.cfg.num_key_value_heads,  # 2
            self.cfg.head_dim,  # 3
        )
        with_grad = ptuning_kv_cache.permute([1, 0, 2, 3])
        return graph, with_grad.split(2), reproduce_seed

    def _train_backward_with_grad(self, ctx: HijackGraph, grad: List[torch.Tensor]) -> None:
        grad_concatenated: torch.Tensor = torch.cat(grad).permute([1, 0, 2, 3]).contiguous()
        grad_concatenated = grad_concatenated.view(self.output_shape)
        ctx.backward(grad_concatenated)

    def _train_forward_no_grad(self, dtype: torch.dtype, reproduce_seed: Optional[int]) -> Tuple[NoSwapAny[Tuple[int, torch.dtype]], List[torch.Tensor], int]:
        if reproduce_seed is None:
            self.seed_inc += 1
            reproduce_seed = self.seed_inc
        forward_ctx = NoSwapAny((reproduce_seed, dtype))
        with torch.no_grad():
            ptuning_kv_cache = self._get_combined_prompt(self.prefix_tokens).to(dtype)
            torch.manual_seed(reproduce_seed)
            ptuning_kv_cache: torch.Tensor = self.dropout(ptuning_kv_cache)
            ptuning_kv_cache = ptuning_kv_cache.view(
                self.cfg.prefix_len,  # 0
                self.cfg.num_layers * 2,  # 1
                self.cfg.num_key_value_heads,  # 2
                self.cfg.head_dim,  # 3
            )
            out = ptuning_kv_cache.permute([1, 0, 2, 3])
            out.requires_grad_(False)
        return forward_ctx, out.split(2), reproduce_seed

    def _train_backward_no_grad(self, ctx: NoSwapAny[Tuple[int, torch.dtype]], grad: List[torch.Tensor]) -> None:
        seed, dtype = ctx.data
        ptuning_kv_cache = self._get_combined_prompt(self.prefix_tokens).to(dtype)
        torch.manual_seed(seed)
        ptuning_kv_cache = self.dropout(ptuning_kv_cache)
        ptuning_kv_cache = ptuning_kv_cache.view(
            self.cfg.prefix_len,  # 0
            self.cfg.num_layers * 2,  # 1
            self.cfg.num_key_value_heads,  # 2
            self.cfg.head_dim,  # 3
        )
        out_with_grad = ptuning_kv_cache.permute([1, 0, 2, 3])
        grad_concatenated: torch.Tensor = torch.cat(grad)
        out_with_grad.backward(gradient=grad_concatenated)

    def load_from_file(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        assert self.cfg.build_prompt_str != ""

    def save_to_file(self, path: str) -> None:
        # os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path)

    def reinit(self, **kwargs: Any):
        from torch.nn.init import xavier_uniform_
        for parameter in self.parameters():
            xavier_uniform_(parameter)

    @property
    def mode(self):
        return None
