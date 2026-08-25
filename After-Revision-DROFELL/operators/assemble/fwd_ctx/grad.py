from typing import List, Optional, Tuple
import torch
from torch.cuda.streams import Stream
from ...container.container import Container
from .trainable import TrainableUserModel
from ..decoders.real_decoders_group import DecodersGroup
from ...core.swappable.real import CanSwapObject
from ...core.swappable.graph import HijackGraphWithMultiGradTensors


class ForwardGradCtx(CanSwapObject):
    def __init__(self) -> None:
        self.user_model_ctx: CanSwapObject
        self.graph: HijackGraphWithMultiGradTensors
        self._swapped = False
        self._expect_device = True

    def swap_out(self, stream: Optional[Stream]) -> None:
        self.user_model_ctx.swap_out(stream)
        self.graph.swap_out(stream)
        self._swapped = True
        self._expect_device = False

    def swap_in(self, device: torch.device, stream: Optional[Stream]) -> None:
        self.user_model_ctx.swap_in(device, stream)
        self.graph.swap_in(device, stream)
        self._swapped = True
        self._expect_device = False

    @property
    def swapped(self) -> bool:
        return self._swapped

    @property
    def expect_device(self) -> bool:
        return self._expect_device


async def forward_grad(
    container: Container,
    hidden_states: torch.Tensor, position_ids: torch.Tensor, input_lens: List[int],
    user_model: TrainableUserModel, decoders: DecodersGroup
) -> Tuple[torch.Tensor, ForwardGradCtx]:
    assert hidden_states.is_leaf
    user_model_ctx, ptuning_kv_caches, _ = \
        await container.step("user_model_fwd", user_model.train_forward, dtype=hidden_states.dtype, reproduce_seed=None)
    assert all([t.is_leaf for t in ptuning_kv_caches])
    # assert all(is_leaf) == any(is_leaf)  # all tensor must have same prop
    # is_connected_graph = not any(is_leaf)  # attach to graph
    lora_config = user_model.mode
    forward_ctx = ForwardGradCtx()
    forward_ctx.user_model_ctx = user_model_ctx
    graph = HijackGraphWithMultiGradTensors(
        tensors=[hidden_states.requires_grad_(True)]+[t.requires_grad_(True) for t in ptuning_kv_caches],
        addrs=decoders.non_free_addrs,
    )
    forward_ctx.graph = graph

    proxy_tensors = graph.tensors
    proxy_hidden_states = proxy_tensors[0]
    hidden_states = proxy_hidden_states
    user_model_blocks = proxy_tensors[1:]

    with graph.hijack_tensor():
        for decoder_i, decoder in enumerate(decoders.decoders):
            user_model_block = user_model_blocks[decoder_i]
            ptuning_kv_cache = None
            lora_hook = None

            if lora_config is None:
                ptuning_kv_cache = user_model_block
            else:
                lora_hook = lora_config.render_lora_hook(user_model_block)

            hidden_states = await decoder.multi_seqs_finetuning_forward(
                container=container,
                hidden_states=hidden_states,
                position_ids=position_ids,
                input_lens=input_lens,
                ptuning_kv_cache=ptuning_kv_cache,
                lora_hook=lora_hook
            )
        graph.set_output_t(hidden_states)

    return hidden_states.detach(), forward_ctx


async def backward_grad(
    container: Container,
    grad: torch.Tensor, ctx: ForwardGradCtx,
    user_model: TrainableUserModel,
    decoders: DecodersGroup
) -> torch.Tensor:
    user_model_ctx = ctx.user_model_ctx
    graph = ctx.graph
    await container.step("decoder_backward", graph.backward, grad=grad)
    grads = graph.grads
    hidden_states_grad = grads[0]
    kv_cache_grads = grads[1:]
    await container.step("user_model_backward", user_model.train_backward, ctx=user_model_ctx, grad=kv_cache_grads)
    return hidden_states_grad
