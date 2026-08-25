from typing import List, Optional, Tuple
import torch

from ...container.container import Container
from .trainable import TrainableUserModel
from ..decoders.real_decoders_group import DecodersGroup
from ...core.swappable.real import CanSwapObject, SwapableTensor


class ForwardCtxFirstDecoder(CanSwapObject):
    def __init__(self,
                 position_idx: torch.Tensor,
                 input_lens: List[int],
                 hidden_states: torch.Tensor,
                 ) -> None:
        self.user_model_reproduce_seed: int

        # input data
        self.hidden_states: SwapableTensor = SwapableTensor(hidden_states)
        self.position_idx = SwapableTensor(position_idx)
        self.input_lens = input_lens

        self._swapped = False
        self._expect_device = True

    def swap_out(self, stream: Optional[torch.cuda.Stream]) -> None:
        self.position_idx.swap_out(stream)
        self.hidden_states.swap_out(stream)
        self._swapped = True
        self._expect_device = False

    def swap_in(self, device: torch.device, stream: Optional[torch.cuda.Stream]) -> None:
        self.position_idx.swap_in(device, stream)
        self.hidden_states.swap_in(device, stream)
        self._swapped = True
        self._expect_device = False

    @property
    def swapped(self) -> bool:
        return self._swapped

    @property
    def expect_device(self) -> bool:
        return self._expect_device


async def forward_checkpoint_first(container: Container, hidden_states: torch.Tensor, position_ids: torch.Tensor, input_lens: List[int],
                                   user_model: TrainableUserModel, decoders: DecodersGroup) -> Tuple[torch.Tensor, ForwardCtxFirstDecoder]:
    lora_config = user_model.mode
    with container.mode("no_grad"):
        def user_model_fwd(dtype: torch.dtype):
            _, user_model_blocks, seed = user_model.train_forward(dtype, None)
            return user_model_blocks, seed
        user_model_blocks, seed = await container.step("user_model_fwd", user_model_fwd, dtype=hidden_states.dtype)
        forward_ctx = ForwardCtxFirstDecoder(position_ids, input_lens, hidden_states)
        forward_ctx.user_model_reproduce_seed = seed
        for decoder_i, decoder in enumerate(decoders.decoders):
            user_model_block = user_model_blocks[decoder_i]
            assert not hidden_states.requires_grad
            assert hidden_states.is_leaf

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
                lora_hook=lora_hook,
            )

    return hidden_states, forward_ctx


async def backward_checkpoint_first_grad(
        container: Container,
        grad: torch.Tensor, ctx: ForwardCtxFirstDecoder,
        user_model: TrainableUserModel, decoders: DecodersGroup
) -> torch.Tensor:
    input_hidden = ctx.hidden_states.get_tensor()
    input_hidden.requires_grad_(True)
    hidden_states = input_hidden
    position_ids = ctx.position_idx.get_tensor()
    input_lens = ctx.input_lens
    seed = ctx.user_model_reproduce_seed
    lora_config = user_model.mode

    user_model_ctx, user_model_blocks, _ = \
        await container.step("user_model_fwd", user_model.train_forward, dtype=hidden_states.dtype, reproduce_seed=seed)
    is_leaf = [t.is_leaf for t in user_model_blocks]
    assert all(is_leaf) == any(is_leaf)  # all tensor must have same prop
    is_connected_graph = not any(is_leaf)  # attach to graph
    if not is_connected_graph:  # detach graph
        for t in user_model_blocks:
            t.requires_grad_(True)

    for decoder_i, decoder in enumerate(decoders.decoders):
        user_model_block = user_model_blocks[decoder_i]
        if user_model_block.is_leaf:
            user_model_block.requires_grad_(True)

        ptuning_kv_cache = None
        lora_hook = None
        if lora_config is None:
            ptuning_kv_cache = user_model_block
        else:
            with torch.enable_grad():
                lora_hook = lora_config.render_lora_hook(user_model_block)
        hidden_states = await decoder.multi_seqs_finetuning_forward(
            container=container,
            hidden_states=hidden_states,
            position_ids=position_ids,
            input_lens=input_lens,
            ptuning_kv_cache=ptuning_kv_cache,
            lora_hook=lora_hook
        )
    await container.step("decoder_backward", hidden_states.backward, gradient=grad)
    assert input_hidden.grad is not None
    grad = input_hidden.grad
    if not is_connected_graph:
        grads: List[torch.Tensor] = []
        for t in user_model_blocks:
            assert t.grad is not None
            grads.append(t.grad)
        await container.step("user_model_backward", user_model.train_backward, ctx=user_model_ctx, grad=grads)
    return grad


async def backward_checkpoint_first_2pass(
        container: Container,
        grad: torch.Tensor, ctx: ForwardCtxFirstDecoder,
        user_model: TrainableUserModel, decoders: DecodersGroup
) -> torch.Tensor:
    input_hidden = ctx.hidden_states.get_tensor()
    hidden_states = input_hidden
    position_ids = ctx.position_idx.get_tensor()
    input_lens = ctx.input_lens
    seed = ctx.user_model_reproduce_seed
    lora_config = user_model.mode
    user_model_ctx, user_model_blocks, _ = \
        await container.step("user_model_fwd", user_model.train_forward, dtype=hidden_states.dtype, reproduce_seed=seed)
    user_model_blocks = list(user_model_blocks)
    each_decoder_input: List[torch.Tensor] = []

    with container.mode("no_grad"):
        for decoder_i, decoder in enumerate(decoders.decoders):
            user_model_block = user_model_blocks[decoder_i]
            if (not user_model_block.is_leaf) or (user_model_block.requires_grad):
                user_model_block = user_model_block.detach()
                user_model_blocks[decoder_i] = user_model_block
            assert not hidden_states.requires_grad
            assert hidden_states.is_leaf
            each_decoder_input.append(hidden_states)

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
                lora_hook=lora_hook,
            )

    ptunings_grad: List[torch.Tensor] = []

    for inv_decoder_i in range(len(decoders.decoders)):
        decoder_i = len(decoders.decoders) - inv_decoder_i - 1
        input_hidden_state = each_decoder_input[decoder_i]
        user_model_block = user_model_blocks[decoder_i]
        del each_decoder_input[decoder_i]
        del user_model_blocks[decoder_i]
        input_hidden_state.requires_grad_(True)
        user_model_block.requires_grad_(True)
        decoder = decoders.decoders[decoder_i]
        ptuning_kv_cache = None
        lora_hook = None

        if lora_config is None:
            ptuning_kv_cache = user_model_block
        else:
            with torch.enable_grad():
                lora_hook = lora_config.render_lora_hook(user_model_block)
        output_with_graph = await decoder.multi_seqs_finetuning_forward(
            container=container,
            hidden_states=input_hidden_state,
            position_ids=position_ids,
            input_lens=input_lens,
            ptuning_kv_cache=ptuning_kv_cache,
            lora_hook=lora_hook,
        )
        await container.step("decoder_backward", output_with_graph.backward, gradient=grad)
        assert input_hidden_state.grad is not None
        assert user_model_block.grad is not None
        ptunings_grad = [user_model_block.grad]+ptunings_grad
        grad = input_hidden_state.grad.detach_()

    await container.step("user_model_backward", user_model.train_backward, ctx=user_model_ctx, grad=ptunings_grad)
    return grad
