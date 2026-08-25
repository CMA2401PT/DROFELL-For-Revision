from typing import List, Optional, Tuple
import torch

from .first_decoder import ForwardCtxFirstDecoder

from ...container import Container

from .trainable import TrainableUserModel

from ..decoders.real_decoders_group import DecodersGroup
from ...core.swappable.real import CanSwapObject, SwapableTensor


class ForwardCtxEachDecoder(CanSwapObject):
    def __init__(self,
                 position_idx: torch.Tensor,
                 input_lens: List[int],
                 ) -> None:
        self.user_model_ctx: CanSwapObject
        self.user_model_reproduce_seed: int

        # input data
        self.position_idx = SwapableTensor(position_idx)
        self.input_lens = input_lens
        self.per_decoder_input: List[Optional[
            Tuple[SwapableTensor, SwapableTensor]
        ]] = []

        self._swapped = False
        self._expect_device = True

    def downgrade(self):
        assert self.per_decoder_input[0] is not None
        first_input = self.per_decoder_input[0][0].tensor
        dctx = ForwardCtxFirstDecoder(self.position_idx.tensor, self.input_lens, first_input)
        dctx.user_model_reproduce_seed = self.user_model_reproduce_seed
        del self.position_idx
        for i in range(len(self.per_decoder_input)):
            self.per_decoder_input[i] = None
        return dctx

    @property
    def swapped(self) -> bool:
        return self._swapped

    @property
    def expect_device(self) -> bool:
        return self._expect_device

    def swap_out(self, stream: Optional[torch.cuda.Stream]) -> None:
        self.position_idx.swap_out(stream)
        for _decoder_i in range(len(self.per_decoder_input)):
            decoder_i = len(self.per_decoder_input)-_decoder_i-1
            decoder_input = self.per_decoder_input[decoder_i]
            if decoder_input is None:
                continue
            decoder_input[0].swap_out(stream)
            decoder_input[1].swap_out(stream)
        self.user_model_ctx.swap_out(stream)
        self._swapped = True
        self._expect_device = False

    def swap_in(self, device: torch.device, stream: Optional[torch.cuda.Stream]) -> None:
        self.position_idx.swap_in(device, stream)
        for _decoder_i in range(len(self.per_decoder_input)):
            decoder_i = len(self.per_decoder_input)-_decoder_i-1
            decoder_input = self.per_decoder_input[decoder_i]
            if decoder_input is None:
                continue
            decoder_input[0].swap_in(device, stream)
            decoder_input[1].swap_in(device, stream)
        self.user_model_ctx.swap_in(device, stream)
        self._swapped = True
        self._expect_device = False

    def set_decoder_input(self, decoder_i: int, input_hidden_state: torch.Tensor, input_kv_cache: torch.Tensor):
        assert input_hidden_state.is_leaf
        assert input_kv_cache.is_leaf
        while decoder_i >= len(self.per_decoder_input):
            self.per_decoder_input.append(None)
        self.per_decoder_input[decoder_i] = (SwapableTensor(input_hidden_state), SwapableTensor(input_kv_cache))

    def get_decoder_input(self, decoder_i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        decoder_input = self.per_decoder_input[decoder_i]
        assert decoder_input is not None
        return decoder_input[0].get_tensor(), decoder_input[1].get_tensor()

    def get_input_aux(self) -> Tuple[torch.Tensor, List[int]]:
        pos_idx: torch.Tensor = self.position_idx.get_tensor()
        return (pos_idx, self.input_lens)


async def forward_checkpoint_each(
        container: Container,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor, input_lens: List[int],
        user_model: TrainableUserModel, decoders: DecodersGroup) -> Tuple[torch.Tensor, ForwardCtxEachDecoder]:

    def user_model_fwd(dtype: torch.dtype):
        user_model_ctx, user_model_blocks, seed = user_model.train_forward(dtype=dtype, reproduce_seed=None)
        forward_ctx = ForwardCtxEachDecoder(position_ids, input_lens)
        forward_ctx.user_model_ctx = user_model_ctx
        return forward_ctx, user_model_blocks, seed
    forward_ctx, user_model_blocks, seed = await container.step("user_model_fwd", user_model_fwd, dtype=hidden_states.dtype)
    forward_ctx.user_model_reproduce_seed = seed
    lora_config = user_model.mode
    with container.mode("no_grad"):
        for decoder_i, decoder in enumerate(decoders.decoders):
            user_model_block = user_model_blocks[decoder_i]
            if (not user_model_block.is_leaf) or (user_model_block.requires_grad):
                user_model_block = user_model_block.detach()
            assert not hidden_states.requires_grad
            assert hidden_states.is_leaf
            forward_ctx.set_decoder_input(decoder_i, hidden_states, user_model_block)
            ptuning_kv_cache = None
            lora_hook = None

            if lora_config is None:
                ptuning_kv_cache = user_model_block
            else:
                lora_hook = lora_config.render_lora_hook(user_model_block)
            with container.stage(stage_name=f"decoder"):
                hidden_states = await decoder.multi_seqs_finetuning_forward(
                    container=container,
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    input_lens=input_lens,
                    ptuning_kv_cache=ptuning_kv_cache,
                    lora_hook=lora_hook
                )
    return hidden_states, forward_ctx


async def backward_checkpoint_each(
        container: Container,
        grad: torch.Tensor, ctx: ForwardCtxEachDecoder,
        user_model: TrainableUserModel, decoders: DecodersGroup) -> torch.Tensor:
    user_model_block_grad: List[torch.Tensor] = []
    position_ids, input_lens = ctx.get_input_aux()
    lora_config = user_model.mode
    for inv_decoder_i in range(len(decoders.decoders)):
        decoder_i = len(decoders.decoders) - inv_decoder_i - 1
        input_hidden_state, user_model_block = ctx.get_decoder_input(decoder_i)
        del ctx.per_decoder_input[decoder_i]
        input_hidden_state.requires_grad_(True)
        user_model_block.requires_grad_(True)
        decoder = decoders.decoders[decoder_i]

        ptuning_kv_cache = None
        lora_hook = None

        if lora_config is None:
            ptuning_kv_cache = user_model_block
        else:
            lora_hook = lora_config.render_lora_hook(user_model_block)

        async def after_stage_and_grad(od: torch.Tensor, rd: torch.Tensor):
            output_with_graph = await decoder.finetuning_forward_after_stage(container=container, hidden_states=od, residual=rd)
            await container.step("after_stage_bwd", output_with_graph.backward, gradient=grad)

        async def attention_and_grad(qd: torch.Tensor, kd: torch.Tensor, vd: torch.Tensor, rd: torch.Tensor):
            fwd, bwd = decoder.create_attention_forward_and_backward_enclosure(container, input_lens, ptuning_kv_cache)
            o, qkvs = await fwd(qd, kd, vd)
            od = o.detach()
            od.requires_grad_(True)
            await after_stage_and_grad(od, rd)
            assert od.grad is not None
            await bwd(qkvs,  od.grad)

        async def output_forward_and_grad():
            qkvr = await decoder.finetuning_forward_before_stage(container=container, hidden_states=input_hidden_state, position_ids=position_ids, lora_hook=lora_hook)
            qkvrd = [t.detach() for t in qkvr]
            _ = [t.requires_grad_(True) for t in qkvrd]
            await attention_and_grad(*qkvrd)
            await container.step("before_stage_bwd", torch.autograd.backward, tensors=qkvr, grad_tensors=[t.grad for t in qkvrd])

        await output_forward_and_grad()

        assert input_hidden_state.grad is not None
        assert user_model_block.grad is not None
        user_model_block_grad = [user_model_block.grad]+user_model_block_grad
        grad = input_hidden_state.grad.detach_()
    await container.step("user_model_backward", user_model.train_backward, ctx=ctx.user_model_ctx, grad=user_model_block_grad)
    return grad
