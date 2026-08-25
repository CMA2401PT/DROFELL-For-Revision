from typing import List, Tuple
import torch

from operators import flags
from ...container.container import Container
from ...core.swappable.real import CanSwapObject
from ..decoders.real_decoders_group import DecodersGroup
from .trainable import TrainableUserModel
from .first_decoder import ForwardCtxFirstDecoder, backward_checkpoint_first_2pass, backward_checkpoint_first_grad, forward_checkpoint_first
from .grad import ForwardGradCtx, backward_grad, forward_grad

from .each_decoder import ForwardCtxEachDecoder, forward_checkpoint_each, backward_checkpoint_each


async def forward(
        container: Container,
        hidden_states: torch.Tensor, position_ids: torch.Tensor, input_lens: List[int],
        user_model: TrainableUserModel, decoders: DecodersGroup) -> Tuple[torch.Tensor, CanSwapObject]:
    assert not hidden_states.is_cpu
    assert not position_ids.is_cpu
    if flags.DECODERS_TRAIN_METHOD == "each_decoder":
        hidden_states, ctx = await forward_checkpoint_each(container, hidden_states, position_ids, input_lens, user_model, decoders)
        return hidden_states, ctx
    elif flags.DECODERS_TRAIN_METHOD == "grad":
        hidden_states, ctx = await forward_grad(container, hidden_states, position_ids, input_lens, user_model, decoders)
        return hidden_states, ctx
    elif flags.DECODERS_TRAIN_METHOD in ["first.grad", "first.2pass"]:
        hidden_states, ctx = await forward_checkpoint_first(container, hidden_states, position_ids, input_lens, user_model, decoders)
        return hidden_states, ctx
    else:
        raise ValueError(flags.DECODERS_TRAIN_METHOD)


async def backward(
        container: Container,
        grad: torch.Tensor, ctx: CanSwapObject,
        user_model: TrainableUserModel, decoders: DecodersGroup) -> torch.Tensor:
    if flags.DECODERS_TRAIN_METHOD == "each_decoder":
        if isinstance(ctx, ForwardCtxEachDecoder):
            return await backward_checkpoint_each(container, grad, ctx, user_model, decoders)
        elif isinstance(ctx, ForwardCtxFirstDecoder):
            if flags.DECODERS_TRAIN_DOWNGRAD_METHOD == "first.grad":
                return await backward_checkpoint_first_grad(container, grad, ctx, user_model, decoders)
            else:
                return await backward_checkpoint_first_2pass(container, grad, ctx, user_model, decoders)
        else:
            raise ValueError(flags.DECODERS_TRAIN_DOWNGRAD_METHOD)
    elif flags.DECODERS_TRAIN_METHOD == "grad":
        assert isinstance(ctx, ForwardGradCtx)
        return await backward_grad(container, grad, ctx, user_model, decoders)
    elif flags.DECODERS_TRAIN_METHOD == "first.grad":
        assert isinstance(ctx, ForwardCtxFirstDecoder)
        return await backward_checkpoint_first_grad(container, grad, ctx, user_model, decoders)
    elif flags.DECODERS_TRAIN_METHOD == "first.2pass":
        assert isinstance(ctx, ForwardCtxFirstDecoder)
        return await backward_checkpoint_first_2pass(container, grad, ctx, user_model, decoders)
    else:
        raise ValueError(flags.DECODERS_TRAIN_METHOD)
