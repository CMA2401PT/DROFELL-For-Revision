
from typing import Any, List, Tuple

from operators import flags
from ...core.swappable.memory import CanSwapObjectMemory

from .trainable import TrainableUserMemoryModel
from ..decoders.memory import DecodersGroupMemoryModel
from operators.core.memory.env import Env
from ...core.memory.static import MemoryTensor
from ...core.memory.env import MemoryDict, MemoryHolder, MemoryNode


class ForwardCtxEachDecoderMemoryModeling(MemoryDict, CanSwapObjectMemory):
    def __init__(self,
                 position_idx: MemoryHolder[MemoryTensor],
                 input_lens: List[int],
                 env: Env) -> None:
        super().__init__(env)
        self.set("position_idx", position_idx)
        self.set("input_lens", input_lens)
        self.set("swapped", False)
        self._swapped = MemoryDict(env, default_pool="cpu")

    def set_decoder_input(self, decoder_i: int, input_hidden_state: MemoryHolder[MemoryTensor], input_kv_cache: MemoryHolder[MemoryTensor]):
        self.set(f"{decoder_i}-hiddenstate", input_hidden_state)
        self.set(f"{decoder_i}-input_kv_cache", input_kv_cache)

    @property
    def swapped(self) -> bool:
        return self.get("swapped")

    def swap_out(self) -> None:
        assert not self.swapped
        self.set("swapped", True)

        def move(name: Any):
            self._swapped.set(name, self.get(name))
            self.remove(name)
        for k, v in list(self.childrens.items()):
            if isinstance(v, MemoryNode):
                move(k)

    def swap_in(self) -> None:
        assert self.swapped
        self.set("swapped", False)

        def move(name: Any):
            self.set(name, self._swapped.get(name))
            self._swapped.remove(name)
        for k, v in list(self.childrens.items()):
            if isinstance(v, MemoryNode):
                move(k)

    def downgrad(self):
        position_idx = self.get("position_idx")
        input_lens = self.get("input_lens")
        hidden_states = self.get("0-hiddenstate")
        dctx = ForwardCtxFirstDecoderMemoryModeling(position_idx, input_lens, hidden_states, self.env)
        for c in list(self.childrens.keys()):
            self.remove(c)
        return dctx


class ForwardCtxFirstDecoderMemoryModeling(MemoryDict, CanSwapObjectMemory):
    def __init__(self,
                 position_idx: MemoryHolder[MemoryTensor],
                 input_lens: List[int],
                 hidden_states: MemoryHolder[MemoryTensor],
                 env: Env) -> None:
        super().__init__(env)
        self.set("position_idx", position_idx)
        self.set("hidden_states", hidden_states)
        self.set("input_lens", input_lens)
        self.set("swapped", False)
        self._swapped = MemoryDict(env, default_pool="cpu")

    @property
    def swapped(self) -> bool:
        return self.get("swapped")

    def swap_out(self) -> None:
        assert not self.swapped
        self.set("swapped", True)

        def move(name: str):
            self._swapped.set(name, self.get(name))
            self.remove(name)
        move("position_idx")
        move("hidden_states")

    def swap_in(self) -> None:
        assert self.swapped

        def move(name: str):
            self.set(name, self._swapped.get(name))
            self._swapped.remove(name)

        move("position_idx")
        move("hidden_states")


def forward_checkpoint_each(
    hidden_states: MemoryHolder[MemoryTensor],
    position_ids: MemoryHolder[MemoryTensor], input_lens: List[int],
    user_model: TrainableUserMemoryModel, decoders: DecodersGroupMemoryModel,
    env: Env,
) -> Tuple[MemoryHolder[MemoryTensor], ForwardCtxEachDecoderMemoryModeling]:
    user_model_ctx, user_model_blocks = user_model.train_forward(env, hidden_states.item.dtype)
    forward_ctx = ForwardCtxEachDecoderMemoryModeling(position_ids, input_lens, env)
    forward_ctx.set("user_model_ctx", user_model_ctx)
    for decoder_i, decoder in enumerate(decoders.decoders):
        user_model_block = user_model_blocks[decoder_i]
        forward_ctx.set_decoder_input(decoder_i, hidden_states, user_model_block)
        hidden_states = decoder.multi_seqs_finetuning_forward(env, hidden_states, position_ids, input_lens)
    return hidden_states, forward_ctx


def forward_checkpoint_first(
    hidden_states: MemoryHolder[MemoryTensor],
    position_ids: MemoryHolder[MemoryTensor], input_lens: List[int],
    user_model: TrainableUserMemoryModel, decoders: DecodersGroupMemoryModel,
    env: Env,
) -> Tuple[MemoryHolder[MemoryTensor], ForwardCtxFirstDecoderMemoryModeling]:
    user_model_ctx, user_model_blocks = user_model.train_forward(env, hidden_states.item.dtype)
    forward_ctx = ForwardCtxFirstDecoderMemoryModeling(position_ids, input_lens, hidden_states, env)
    forward_ctx.set("user_model_ctx", user_model_ctx)
    for decoder_i, decoder in enumerate(decoders.decoders):
        hidden_states = decoder.multi_seqs_finetuning_forward(env, hidden_states, position_ids, input_lens)
    return hidden_states, forward_ctx


def forward(
    hidden_states: MemoryHolder[MemoryTensor],
    position_ids: MemoryHolder[MemoryTensor], input_lens: List[int],
    user_model: TrainableUserMemoryModel, decoders: DecodersGroupMemoryModel,
    env: Env
) -> Tuple[MemoryHolder[MemoryTensor], Any]:
    assert not hidden_states.item.is_cpu
    assert not position_ids.item.is_cpu
    if flags.DECODERS_TRAIN_METHOD == "each_decoder":
        hidden_states, ctx = forward_checkpoint_each(hidden_states, position_ids, input_lens, user_model, decoders, env)
        return hidden_states, ctx
    elif flags.DECODERS_TRAIN_METHOD in ["first.grad", "first.2pass"]:
        hidden_states, ctx = forward_checkpoint_first(hidden_states, position_ids, input_lens, user_model, decoders, env)
        return hidden_states, ctx
    else:
        raise ValueError(flags.DECODERS_TRAIN_METHOD)
