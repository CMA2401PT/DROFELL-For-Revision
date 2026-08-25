import random
import torch
from typing import Any, Dict, TypeVar, List, Callable, Optional
from ..common.base_types import ModelDescribe
from operators.container.container import Container
from operators.assemble.decoders.real import DecodersImplement
import extend_asyncio as asyncio
from ..common.brain import OperateType, ListWithHook, Profile
from .memory import MemoryConfigBasic
from .basic_operate import BasicBrainOperates

OpPropType = TypeVar("OpPropType", str, int)
T = TypeVar("T")


# def select_multi_by_prop_count(
#     pendings: ListWithHook[OperateType],
#     read_fn: Callable[[OperateType], OpPropType]
# ):
#     pending_counts: Dict[OpPropType, int] = collections.defaultdict(lambda: 0)
#     ops = pendings.take_and_clear()

#     props = [read_fn(op) for op in ops]
#     for prop in props:
#         pending_counts[prop] += 1
#     selected_prop: OpPropType = sorted(pending_counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]
#     rest: List[OperateType] = [op for (op, p) in zip(ops, props) if p != selected_prop]
#     selected: List[OperateType] = [op for (op, p) in zip(ops, props) if p == selected_prop]
#     pendings.put_back(rest)
#     return selected_prop, selected


# def select_multi_by_prop_priority(
#     pendings: ListWithHook[OperateType],
#     read_fn: Callable[[OperateType], OpPropType],
#     max_concurrency: int,
# ):
#     ops = pendings.take_and_clear()
#     props = [read_fn(op) for op in ops]
#     selected_prop = sorted(props)[0]
#     rest: List[OperateType] = [op for (op, p) in zip(ops, props) if p != selected_prop]
#     selected: List[OperateType] = [op for (op, p) in zip(ops, props) if p == selected_prop]
#     if len(selected) > max_concurrency:
#         rest = selected[max_concurrency:]+rest
#         selected = selected[:max_concurrency]
#     pendings.put_back(rest)
#     return selected_prop, selected


# def select_single_by_prop_priority(pendings: ListWithHook[OperateType], read_fn: Callable[[OperateType], OpPropType]):
#     pri_prop, pri_idx = sorted([(read_fn(p), i) for i, p in enumerate(pendings.list)])[0]
#     select_op = pendings.pop(pri_idx)
#     return pri_prop, select_op


def select_multi_with_group(
    pendings: ListWithHook[OperateType],
    read_fn: Callable[[OperateType], OpPropType],
    max_concurrency: int,
):
    p: Optional[OpPropType] = None
    selected: List[OperateType] = []
    while len(selected) < max_concurrency and len(pendings):
        op = pendings.list[0]
        prop = read_fn(op)
        if p is None:
            p = prop
        elif p != prop:
            break
        pendings.pop(0)
        selected.append(op)
    assert p is not None
    return p, selected


class BasicBrain(BasicBrainOperates):
    def __init__(self,
                 models_path: str,
                 device: torch.device,
                 container: Container,
                 max_seq_len: int,
                 max_concurrency: int,
                 decoder_implement: DecodersImplement,
                 memory_config: Optional[MemoryConfigBasic] = None,
                 individual_train_thread: bool = False
                 ):
        if memory_config is None:
            memory_config = MemoryConfigBasic()
        if memory_config.root_memory == -1:
            memory_config.update_root_memory(device)
        super().__init__(
            models_path=models_path,
            device=device,
            container=container,
            memory_config=memory_config,
            max_seq_len=max_seq_len,
            max_concurrency=max_concurrency,
            decoder_implement=decoder_implement,
            individual_train_thread=individual_train_thread
        )
        self._loop = asyncio.get_running_loop()
        self._loop_task = self._loop.create_task(self.loop())

        self.operates_create_decoders_infer_binding.compare_fn = lambda op: str(op.binding)
        self.operates_create_decoders_train_binding.compare_fn = lambda op: str(op.binding)

        self.operates_put_user_model.compare_fn = lambda op: str(op.binding)
        self.operates_put_blocks.compare_fn = lambda op: str(self.decoders_infer_bindings[op.binding][0])
        self.operates_get_blocks.compare_fn = lambda op: str(self.decoders_infer_bindings[op.binding][0])
        self.operates_dump_train_states.compare_fn = lambda op: str(op.binding)
        self.operates_load_train_states.compare_fn = lambda op: str(op.binding)

        # infer
        self.operates_infer_tensor_to_embedding_tokens.compare_fn = lambda op: str(op.binding)
        self.operates_hidden_state_to_sampling_output.compare_fn = lambda op: str(op.binding)
        self.operates_do_generating.compare_fn = lambda op: str(self.decoders_infer_bindings[op.binding][0])
        self.operates_do_prefill.compare_fn = lambda op: (
            f"{self.decoders_infer_bindings[op.binding][0]}:"
            f"recovery={int(op.recovery_decode)}"
        )

        # train
        self.operates_train_tensor_to_embedding_tokens.compare_fn = lambda op: str(op.binding)
        self.operates_compute_logits_loss_and_grad.compare_fn = lambda op: str(op.binding)
        self.operates_forward.compare_fn = lambda op: str(op.binding)
        self.operates_backward.compare_fn = lambda op: str(op.binding)
        self.operates_weight_update.compare_fn = lambda op: str(op.binding)

    @property
    def memory(self):
        return self.basic_memory_config.root_memory, self.memory_modeling.env.pools[self.basic_memory_config.base_models_name]

    async def gen_profile(
        self,
        model_name: str,
        train_kwargs: Optional[Dict[str, Any]] = None,
        generating_kwargs: Optional[Dict[str, Any]] = None
    ) -> Profile:
        binding = self.make_model_decoders(model_name, (0, 1))
        name = ModelDescribe.new(model_name, start=-1, end=-1)
        perdecoder_memory = self.memory_modeling.base_models.get(name).item.memory
        fwd_tps, fwd_bwd_tps, generating_tps =\
            await self.get_underlay_decoders(binding).\
            gen_profile(self.container, train_kwargs=train_kwargs, generating_kwargs=generating_kwargs)
        self.remove_model_decoders(binding)
        return Profile(perdecoder_memory=perdecoder_memory, perdecoder_forward_tps=fwd_tps, perdecoder_frowarad_backward_tps=fwd_bwd_tps, perdecoder_generating_tps=generating_tps)

    async def loop(self):
        while True:
            await self.wait_task()
            # postpone until all activate tasks complete
            await self.container.step("", lambda: None)
            await self.barrier_maintainer.wait()
            if await self._priority_operate():
                continue
            await self._restricted_operate()
            assert len(self.memory_modeling.env.revert_stack) == 0

    async def _priority_operate(self):
        if False:
            pass
        elif (len(self.operates_put_blocks)):
            op = self.operates_put_blocks.pop(0)
            await self.put_blocks(op)
            return True
        elif len(self.operates_put_user_model):
            op = self.operates_put_user_model.pop(0)
            await self.put_user_model(op)
            return True
        elif (len(self.operates_get_blocks)):
            op = self.operates_get_blocks.pop(0)
            await self.get_blocks(op)
            return True
        elif len(self.operates_create_decoders_infer_binding):
            op = self.operates_create_decoders_infer_binding.pop(0)
            await self.create_decoders_infer_binding(op)
            return True
        # train-data movement
        elif len(self.operates_dump_train_states):
            i = random.randint(0, self.operates_dump_train_states.__len__()-1)
            op = self.operates_dump_train_states.list.pop(i)
            await self.dump_train_states(op)
            return True
        elif len(self.operates_load_train_states):
            i = random.randint(0, self.operates_load_train_states.__len__()-1)
            op = self.operates_load_train_states.pop(i)
            await self.load_train_states(op)
            return True

        elif len(self.operates_create_decoders_train_binding):
            i = random.randint(0, self.operates_create_decoders_train_binding.__len__()-1)
            op = self.operates_create_decoders_train_binding.pop(i)
            await self.create_decoders_train_binding(op)
            return True
        return False

    async def _restricted_operate(self):
        if False:
            pass
        # infer
        # infer-compute
        elif len(self.operates_infer_tensor_to_embedding_tokens):
            binding, ops = select_multi_with_group(
                pendings=self.operates_infer_tensor_to_embedding_tokens,
                read_fn=lambda op: str(op.binding),
                max_concurrency=self.max_concurrency,
            )
            await self.infer_tensor_to_embedding_tokens(binding, ops)

        elif len(self.operates_hidden_state_to_sampling_output):
            binding, ops = select_multi_with_group(
                pendings=self.operates_hidden_state_to_sampling_output,
                read_fn=lambda op: str(op.binding),
                max_concurrency=self.max_concurrency,
            )
            await self.hidden_state_to_sampling_output(binding, ops)

        elif (len(self.operates_do_prefill)):
            _, ops = select_multi_with_group(
                pendings=self.operates_do_prefill,
                read_fn=lambda op: (
                    f"{self.decoders_infer_bindings[op.binding][0]}:"
                    f"recovery={int(op.recovery_decode)}"
                ),
                max_concurrency=self.max_concurrency,
            )
            binding = self.decoders_infer_bindings[ops[0].binding][0]
            await self.do_prefill(binding, ops)

        elif (len(self.operates_do_generating)):
            binding, ops = select_multi_with_group(
                pendings=self.operates_do_generating,
                read_fn=lambda op: str(self.decoders_infer_bindings[op.binding][0]),
                max_concurrency=self.max_concurrency,
            )
            await self.do_generating(binding, ops)
        # infer-data movement

        # train
        # train-compute
        elif len(self.operates_train_tensor_to_embedding_tokens):
            i = random.randint(0, self.operates_train_tensor_to_embedding_tokens.__len__()-1)
            op = self.operates_train_tensor_to_embedding_tokens.pop(i)
            await self.train_tensor_to_embedding_tokens(op)

        elif len(self.operates_compute_logits_loss_and_grad):
            i = random.randint(0, self.operates_compute_logits_loss_and_grad.__len__()-1)
            op = self.operates_compute_logits_loss_and_grad.pop(i)
            await self.compute_logits_loss_and_grad(op)

        elif len(self.operates_forward):
            i = random.randint(0, self.operates_forward.__len__()-1)
            op = self.operates_forward.pop(i)
            await self.do_forward(op)

        elif len(self.operates_backward):
            i = random.randint(0, self.operates_backward.__len__()-1)
            op = self.operates_backward.pop(i)
            await self.do_backward(op)

        elif len(self.operates_weight_update):
            i = random.randint(0, self.operates_weight_update.__len__()-1)
            op = self.operates_weight_update.pop(i)
            await self.weight_update(op)
