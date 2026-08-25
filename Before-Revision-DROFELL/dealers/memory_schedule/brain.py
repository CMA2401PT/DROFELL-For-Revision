import random
import torch
from typing import Any, Dict, Tuple, TypeVar, List, Callable, Optional
from .restriction import MemoryConfig, MemoryConfigStatic, MemoryLeaf
from .swapper_core import SwapperCore
from operators.container.container import Container
import extend_asyncio as asyncio
from ..real_basic.basic_operate import BasicBrainOperates
from ..common.brain import OperateType, ListWithHook, Profile


OpPropType = TypeVar("OpPropType", str, int)
T = TypeVar("T")


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


class Brain(BasicBrainOperates):
    def __init__(self,
                 models_path: str,
                 device: torch.device,
                 container: Container,
                 memory_config: Optional[MemoryConfig] = None,
                 max_concurrency: int = 8,
                 individual_train_thread: bool = False
                 ):
        if memory_config is None:
            memory_config = MemoryConfigStatic()
        if memory_config.root_memory == -1:
            memory_config.update_root_memory(device)
        super().__init__(
            models_path=models_path,
            device=device,
            container=container,
            memory_config=memory_config,
            individual_train_thread=individual_train_thread
        )
        self.memory_config = memory_config
        assert self.memory_config.cache_engine_implement == "scalable"
        self.reinit_restriction()

        assert not self.restriction.oom
        self.max_concurrency: int = max_concurrency
        self._loop = asyncio.get_running_loop()

        self.swapper_core = SwapperCore(self)  # type:ignore

        self._loop_task = self._loop.create_task(self.loop())
        self._wait_new_operate_fut: Optional[asyncio.Future[None]] = None

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
        self.operates_do_prefill.compare_fn = lambda op: str(self.decoders_infer_bindings[op.binding][0])

        # train
        self.operates_train_tensor_to_embedding_tokens.compare_fn = lambda op: str(op.binding)
        self.operates_compute_logits_loss_and_grad.compare_fn = lambda op: str(op.binding)
        self.operates_forward.compare_fn = lambda op: str(op.binding)
        self.operates_backward.compare_fn = lambda op: str(op.binding)
        self.operates_weight_update.compare_fn = lambda op: str(op.binding)

        self.DEBUG_MEMORY = False

    def reinit_restriction(self):
        self.restriction = self.memory_config.restriction(self.memory_modeling.env)
        self.restriction.update(self.memory_config.root_memory)
        assert not self.restriction.oom
        # print("memory restriction and size:")
        # print(self.restriction)

    def make_model_head(self, base_model: str) -> str:
        ret = super().make_model_head(base_model)
        self.reinit_restriction()
        return ret

    def make_model_tail(self, base_model: str) -> str:
        ret = super().make_model_tail(base_model)
        self.reinit_restriction()
        return ret

    def make_model_decoders(self, base_model: str, decoders_range: Tuple[int, int]) -> Any:
        ret = super().make_model_decoders(base_model, decoders_range)
        self.reinit_restriction()
        return ret

    @property
    def memory(self):
        return self.basic_memory_config.root_memory, self.memory_modeling.env.pools[self.basic_memory_config.base_models_name]

    def query_extra_info(self, key: str) -> Any:
        if key == "detail_memory":
            return self.memory_modeling.env.detail_memory()
        if key == "detail_memory_restriction":
            return self.restriction.__str__()

    async def gen_profile(
        self,
        model_name: str,
        train_kwargs: Optional[Dict[str, Any]] = None,
        generating_kwargs: Optional[Dict[str, Any]] = None
    ) -> Profile:
        binding = self.make_model_decoders(model_name, (0, 1))
        memory1 = self.memory_modeling.base_models.get(binding).item.memory
        binding2 = self.make_model_decoders(model_name, (1, 3))
        memory2 = self.memory_modeling.base_models.get(binding2).item.memory
        perdecoder_memory = memory2-memory1
        fwd_tps, fwd_bwd_tps, generating_tps =\
            await self.get_underlay_decoders(binding).\
            gen_profile(self.container, train_kwargs=train_kwargs, generating_kwargs=generating_kwargs)
        # print(self.memory_modeling.base_models.childrens.keys())
        self.remove_model_decoders(binding)
        self.remove_model_decoders(binding2)
        return Profile(perdecoder_memory=perdecoder_memory, perdecoder_forward_tps=fwd_tps, perdecoder_frowarad_backward_tps=fwd_bwd_tps, perdecoder_generating_tps=generating_tps)

    def update_restriction(self):
        self.restriction.update(self.memory_config.root_memory)

    def update_and_check_oom(self):
        self.restriction.update(self.memory_config.root_memory)
        return self.restriction.oom

    def update_and_check_oom_items(self):
        self.restriction.update(self.memory_config.root_memory)
        return self.restriction.oom_items()

    def check_oom_with_delta(self, delta: Dict[str, int]) -> Dict[str, MemoryLeaf]:
        pools = self.memory_modeling.env.pools
        for k, d in delta.items():
            pools[k] = pools[k]+d
        self.restriction.update(self.memory_config.root_memory)
        oom_items = self.restriction.oom_items()
        for k, d in delta.items():
            pools[k] = pools[k]-d
        return oom_items

    async def loop(self):
        while True:
            assert len(self.memory_modeling.env.revert_stack) == 0
            assert not self.update_and_check_oom(), self.restriction
            await self.wait_task()
            # postpone until all activate tasks complete
            await self.container.step("", lambda: None)
            await self.barrier_maintainer.wait()
            if await self._operate():
                # print("MEMORY:")
                # print(self.memory_modeling.env.detail_memory())
                continue

            if self.DEBUG_MEMORY:
                self.update_and_check_oom()
                print(self.restriction)
                print(self.memory_modeling.env.detail_memory())
                await self._operate(debug="print")
                raise ValueError("DEBUG MEMORY: Insufficient memory for any pending operate, DEAD LOCK")
            else:
                print(self.restriction)
                print(self.memory_modeling.env.detail_memory())
                print("Insufficient memory for any pending operate, fall asleep")
                # reach memory limit
                # to avoid empty loop, wait for memory free
                # try to execute operate and check if memory free
                await self.wait_new_operate()
                print("resume: new operate comes")

    async def wait_new_operate(self):
        assert self._wait_new_operate_fut is None
        fut: asyncio.Future[None] = asyncio.Future()
        self._wait_new_operate_fut = fut
        await fut

    def _awake(self, op: OperateType) -> OperateType:
        op = super()._awake(op)
        # when new op comes, could free memory
        self._awake_new_operate()
        return op

    def _awake_new_operate(self):
        if self._wait_new_operate_fut is not None:
            fut = self._wait_new_operate_fut
            self._wait_new_operate_fut = None
            fut.set_result(None)

    def remove_model_head(self, binding: str):
        super().remove_model_head(binding)
        self.update_restriction()
        self._awake_new_operate()

    def remove_model_tail(self, binding: str):
        super().remove_model_tail(binding)
        self.update_restriction()
        self._awake_new_operate()

    def remove_model_decoders(self, binding: str):
        super().remove_model_decoders(binding)
        self.update_restriction()
        self._awake_new_operate()

    def remove_decoders_infer_binding(self, binding: str) -> None:
        super().remove_decoders_infer_binding(binding)
        self.update_restriction()
        self._awake_new_operate()

    def remove_decoders_train_binding(self, binding: str) -> None:
        super().remove_decoders_train_binding(binding)
        self.update_restriction()
        self._awake_new_operate()

    def swapout(self, target: str, **kwargs: Any) -> None:
        super().swapout(target=target, **kwargs)
        self.update_restriction()
        self._awake_new_operate()

    async def _operate(self, debug: Optional[str] = None):
        if (len(self.operates_put_blocks)):
            op = self.operates_put_blocks.pop(0)
            await self.put_blocks(op)
            return True
        if len(self.operates_put_user_model):
            op = self.operates_put_user_model.pop(0)
            await self.put_user_model(op)
            return True
        if (len(self.operates_get_blocks)):
            op = self.operates_get_blocks.pop(0)
            await self.get_blocks(op)
            return True
        # train-data movement
        if len(self.operates_dump_train_states):
            op = self.operates_dump_train_states.list.pop(0)
            await self.dump_train_states(op)
            return True
        if len(self.operates_load_train_states):
            op = self._select_operates_load_train_states(debug)
            if op is not None:
                await self.load_train_states(op, modeling=False)
                return True
        if len(self.operates_create_decoders_infer_binding):
            op = self._select_operates_create_decoders_infer_binding(debug)
            if op is not None:
                await self.create_decoders_infer_binding(op, modeling=False)
                return True

        if len(self.operates_create_decoders_train_binding):
            op = self._select_operates_create_decoders_train_binding(debug)
            if op is not None:
                await self.create_decoders_train_binding(op, modeling=False)
                return True

        # infer
        # infer-compute
        if len(self.operates_infer_tensor_to_embedding_tokens):
            binding, ops = select_multi_with_group(
                pendings=self.operates_infer_tensor_to_embedding_tokens,
                read_fn=lambda op: str(op.binding),
                max_concurrency=self.max_concurrency,
            )
            await self.infer_tensor_to_embedding_tokens(binding, ops)
            return True
        if len(self.operates_hidden_state_to_sampling_output):
            binding, ops = select_multi_with_group(
                pendings=self.operates_hidden_state_to_sampling_output,
                read_fn=lambda op: str(op.binding),
                max_concurrency=self.max_concurrency,
            )
            await self.hidden_state_to_sampling_output(binding, ops)
            return True
        if len(self.operates_do_prefill) or len(self.operates_do_generating):
            if await self.swapper_core.do_prefill_or_generating(debug):
                return True

        # train
        # train-compute
        if len(self.operates_train_tensor_to_embedding_tokens):
            sec = self._select_operates_train_tensor_to_embedding_tokens(debug)
            if sec is not None:
                op, bind = sec
                await self.train_tensor_to_embedding_tokens(op, bind)
                return True
        if len(self.operates_compute_logits_loss_and_grad):
            sec = self._select_operates_compute_logits_loss_and_grad(debug)
            if sec is not None:
                op, bind = sec
                await self.compute_logits_loss_and_grad(op, bind)
                return True
        # when preallocate=True, this will not allocate any memory
        if len(self.operates_weight_update):
            op = self.operates_weight_update.pop(0)
            await self.weight_update(op)
            return True
        if len(self.operates_backward):
            sec = self._select_operates_backward(debug)
            if sec is not None:
                op, bind = sec
                await self.do_backward(op, bind)
                return True
        if len(self.operates_forward):
            if await self.swapper_core.do_forward(debug):
                return True
        return False

    def _select_operates_load_train_states(self, debug: Optional[str] = None):
        for idx, op in enumerate(self.operates_load_train_states.list):
            self.memory_modeling.load_train_states(op)
            if self.update_and_check_oom():
                if debug is not None:
                    print(f"fail to operate load_train_states: {op}")
                    print(self.restriction)
                self.memory_modeling.revert_all()
            else:
                self.operates_load_train_states.list.pop(idx)
                return op

    def _select_operates_create_decoders_infer_binding(self, debug: Optional[str] = None):
        for idx, op in enumerate(self.operates_create_decoders_infer_binding.list):
            self.memory_modeling.create_decoders_infer_binding(op)
            if self.update_and_check_oom():
                if debug is not None:
                    print(f"fail to operate create_decoders_infer_binding: {op}")
                    print(self.restriction)
                self.memory_modeling.revert_all()
            else:
                self.operates_create_decoders_infer_binding.list.pop(idx)
                return op

    def _select_operates_create_decoders_train_binding(self, debug: Optional[str] = None):
        for idx, op in enumerate(self.operates_create_decoders_train_binding.list):
            self.memory_modeling.create_decoders_train_binding(op)
            if self.update_and_check_oom():
                if debug is not None:
                    print(f"fail to operate create_decoders_train_binding: {op}")
                    print(self.restriction)
                self.memory_modeling.revert_all()
            else:
                self.operates_create_decoders_train_binding.list.pop(idx)
                return op

    def rand_enumerate(self, seq: List[OperateType]):
        n = len(seq)
        for idx in random.sample(range(n), n):
            yield idx, seq[idx]

    def _select_operates_train_tensor_to_embedding_tokens(self, debug: Optional[str] = None):
        for idx, op in self.rand_enumerate(self.operates_train_tensor_to_embedding_tokens.list):
            vb, bind = self.memory_modeling.train_tensor_to_embedding_tokens(op)
            vb()
            if self.update_and_check_oom():
                if debug is not None:
                    print(f"fail to operate train_tensor_to_embedding_tokens: {op}")
                    print(self.restriction)
                self.memory_modeling.revert_all()
            else:
                self.operates_train_tensor_to_embedding_tokens.list.pop(idx)
                return op, bind

    def _select_operates_compute_logits_loss_and_grad(self, debug: Optional[str] = None):
        for idx, op in self.rand_enumerate(self.operates_compute_logits_loss_and_grad.list):
            vb, bind = self.memory_modeling.compute_logits_loss_and_grad(op)
            vb()
            if self.update_and_check_oom():
                if debug is not None:
                    print(f"fail to operate compute_logits_loss_and_grad: {op}")
                    print(self.restriction)
                self.memory_modeling.revert_all()
            else:
                self.operates_compute_logits_loss_and_grad.list.pop(idx)
                return op, bind

    def _select_operates_backward(self, debug: Optional[str] = None):
        for idx, op in self.rand_enumerate(self.operates_backward.list):
            vb, bind = self.memory_modeling.backward(op)
            vb()
            if self.update_and_check_oom():
                if debug is not None:
                    print(f"fail to operate backward: {op}")
                    print(self.restriction)
                self.memory_modeling.revert_all()
            else:
                self.operates_backward.list.pop(idx)
                return op, bind
