from typing import Optional, Union, cast
from operators import flags
from operators.assemble.fwd_ctx.memory import ForwardCtxEachDecoderMemoryModeling, forward
from operators.cache_engine.common import MemoryCacheEngine
from operators.cache_engine.memory import ScalableCacheEngineMemoryModel, VllmFixCacheEngineMemoryModel
from operators.core.head.memory import ModelHeadMemoryModel
from operators.core.lora.memory import LoraMemoryModel
from operators.core.lora.common import LoraConfig
from operators.core.memory.static import MemoryFloat16, MemoryInt32, MemoryItem, MemoryTensor
from operators.core.ptuning.memory import PtuningMemoryModel
from operators.core.ptuning.common import PtuningConfig
from operators.core.swappable.memory import CanSwapObjectMemory
from operators.core.tail.memory import ModelTailMemoryModel
from ..common.brain import *
from ..common.base_types import *

from operators.assemble.decoders.memory import DecodersGroupMemoryModel, make_decoders_memory_model
from operators.core.memory.env import Env, MemoryDict, MemoryHolder


class TrainBindingDataMemoryModel(MemoryDict, MemoryItem):
    def __init__(self,
                 decoders: DecodersGroupMemoryModel,
                 user_model_config: Union[PtuningConfig, LoraConfig],
                 optimizer_config: OptimizerConfig,
                 scheduler_config: Optional[SchedulerConfig],
                 data_init_method: Optional[UserModelDataInitConfig],
                 env: Env,
                 preallocate: bool) -> None:
        MemoryDict.__init__(self, env)
        self.decoders = decoders
        self.user_model_config, self.user_model = self._init_user_model(self.env, user_model_config)
        # self.set("user_model_grad", self.user_model)
        self.preallocate = preallocate
        if self.preallocate:
            self._ensure_optimizer_param(env)
            self.user_model.ensure_backward(env)

    def _ensure_optimizer_param(self, env: Env):
        if not "optimizer_param" in self.user_model.childrens:
            self.user_model.set("optimizer_param", env.trace(MemoryTensor((1,))))
            for k, t in self.user_model.trainable_parameters.items():
                self.user_model.set(f"optimizer_param_{k}_0", env.trace(MemoryTensor.like(t)))
                self.user_model.set(f"optimizer_param_{k}_1", env.trace(MemoryTensor.like(t)))

    def weight_update(self, env: Env):
        if not self.preallocate:
            self._ensure_optimizer_param(env)
            self.user_model.remove("grad")

    def load_train_states(self, env: Env, states: Dict[str, Any]):
        if not self.preallocate:
            if not "optimizer" in states or states["optimizer"] is None:
                return
            self._ensure_optimizer_param(env)

    @property
    def memory(self):
        return self.user_model.memory

    def _init_user_model(self, env: Env, user_model_config: Union[PtuningConfig, LoraConfig]):
        if isinstance(user_model_config, PtuningConfig):
            user_model_config.check_or_set_decoders_config(
                num_layers=len(self.decoders.decoders),
                num_key_value_heads=self.decoders.decoders[0].config.num_key_value_heads,
                head_dim=self.decoders.decoders[0].config.head_dim
            )
            return user_model_config, PtuningMemoryModel(config=user_model_config, env=env)
        else:
            user_model_config.check_or_set_decoders_config(
                num_layers=len(self.decoders.decoders),
                num_key_value_heads=self.decoders.decoders[0].config.num_key_value_heads,
                num_query_heads=self.decoders.decoders[0].config.num_query_heads,
                head_dim=self.decoders.decoders[0].config.head_dim,
            )
            return user_model_config, LoraMemoryModel(config=user_model_config, env=env)


def train_tensor_to_embedding_tokens(
    head: ModelHeadMemoryModel,
    operate: OperateTrainTensorToEmbeddingTokens,
    env: Env
):
    hidden_states = env.trace(MemoryTensor.from_tensor(operate.hidden_states))
    hidden_states = head.train_to_embedding_tokens(env, hidden_states, operate.dtype)
    if operate.cpu_out:
        hidden_states = env.trace(hidden_states.item.cpu())
    return hidden_states


def do_forward(
    binding_data: TrainBindingDataMemoryModel,
    operate: OperateForward,
    env: Env
):
    meta = operate.meta
    device_pos_idx = meta.device_pos_idx
    assert meta.hidden_states is not None

    if operate.cpu_in:
        hidden_states = env.trace(MemoryTensor.from_carrier(meta.hidden_states).device())
    else:
        hidden_states = cast(MemoryHolder[MemoryTensor], env.retrack(meta.hidden_states.id))

    if device_pos_idx is None:
        device_pos_idx = env.trace(MemoryTensor.from_tensor(meta.cpu_pos_idx).device())
    else:
        device_pos_idx = cast(MemoryHolder[MemoryTensor], env.retrack(device_pos_idx))
    # assert isinstance(binding_data.user_model, PtuningMemoryModel), NotImplementedError
    hidden_states, ctx = forward(
        hidden_states,
        device_pos_idx,
        meta.input_lens,
        binding_data.user_model, binding_data.decoders,
        env,
    )
    binding_data.set(meta.sample_batch_id, ctx)
    if operate.cpu_out:
        hidden_states = env.trace(hidden_states.item.cpu())
    if operate.is_last or operate.cpu_out:
        device_pos_idx = None
    return hidden_states, device_pos_idx, ctx


def do_backward(
    binding_data: TrainBindingDataMemoryModel,
    operate: OperateBackward,
    env: Env
):
    if operate.cpu_in:
        grad_tensor = env.trace(MemoryTensor.from_carrier(operate.grad).device())
    else:
        grad_tensor = cast(MemoryHolder[MemoryTensor], env.retrack(operate.grad))

    forward_ctx = cast(ForwardCtxEachDecoderMemoryModeling, binding_data.get(operate.context_id))
    binding_data.remove(operate.context_id)
    # hidden_states_grad = env.trace(MemoryTensor(grad_tensor.item.size, grad_tensor.item.dtype, grad_tensor.item.is_cpu))
    hidden_states_grad = binding_data.user_model.train_backward(env, forward_ctx, grad_tensor)

    if not operate.has_next:
        hidden_states_grad = None
        return None
    if operate.cpu_out:
        hidden_states_grad = env.trace(hidden_states_grad.item.cpu())
    return hidden_states_grad


def compute_logits_loss_and_grad(
    model_tail: ModelTailMemoryModel,
    operate: OperateComputeLogitsLossAndGrad,
    env: Env
):
    label_ids = env.trace(MemoryTensor.from_tensor(operate.label_ids))
    if operate.cpu_in:
        hidden_states = env.trace(MemoryTensor.from_carrier(operate.carrier).device())
    else:
        hidden_states = env.retrack(operate.carrier)
    grad_data = model_tail.compute_loss_and_grad(hidden_states, label_ids, env)
    if operate.cpu_in:
        grad_data = env.trace(grad_data.item.cpu())
    return grad_data


@dataclass
class SeqComplexMemoryModel:
    env: Env
    block_manager: MemoryCacheEngine
    seq: SequenceLengthData
    user_model: Union[None, List[int], Tuple[LoraRenderConfig, List[int]]]
    occupied_blocks: List[int] = cast(List[int], field(default_factory=list))
    occupied_slots: List[int] = cast(List[int], field(default_factory=list))

    block_mapping: Dict[int, int] = cast(
        Dict[int, int], field(default_factory=dict))
    slot_mapping: Dict[int, int] = cast(
        Dict[int, int], field(default_factory=dict))

    def append_slot(self) -> int:
        current_slots = len(self.occupied_slots)
        num_blocks_req = (
            current_slots+1+(flags.BLOCK_SIZE-1))//flags.BLOCK_SIZE
        if num_blocks_req > len(self.occupied_blocks):
            b = self.block_manager.allocate_block()
            self.occupied_blocks.append(b)
            logic_block = len(self.block_mapping)
            self.block_mapping[logic_block] = b

            def revert_block():
                del self.block_mapping[logic_block]
                self.occupied_blocks = self.occupied_blocks[:-1]
            self.env.revert_stack.append(revert_block)
        b = self.occupied_blocks[-1]
        slot_offset = current_slots % flags.BLOCK_SIZE
        slot = b*flags.BLOCK_SIZE+slot_offset
        self.occupied_slots.append(slot)
        logic_slot: int = len(self.slot_mapping)
        self.slot_mapping[logic_slot] = slot

        def revert():
            del self.slot_mapping[logic_slot]
            self.occupied_slots = self.occupied_slots[:-1]
        self.env.revert_stack.append(revert)
        return slot

    @property
    def user_model_related_blocks(self) -> List[int]:
        if self.user_model is None:
            return []
        elif isinstance(self.user_model, tuple):
            return self.user_model[1]
        else:
            return self.user_model

    @property
    def user_model_mode(self) -> Optional[LoraRenderConfig]:
        if self.user_model is None:
            return None
        elif isinstance(self.user_model, tuple):
            return self.user_model[0]
        else:
            return None

    @property
    def ptuning_user_blocks(self) -> Optional[List[int]]:
        if self.user_model is None or isinstance(self.user_model, tuple):
            return None
        else:
            return self.user_model

    @property
    def current_slots_count(self) -> int:
        return len(self.occupied_slots)

    @property
    def current_blocks_count(self) -> int:
        return len(self.occupied_blocks)

    @property
    def generate_req_block(self) -> bool:
        current_slots = len(self.occupied_slots)
        num_blocks_req = (
            current_slots+1+(flags.BLOCK_SIZE-1))//flags.BLOCK_SIZE
        return num_blocks_req > len(self.occupied_blocks)

    @property
    def prefill_req_blocks(self) -> int:
        return (self.seq.prompt_len+flags.BLOCK_SIZE-1)//flags.BLOCK_SIZE


class MemoryPerTokenHidden(MemoryItem):
    def __init__(self, hidden_size: int) -> None:
        self.context_lens_t = MemoryTensor((1,), MemoryInt32)
        self.positions_t = MemoryTensor((1,), MemoryInt32)
        self.hidden = MemoryTensor(size=(hidden_size,), dtype=MemoryFloat16)

    @property
    def memory(self) -> int:
        return self.context_lens_t.memory+self.positions_t.memory+self.hidden.memory


@dataclass
class MemoryConfigBasic:
    cache_engine_implement: Literal['scalable', 'fix'] = "scalable"
    base_models_name: str = "base_model"
    train_bindings_name: str = "train_bindings"
    train_grad_optim_name: str = "train_grad_optim"
    train_ctx_name: str = "train_forward_ckpt"
    kv_cache_name: str = "kv_cache_engines"
    preswap: bool = False
    root_memory: int = -1
    preallocate: bool = True

    def update_root_memory(self, device: Optional[torch.device]):
        if device is None:
            assert self.root_memory != -1
        else:
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            self.root_memory = total_mem


class MemoryBrain:
    def __init__(self, models_path: str, basic_memory_config: MemoryConfigBasic) -> None:
        self._models = ModelsLib(lib_path=models_path)
        self.cache_kernel_implement: Literal['vllm', 'dual_table'] = "dual_table"
        self.decoder_implement: DecodersImplement = "our_fuse"
        self.max_allowed_len: int = 128*1024
        self.basic_memory_config = basic_memory_config
        assert self.basic_memory_config.root_memory != -1

        self.env = Env()
        self.base_models = MemoryDict(self.env, basic_memory_config.base_models_name)
        self.train_bindings = MemoryDict(self.env, basic_memory_config.train_bindings_name)
        self.train_grad_optim = MemoryDict(self.env, basic_memory_config.train_grad_optim_name)
        self.train_ctx = MemoryDict(self.env, basic_memory_config.train_ctx_name)
        self.kv_cache = MemoryDict(self.env, basic_memory_config.kv_cache_name)
        self.decoders_blocks_desc = MemoryDict(self.env)
        self.decoders_infer_bindings = MemoryDict(self.env)
        self.env.set("base_models", self.base_models)
        self.env.set("train_bindings", self.train_bindings)
        self.env.set("train_grad_optim", self.train_grad_optim)
        self.env.set("train_forward_ckpt", self.train_ctx)
        self.env.set("kv_cache_engines", self.kv_cache)
        self.env.merge_all()

        self.train_ctx_swappable = MemoryDict(self.env)
        self.incremental_idx = 0

    def next_unique_id(self):
        base = f"{self.incremental_idx}"
        self.incremental_idx += 1

        def revert():
            self.incremental_idx -= 1
        self.env.revert_stack.append(revert)
        return f"{base}"

    @property
    def total(self):
        return self.env.total

    @property
    def pools(self):
        return self.env.pools

    def pool(self, name: str):
        return self.env.pools[name]

    @property
    def all_merged(self):
        return len(self.env.revert_stack) == 0

    def revert_all(self):
        self.env.revert_all()

    def merge_all(self):
        self.env.merge_all()

    def snapshot(self):
        return self.env.snapshot()

    def swapout(self, target: str, **kwargs: Any):
        if target == "swap_blocks":
            if "manager" in kwargs:
                ce = self.get_decoders_kvcache_manager(kwargs["manager"])
                assert isinstance(ce, ScalableCacheEngineMemoryModel)
            elif "engine" in kwargs:
                ce = self.kv_cache[kwargs["engine"]]
            ce.swap_out(target_blocks=kwargs["bs"])
        elif target == "release_blocks":
            ce = self.get_decoders_kvcache_manager(kwargs["manager"])
            ce.release_blocks(kwargs["bs"])
        elif target == "train_ctx":
            obj = (kwargs["binding"], kwargs["context_id"])
            ctx = cast(CanSwapObjectMemory, self.train_ctx_swappable.get(obj))
            ctx.swap_out()
            self.train_ctx_swappable.remove(obj)
        elif target == "downgrad":
            binding = kwargs["binding"]
            context_id = kwargs["context_id"]
            obj = (binding, context_id)
            ctx = self.train_ctx_swappable[obj]
            assert isinstance(ctx, ForwardCtxEachDecoderMemoryModeling)
            dctx = ctx.downgrad()
            self.train_ctx_swappable.set(obj, dctx)
            self.train_bindings[binding].item.set(context_id, dctx)
        else:
            raise ValueError(target)

    def make_model_head(self, base_model: str):
        name = ModelDescribe.new(base_model, is_head=True).__str__()
        if name in self.base_models:
            return name
        model_info = self._models.models_storage[base_model]
        cfg = model_info.config.model_head_config
        head_memory_model = ModelHeadMemoryModel(cfg)
        self.base_models.set_item(name, head_memory_model)

    def remove_model_head(self, binding: str):
        del self.base_models[binding]

    def make_model_tail(self, base_model: str):
        name = ModelDescribe.new(base_model, is_tail=True).__str__()
        if name in self.base_models:
            return
        model_info = self._models.models_storage[base_model]
        cfg = model_info.config.model_tail_config
        head_memory_model = ModelTailMemoryModel(cfg)
        self.base_models.set_item(name, head_memory_model)

    def remove_model_tail(self, binding: str):
        del self.base_models[binding]

    def _make_model_decoders(self, base_model: str, decoders_range: Tuple[int, int]):
        name = ModelDescribe.new(base_model, start=decoders_range[0], end=decoders_range[1]).__str__()
        if name in self.base_models:
            return name, cast(DecodersGroupMemoryModel, self.base_models[name])
        model_info = self._models.models_storage[base_model]
        model_config = model_info.config
        start_layer, end_layer = decoders_range
        decoders_memory_model = \
            make_decoders_memory_model(
                implement=self.decoder_implement,
                config=model_config.quant_decoder_config,
                state_files=[
                    model_info.decoder_path(i)
                    for i in range(start_layer, end_layer)
                ],)
        self.base_models.set_item(name, decoders_memory_model)
        return name, decoders_memory_model

    def make_model_decoders(self, base_model: str, decoders_range: Tuple[int, int]):
        binding_name, decoders_group = self._make_model_decoders(base_model, decoders_range)
        block_info = decoders_group.block_info
        num_layers = decoders_group.num_layers
        desc = f"{block_info.num_key_value_heads}*{block_info.head_dim}*{num_layers}"
        self.decoders_blocks_desc.set(binding_name, desc)

    def remove_model_decoders(self, binding: str):
        del self.base_models[binding]
        desc = self.decoders_blocks_desc.get(binding)
        del self.decoders_blocks_desc[binding]
        for d in self.decoders_blocks_desc.childrens.values():
            if d == desc:
                return
        if desc in self.kv_cache:
            del self.kv_cache[desc]

    def _lazy_init_cache_engine(self, decoders_group: DecodersGroupMemoryModel, max_memory: int) -> MemoryCacheEngine:
        block_info = decoders_group.block_info
        per_block_bytes = block_info.bytes_per_block*decoders_group.num_layers*2  # 2 for KV
        max_blocks = max_memory//per_block_bytes
        # print(f"under memory limit: {max_memory/(2**20):.2f}MiB, blocks available: {max_blocks}")
        blocks_count = max_blocks
        if self.basic_memory_config.cache_engine_implement == "scalable":
            cache_engine = ScalableCacheEngineMemoryModel(
                env=self.env,
                block_info=block_info,
                max_blocks_count=blocks_count,
                num_layers=decoders_group.num_layers,
                preswap=self.basic_memory_config.preswap,
            )
        else:
            cache_engine = VllmFixCacheEngineMemoryModel(
                env=self.env,
                block_info=block_info,
                num_device_blocks=blocks_count,
                num_cpu_blocks=blocks_count,
                num_layers=decoders_group.num_layers,
            )
        return cache_engine

    def get_decoders_kvcache_manager(self, binding: str) -> MemoryCacheEngine:
        desc = self.decoders_blocks_desc[binding]
        if desc in self.kv_cache:
            return self.kv_cache[desc]
        else:
            # lazy init
            decoders_group = cast(MemoryHolder[DecodersGroupMemoryModel], self.base_models[binding]).item
            if self.basic_memory_config.cache_engine_implement == "fix":
                root_mem = self.basic_memory_config.root_memory
                base_mem = self.pool(self.basic_memory_config.base_models_name)
                reserved = int(max(min(2**30, (root_mem-base_mem)*0.2), 2**30))
                available = root_mem-reserved-base_mem
            else:
                available = 128*(2**30)  # 128GiB, memory modeling and controller will take over memory limit
            cache_engine = self._lazy_init_cache_engine(decoders_group, available)
            self.kv_cache.set(desc, cache_engine)
            return cache_engine

    def create_decoders_train_binding(self, op: OperateCreateDecodersTrainBinding):
        binding_name = op.binding_name+f",{self.next_unique_id()}"
        decoders = cast(MemoryHolder[DecodersGroupMemoryModel], self.base_models[op.binding]).item
        _binding = TrainBindingDataMemoryModel(
            decoders=decoders,
            user_model_config=op.user_model_config,
            optimizer_config=op.optimizer_config,
            scheduler_config=op.scheduler_config,
            data_init_method=op.data_init_method,
            env=self.env,
            preallocate=self.basic_memory_config.preallocate,
        )
        self.train_bindings.set(binding_name, MemoryHolder(_binding, self.env))
        self.train_ctx.set(binding_name, _binding)
        self.train_grad_optim.set(binding_name, _binding.user_model)

    def train_tensor_to_embedding_tokens(self, op: OperateTrainTensorToEmbeddingTokens):
        model_head = cast(MemoryHolder[ModelHeadMemoryModel], self.base_models[op.binding]).item
        memory_ret = train_tensor_to_embedding_tokens(model_head, op, self.env)

        def vbind():
            self.env.vuntrack(memory_ret)

        def bind(ret: DeviceTensorCarrier):
            self.env.unmanaged.remove(id(memory_ret))
            self.env.untrack(memory_ret, ret.id)
        return vbind, bind

    def forward(self, op: OperateForward):
        binding_data = cast(MemoryHolder[TrainBindingDataMemoryModel], self.train_bindings[op.binding]).item
        hidden_states, device_pos_idx, ctx = do_forward(binding_data, op, self.env)
        self.train_ctx_swappable.set((op.binding, op.context_id), ctx)

        def vbind():
            if not hidden_states.item.is_cpu:
                self.env.vuntrack(hidden_states)
            if device_pos_idx is not None:
                self.env.vuntrack(device_pos_idx)

        def bind(ret: TrainForwardMeta):
            if not hidden_states.item.is_cpu:
                assert ret.hidden_states is not None
                self.env.unmanaged.remove(id(hidden_states))
                self.env.untrack(hidden_states, ret.hidden_states)
            if device_pos_idx is not None:
                assert ret.device_pos_idx is not None
                self.env.unmanaged.remove(id(device_pos_idx))
                self.env.untrack(device_pos_idx, ret.device_pos_idx)
        return vbind, bind

    def compute_logits_loss_and_grad(self, op: OperateComputeLogitsLossAndGrad):
        model_tail = cast(MemoryHolder[ModelTailMemoryModel], self.base_models[op.binding]).item
        grad = compute_logits_loss_and_grad(model_tail, op, self.env)

        def vbind():
            if not grad.item.is_cpu:
                self.env.vuntrack(grad)

        def bind(ret: Tuple[float, DeviceTensorCarrier]):
            if not grad.item.is_cpu:
                self.env.unmanaged.remove(id(grad))
                self.env.untrack(grad, ret[1])
        return vbind, bind

    def backward(self, op: OperateBackward):
        binding_data = cast(MemoryHolder[TrainBindingDataMemoryModel], self.train_bindings[op.binding]).item
        grad = do_backward(binding_data, op, self.env)
        self.train_ctx_swappable.remove((op.binding, op.context_id))

        def vbind():
            if grad is not None and not grad.item.is_cpu:
                self.env.vuntrack(grad)

        def bind(ret: Optional[DeviceTensorCarrier]):
            if grad is not None and not grad.item.is_cpu:
                assert ret is not None
                self.env.unmanaged.remove(id(grad))
                self.env.untrack(grad, ret)
        return vbind, bind

    def weight_update(self, op: OperateWeightUpdate):
        binding_data = cast(MemoryHolder[TrainBindingDataMemoryModel], self.train_bindings[op.binding]).item
        binding_data.weight_update(self.env)

    def load_train_states(self, op: OperateLoadTrainStates):
        binding_data = cast(MemoryHolder[TrainBindingDataMemoryModel], self.train_bindings[op.binding]).item
        binding_data.load_train_states(self.env, op.states)

    def dump_train_states(self, op: OperateDumpTrainStates):
        pass

    def put_user_model(self, op: OperatePutUserModel):
        cache_engine = self.get_decoders_kvcache_manager(op.binding)
        blocks = op.user_model_blocks
        num_blocks = len(blocks[0])
        blocks_idx = cache_engine.allocate_blocks(num_blocks)
        cache_engine.set_blocks_readonly(blocks_idx)

    def create_decoders_infer_binding(self, op: OperateCreateDecodersInferBinding):
        binding = op.binding
        cache_engine = self.get_decoders_kvcache_manager(op.binding)
        seq_complex = SeqComplexMemoryModel(
            env=self.env,
            block_manager=cache_engine,
            seq=op.seq_length_data.clone(),
            user_model=op.user_model)
        if op.allocate_slots:
            num_tokens = op.seq_length_data.num_tokens
            while num_tokens > 1:
                num_tokens -= 1
                seq_complex.append_slot()
        decoders = cast(DecodersGroupMemoryModel, self.base_models[binding].item)
        config = decoders.config
        binding_name = op.binding_name+f",{self.next_unique_id()}"
        self.decoders_infer_bindings.set(binding_name, (binding, seq_complex, config.hidden_size))

    def infer_tensor_to_embedding_tokens(self, op: OperateInferTensorToEmbeddingTokens):
        head = cast(MemoryHolder[ModelHeadMemoryModel], self.base_models[op.binding]).item
        if op.cpu_out:
            def vbind(): pass
            def bind(ret: Any): pass
            return vbind, bind
        t = op.tokens
        l = len(t)
        inp = self.env.trace(MemoryTensor((l,), dtype=MemoryInt32, cpu=True))
        out = head.infer_to_embedding_tokens(self.env, inp, MemoryFloat16)

        def vbind():
            self.env.vuntrack(out)

        def bind(ret: Any):
            self.env.unmanaged.remove(id(out))
            self.env.untrack(out, ret)
        return vbind, bind

    def do_prefill(self, op: OperateDoPrefill):
        if not op.cpu_in:
            self.env.retrack(op.carrier)
        binding, sc, _ = cast(Tuple[str, SeqComplexMemoryModel, int], self.decoders_infer_bindings[op.binding])
        actual_prompt_len = op.carrier.shape[0]
        related_blocks: List[int] = []
        read_only_blocks: List[int] = []
        for _ in range(actual_prompt_len):
            slot = sc.append_slot()
            related_blocks.append(slot//flags.BLOCK_SIZE)
            read_only_blocks += sc.occupied_blocks[:-1]
        related_blocks.extend(sc.user_model_related_blocks)
        related_blocks = list(set(related_blocks))
        cache_engine = self.get_decoders_kvcache_manager(binding)
        cache_engine.set_blocks_readonly(read_only_blocks)
        cache_engine.get_cache_provider(related_blocks)
        if op.cpu_out:
            def _vbind(): pass
            def _bind(ret: Any): pass
            return _vbind, _bind
        else:
            out = self.env.trace(MemoryTensor(size=op.carrier.shape, dtype=MemoryFloat16))

            def vbind(): self.env.vuntrack(out)

            def bind(ret: DeviceTensorCarrier):
                self.env.unmanaged.remove(id(out))
                self.env.untrack(out, ret)
            return vbind, bind

    def hidden_state_to_sampling_output(self, op: OperateHiddenStateToSamplingOutput):
        if not op.cpu_in:
            self.env.retrack(op.carrier)

    def do_generating(self, op: OperateDoGenerating):
        if not op.cpu_in:
            self.env.retrack(op.carrier)
        binding, sc, hidden_size = cast(Tuple[str, SeqComplexMemoryModel, int], self.decoders_infer_bindings[op.binding])
        read_only_blocks: List[int] = []
        sc.append_slot()
        read_only_blocks.extend(sc.occupied_blocks[-3:-1])
        related_blocks = sc.user_model_related_blocks+sc.occupied_blocks
        cache_engine = self.get_decoders_kvcache_manager(binding)
        cache_engine.set_blocks_readonly(read_only_blocks)
        cache_engine.get_cache_provider(related_blocks)
        if op.cpu_out:
            def _vbind(): pass
            def _bind(ret: Any): pass
            return _vbind, _bind
        else:
            out = self.env.trace(MemoryPerTokenHidden(hidden_size))

            def vbind():
                self.env.vuntrack(out)

            def bind(ret: Any):
                self.env.unmanaged.remove(id(out))
                self.env.untrack(out, ret)
            return vbind, bind

    def get_blocks(self, op: OperateGetBlocks):
        binding, sc, _ = cast(Tuple[str, SeqComplexMemoryModel, int], self.decoders_infer_bindings[op.binding])
        cache_engine = self.get_decoders_kvcache_manager(binding)
        local_blocks = [sc.block_mapping[b] for b in op.seq_blocks]
        cache_engine.get_blocks(local_blocks)

    def put_blocks(self, op: OperatePutBlocks):
        binding, sc, _ = cast(Tuple[str, SeqComplexMemoryModel, int], self.decoders_infer_bindings[op.binding])
        cache_engine = self.get_decoders_kvcache_manager(binding)
        local_blocks = [sc.block_mapping[b] for b in op.seq_blocks]
        cache_engine.put_blocks(local_blocks)
        if len(local_blocks) > 0:
            # could be last block of prefill
            cache_engine.set_blocks_readonly(local_blocks[:-1])

    def remove_decoders_train_binding(self, binding: str) -> None:
        self.train_bindings.remove(binding)
        self.train_grad_optim.remove(binding)
        self.train_ctx.remove(binding)

    def remove_decoders_infer_binding(self, binding: str) -> None:
        _binding, sc, _ = cast(Tuple[str, SeqComplexMemoryModel, int], self.decoders_infer_bindings[binding])
        cache_engine = self.get_decoders_kvcache_manager(_binding)
        cache_engine.release_blocks(list(sc.block_mapping.values()))
        del self.decoders_infer_bindings[binding]
