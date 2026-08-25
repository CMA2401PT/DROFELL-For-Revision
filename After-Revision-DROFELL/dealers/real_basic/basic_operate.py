from typing import Any, Dict, List, Optional, Set, Tuple, cast
from utils.format import format_size
from operators.assemble.fwd_ctx.each_decoder import ForwardCtxEachDecoder
from operators.cache_engine.common import CacheEngine
from operators.cache_engine.scalable import ScalableCacheEngine
from operators.cache_engine.vllm_fix import VllmFixCacheEngine
from operators import flags as operator_flags
from operators.container.container import Container
from operators.core.swappable.real import CanSwapObject
from ..common.brain import *
from .memory import MemoryBrain, MemoryConfigBasic, SeqComplexMemoryModel
from .infer_binding import SeqComplex
from .decode_graph import (
    DecodeGraphRunner,
    GraphKey,
    HeadDecodeGraphRunner,
    StageGraphKey,
    TailDecodeGraphRunner,
)
from .graph_memory import raise_graph_capture_oom
from .thread_worker import AsyncTaskRunner, ThreadTaskRunner
from . import head, tail, train_binding, infer_binding
from . import flags


class BasicBrainOperates(Brain):
    def __init__(self,
                 models_path: str,
                 device: torch.device,
                 container: Container,
                 memory_config: MemoryConfigBasic,
                 max_seq_len: int,
                 max_concurrency: int,
                 decoder_implement: DecodersImplement,
                 individual_train_thread: bool = False
                 ) -> None:
        super().__init__(
            models_path,
            device,
            max_seq_len=max_seq_len,
            max_concurrency=max_concurrency,
            decoder_implement=decoder_implement,
        )
        self.basic_memory_config = memory_config
        self.memory_modeling = MemoryBrain(
            models_path,
            memory_config,
            max_seq_len=max_seq_len,
            decoder_implement=decoder_implement,
            device=device,
        )
        self.decode_graph_runner = DecodeGraphRunner(
            enabled=False,
            max_context_len=max_seq_len,
            batch_sizes=tuple(range(1, max_concurrency + 1)),
            on_capture=self._on_decode_graph_capture,
            on_remove=self._on_decode_graph_remove,
        )
        self.head_decode_graph_runner = HeadDecodeGraphRunner(
            enabled=False,
            batch_sizes=tuple(range(1, max_concurrency + 1)),
            on_capture=self._on_head_decode_graph_capture,
            on_remove=self._on_head_decode_graph_remove,
        )
        self.tail_decode_graph_runner = TailDecodeGraphRunner(
            enabled=False,
            batch_sizes=tuple(range(1, max_concurrency + 1)),
            on_capture=self._on_tail_decode_graph_capture,
            on_remove=self._on_tail_decode_graph_remove,
        )
        # Explicit topology declarations. They are reconciled whenever any
        # participating base-model stage is made, independent of load order.
        self._head_decode_neighbors: Dict[str, str] = {}
        self._tail_decode_neighbors: Dict[str, str] = {}
        self.container = container
        self.decoders_train_bindings: Dict[str,
                                           train_binding.TrainBindingData] = {}
        self.decoders_blocks_desc: Dict[str, str] = {}
        self.kv_cache_engines: Dict[str,  CacheEngine] = {}
        # infer_binding->(decoders_binding,seq_complex)
        self.decoders_infer_bindings: Dict[str, Tuple[str, SeqComplex]] = {}
        self.incremental_idx = 0
        self.train_ctx_swappable: Dict[Tuple[str, str], CanSwapObject] = {}
        self.individual_train_thread = individual_train_thread
        self.train_runner = AsyncTaskRunner(container)
        if self.individual_train_thread:
            train_stream = torch.cuda.Stream(container.stream)
            # we need this to avoid blocking other coroutine
            self.train_runner = ThreadTaskRunner(
                max_workers=1, get_stream=lambda: train_stream)

    def capture_decode_graph(self) -> None:
        if self._base_heads or self._base_decoders or self._base_tails:
            raise RuntimeError(
                "decode graph must be enabled before loading models")
        self.decode_graph_runner = DecodeGraphRunner(
            enabled=True,
            max_context_len=self.max_seq_len,
            batch_sizes=tuple(range(1, self.max_concurrency + 1)),
            on_capture=self._on_decode_graph_capture,
            on_remove=self._on_decode_graph_remove,
        )
        self.head_decode_graph_runner = HeadDecodeGraphRunner(
            enabled=True,
            batch_sizes=tuple(range(1, self.max_concurrency + 1)),
            on_capture=self._on_head_decode_graph_capture,
            on_remove=self._on_head_decode_graph_remove,
        )
        self.tail_decode_graph_runner = TailDecodeGraphRunner(
            enabled=True,
            batch_sizes=tuple(range(1, self.max_concurrency + 1)),
            on_capture=self._on_tail_decode_graph_capture,
            on_remove=self._on_tail_decode_graph_remove,
        )

    def _capture_graph_or_fail(
        self,
        graph_kind: str,
        binding: str,
        capture: Callable[[], None],
    ) -> None:
        try:
            capture()
        except torch.OutOfMemoryError as error:
            raise_graph_capture_oom(
                graph_kind=graph_kind,
                binding=binding,
                graph_budget=self.basic_memory_config.graph_budget,
                cause=error,
            )

    def _track_graph_runtime_memory(self, key: Any, requested_bytes: int) -> None:
        self.memory_modeling.track_runtime_memory(key, requested_bytes)

    def _remove_graph_runtime_memory(self, key: Any) -> None:
        self.memory_modeling.remove_runtime_memory(key)

    def _on_decode_graph_capture(self, key: GraphKey, requested_bytes: int) -> None:
        self._track_graph_runtime_memory(
            ("decode_graph", *key.as_tuple()), requested_bytes
        )

    def _on_decode_graph_remove(self, key: GraphKey) -> None:
        self._remove_graph_runtime_memory(
            ("decode_graph", *key.as_tuple())
        )

    def _on_head_decode_graph_capture(
        self, key: StageGraphKey, requested_bytes: int
    ) -> None:
        self._track_graph_runtime_memory(
            ("decode_graph", "head", *key), requested_bytes
        )

    def _on_head_decode_graph_remove(self, key: StageGraphKey) -> None:
        self._remove_graph_runtime_memory(
            ("decode_graph", "head", *key)
        )

    def _on_tail_decode_graph_capture(
        self, key: StageGraphKey, requested_bytes: int
    ) -> None:
        self._track_graph_runtime_memory(
            ("decode_graph", "tail", *key), requested_bytes
        )

    def _on_tail_decode_graph_remove(self, key: StageGraphKey) -> None:
        self._remove_graph_runtime_memory(
            ("decode_graph", "tail", *key)
        )

    def query_extra_info(self, key: str) -> Any:
        if key == "decode_graph":
            stats = dict(self.decode_graph_runner.stats())
            stats["head"] = dict(self.head_decode_graph_runner.stats())
            stats["tail"] = dict(self.tail_decode_graph_runner.stats())
            return stats
        return super().query_extra_info(key)

    def next_unique_id(self):
        base = f"{self.incremental_idx}"
        self.incremental_idx += 1
        return f"{base}"

    def make_model_head(
        self,
        base_model: str,
        adjacent_decoder: Optional[str] = None,
    ) -> str:
        self.memory_modeling.make_model_head(base_model)
        self.memory_modeling.merge_all()
        binding = super().make_model_head(base_model, adjacent_decoder)
        self._capture_graph_or_fail(
            "head",
            binding,
            lambda: self.head_decode_graph_runner.capture_binding(
                binding=binding,
                model_head=self._base_heads[binding],
                stream=self.container.stream,
                device=self.container.device,
            ),
        )
        if adjacent_decoder is not None:
            self._register_decode_graph_neighbor(
                stage_binding=binding,
                decoder_binding=adjacent_decoder,
                neighbors=self._head_decode_neighbors,
                stage="head",
            )
            self._capture_decoder_graph_binding(adjacent_decoder)
        return binding

    def make_model_tail(
        self,
        base_model: str,
        adjacent_decoder: Optional[str] = None,
    ) -> str:
        self.memory_modeling.make_model_tail(base_model)
        self.memory_modeling.merge_all()
        binding = super().make_model_tail(base_model, adjacent_decoder)
        self._capture_graph_or_fail(
            "tail",
            binding,
            lambda: self.tail_decode_graph_runner.capture_binding(
                binding=binding,
                model_tail=self._base_tails[binding],
                stream=self.container.stream,
                device=self.container.device,
            ),
        )
        if adjacent_decoder is not None:
            self._register_decode_graph_neighbor(
                stage_binding=binding,
                decoder_binding=adjacent_decoder,
                neighbors=self._tail_decode_neighbors,
                stage="tail",
            )
            self._capture_decoder_graph_binding(adjacent_decoder)
        return binding

    def _register_decode_graph_neighbor(
        self,
        *,
        stage_binding: str,
        decoder_binding: str,
        neighbors: Dict[str, str],
        stage: str,
    ) -> None:
        stage_desc = ModelDescribe.from_str(stage_binding)
        decoder_desc = ModelDescribe.from_str(decoder_binding)
        if not decoder_desc.is_decoders:
            raise ValueError(f"adjacent {stage} target is not a decoder")
        if stage_desc.base_model != decoder_desc.base_model:
            raise ValueError(f"adjacent {stage} and decoder models differ")
        previous = neighbors.get(stage_binding)
        if previous is not None and previous != decoder_binding:
            raise ValueError(f"{stage} adjacency cannot be changed")
        for other_stage, other_decoder in neighbors.items():
            if other_stage != stage_binding \
                    and other_decoder == decoder_binding:
                raise ValueError(
                    f"decoder already has an adjacent {stage} binding"
                )
        neighbors[stage_binding] = decoder_binding

    def _desired_decode_graph_variants(
        self,
        decoder_binding: str,
    ) -> Set[Tuple[Optional[str], Optional[str]]]:
        head_binding = next((
            stage for stage, decoder in self._head_decode_neighbors.items()
            if decoder == decoder_binding and stage in self._base_heads
        ), None)
        tail_binding = next((
            stage for stage, decoder in self._tail_decode_neighbors.items()
            if decoder == decoder_binding and stage in self._base_tails
        ), None)
        if head_binding is not None and tail_binding is not None:
            return {(head_binding, tail_binding)}
        if head_binding is not None:
            return {(head_binding, None)}
        if tail_binding is not None:
            return {(None, tail_binding)}
        return set()

    def make_model_decoders(self, base_model: str, decoders_range: Tuple[int, int]) -> Any:
        self.memory_modeling.make_model_decoders(base_model, decoders_range)
        self.memory_modeling.merge_all()
        binding_name = super().make_model_decoders(base_model, decoders_range)
        decoders_group = self._base_decoders[binding_name]
        block_info = decoders_group.block_info
        num_layers = decoders_group.num_layers
        desc = f"{block_info.num_key_value_heads}*{block_info.head_dim}*{num_layers}"
        self.decoders_blocks_desc[binding_name] = desc
        self._capture_decoder_graph_binding(binding_name)
        return binding_name

    def _capture_decoder_graph_binding(self, binding: str) -> None:
        if not self.decode_graph_runner.enabled \
                or binding not in self._base_decoders:
            return
        self._capture_graph_or_fail(
            "decoder",
            binding,
            lambda: self._capture_decoder_graph_binding_unchecked(binding),
        )

    def _capture_decoder_graph_binding_unchecked(self, binding: str) -> None:
        cache_engine = self.get_decoders_kvcache_manager(binding)
        cache_engine_id = id(cache_engine)
        desired_variants = {
            (None, None),
            *self._desired_decode_graph_variants(binding),
        }
        captured_variants = self.decode_graph_runner.variants_for_binding(
            binding, cache_engine_id
        )
        for head_binding, tail_binding in (
            captured_variants - desired_variants
        ):
            self.decode_graph_runner.remove_variant(
                binding=binding,
                cache_engine_id=cache_engine_id,
                head_binding=head_binding,
                tail_binding=tail_binding,
            )
        missing_variants = desired_variants - captured_variants
        if not missing_variants:
            return
        max_batch_size = max(self.decode_graph_runner.batch_sizes)
        capture_block_count = (
            max_batch_size + operator_flags.BLOCK_SIZE - 1
        ) // operator_flags.BLOCK_SIZE
        self.memory_modeling.prepare_decode_graph_capture(
            binding, capture_block_count
        )
        self.memory_modeling.merge_all()
        with torch.cuda.stream(self.container.stream), torch.device(self.device):
            temporary_blocks = cache_engine.allocate_blocks(
                capture_block_count
            )
            try:
                if len(temporary_blocks) != capture_block_count:
                    raise RuntimeError(
                        "not enough KV cache blocks for decode graph capture"
                    )
                cache_provider, translator = cache_engine.get_cache_provider(
                    temporary_blocks
                )
                block_tables = []
                slots = []
                for sequence_i in range(max_batch_size):
                    block = temporary_blocks[
                        sequence_i // operator_flags.BLOCK_SIZE
                    ]
                    slot = (
                        block * operator_flags.BLOCK_SIZE
                        + sequence_i % operator_flags.BLOCK_SIZE
                    )
                    block_tables.append([translator.translate_block(block)])
                    slots.append(translator.translate_slot(slot))
                decoders = self._base_decoders[binding]
                for head_binding, tail_binding in sorted(
                    missing_variants,
                    key=lambda variant: (
                        variant[0] or "", variant[1] or ""
                    ),
                ):
                    self.decode_graph_runner.capture_binding(
                        binding=binding,
                        cache_engine_id=cache_engine_id,
                        decoders=decoders,
                        cache_provider=cache_provider,
                        stream=self.container.stream,
                        device=self.container.device,
                        position_ids_dtype=(
                            decoders.decoders[0].rotary_emb.position_ids_dtype
                        ),
                        block_tables=block_tables,
                        slots=slots,
                        head_binding=head_binding,
                        tail_binding=tail_binding,
                        model_head=(
                            self._base_heads[head_binding]
                            if head_binding is not None else None
                        ),
                        model_tail=(
                            self._base_tails[tail_binding]
                            if tail_binding is not None else None
                        ),
                    )
            finally:
                cache_engine.release_blocks(temporary_blocks)

    def remove_decoders_infer_binding(self, binding: str) -> None:
        self.memory_modeling.remove_decoders_infer_binding(binding)
        self.memory_modeling.merge_all()
        decoders_binding, seq_complex = self.decoders_infer_bindings[binding]
        cache_engine = self.get_decoders_kvcache_manager(decoders_binding)
        cache_engine.release_blocks(list(seq_complex.block_mapping.values()))
        del self.decoders_infer_bindings[binding]

    def remove_decoders_train_binding(self, binding: str) -> None:
        self.memory_modeling.remove_decoders_train_binding(binding)
        self.memory_modeling.merge_all()
        # binding_data = self._decoders_train_bindings[binding]
        # if not flags.SIMULATE_DROP:
        #     assert (len(binding_data.forward_ctxs) == 0)
        del self.decoders_train_bindings[binding]

    def remove_model_head(self, binding: str):
        self.decode_graph_runner.remove_head_binding(binding)
        self._head_decode_neighbors.pop(binding, None)
        self.head_decode_graph_runner.remove_binding(binding)
        self.memory_modeling.remove_model_head(binding=binding)
        self.memory_modeling.merge_all()
        del self._base_heads[binding]

    def remove_model_tail(self, binding: str):
        self.decode_graph_runner.remove_tail_binding(binding)
        self._tail_decode_neighbors.pop(binding, None)
        self.tail_decode_graph_runner.remove_binding(binding)
        self.memory_modeling.remove_model_tail(binding=binding)
        self.memory_modeling.merge_all()
        del self._base_tails[binding]

    def remove_model_decoders(self, binding: str):
        self.decode_graph_runner.remove_binding(binding)
        self._head_decode_neighbors = {
            stage: decoder
            for stage, decoder in self._head_decode_neighbors.items()
            if decoder != binding
        }
        self._tail_decode_neighbors = {
            stage: decoder
            for stage, decoder in self._tail_decode_neighbors.items()
            if decoder != binding
        }
        self.memory_modeling.remove_model_decoders(binding=binding)
        self.memory_modeling.merge_all()
        del self._base_decoders[binding]
        desc = self.decoders_blocks_desc[binding]
        del self.decoders_blocks_desc[binding]
        for d in self.decoders_blocks_desc.values():
            if d == desc:
                return
        if desc in self.kv_cache_engines:
            del self.kv_cache_engines[desc]

    def get_seq_occupied_blocks(self, name: str) -> List[int]:
        _, sc = self.decoders_infer_bindings[name]
        return sc.occupied_blocks

    def make_cache_engine(self, decoders_group: DecodersGroup, max_memory: int):
        block_info = decoders_group.block_info
        per_block_bytes = block_info.bytes_per_block * \
            decoders_group.num_layers*2  # 2 for KV
        print(
            f"per block size: {block_info.bytes_per_block}*{decoders_group.num_layers}*2={per_block_bytes/(2**20):.2f}MiB")
        max_blocks = max_memory//per_block_bytes
        print(f"limited to: {max_blocks} blocks")
        blocks_count = max_blocks
        if self.basic_memory_config.cache_engine_implement == "scalable":
            cache_engine = ScalableCacheEngine(
                block_info=block_info,
                max_blocks_count=blocks_count,
                num_layers=decoders_group.num_layers,
                device=self.device,
                preswap=self.basic_memory_config.preswap,
            )
        else:
            cache_engine = VllmFixCacheEngine(
                block_info=block_info,
                num_device_blocks=blocks_count,
                num_cpu_blocks=blocks_count,
                num_layers=decoders_group.num_layers,
                device=self.device,
            )
        return cache_engine

    def get_decoders_kvcache_manager(self, binding: str):
        desc = self.decoders_blocks_desc[binding]
        if desc in self.kv_cache_engines:
            return self.kv_cache_engines[desc]
        else:
            # lazy init
            decoders_group = self._base_decoders[binding]
            if self.basic_memory_config.cache_engine_implement == "fix":
                root_mem = self.basic_memory_config.root_memory
                base_mem = self.memory_modeling.pool(
                    self.basic_memory_config.base_models_name)
                reserved = int(max(min(2**30, (root_mem-base_mem)*0.2), 2**30))
                print(
                    f"create cache engine for: {desc}, cache engine will pre-occupy available space")
                print(f"\troot memory:{format_size(root_mem)}")
                print(f"\teserved memory:{format_size(reserved)}")
                print(f"\tbase_mem memory:{format_size(base_mem)}")
                available = root_mem-reserved-base_mem
                print(f"\tmax available memory:{format_size(available)}")
            else:
                print(
                    f"create cache engine for: {desc}, memory modeling and controller will take over memory limit")
                available = 128*(2**30)
            cache_engine = self.make_cache_engine(decoders_group, available)
            self.kv_cache_engines[desc] = cache_engine
            return cache_engine

    async def create_decoders_infer_binding(self, op: OperateCreateDecodersInferBinding, modeling: bool = True):
        op.on_start()
        if modeling:
            self.memory_modeling.create_decoders_infer_binding(op)
        self.memory_modeling.merge_all()
        binding = op.binding
        cache_engine = \
            self.get_decoders_kvcache_manager(binding)
        seq_complex = SeqComplex(
            block_manager=cache_engine, seq=op.seq_length_data, user_model=op.user_model,
        )
        if op.allocate_slots:
            num_tokens = op.seq_length_data.num_tokens
            while num_tokens > 1:
                num_tokens -= 1
                seq_complex.append_slot()
        binding_name = op.binding_name+f",{self.next_unique_id()}"
        self.decoders_infer_bindings[binding_name] = \
            (binding, seq_complex)
        op.set_result(binding_name)
        op.on_finish()

    async def create_decoders_train_binding(self, op: OperateCreateDecodersTrainBinding, modeling: bool = True):
        op.on_start()
        if modeling:
            self.memory_modeling.create_decoders_train_binding(op)
        self.memory_modeling.merge_all()
        decoders = self._base_decoders[op.binding]

        async def _wrap(op: OperateCreateDecodersTrainBinding, container: Container):
            binding_data = await container.step(
                "create-train-binding", train_binding.TrainBindingData,
                decoders=decoders,
                user_model_config=op.user_model_config,
                optimizer_config=op.optimizer_config,
                scheduler_config=op.scheduler_config,
                data_init_method=op.data_init_method
            )
            binding_name = op.binding_name+f",{self.next_unique_id()}"
            self.decoders_train_bindings[binding_name] = binding_data
            op.set_result(binding_name)
        await self.train_runner.run_wait(op, _wrap, op.on_finish)

    async def put_user_model(self, op: OperatePutUserModel):
        binding = op.binding
        cache_engine = \
            self.get_decoders_kvcache_manager(binding)

        blocks = op.user_model_blocks
        num_blocks = len(blocks[0])

        blocks_idx = cache_engine.allocate_blocks(num_blocks)
        op.on_start()
        self.memory_modeling.put_user_model(op)
        self.memory_modeling.merge_all()

        await self.container.step("put_blocks",
                                  cache_engine.put_blocks,
                                  target_blocks=blocks_idx,
                                  kv_cache=blocks
                                  )
        cache_engine.set_blocks_readonly(blocks_idx)
        handle = UserModelHandle(
            blocks=blocks_idx,
            _free=lambda: self.swapout(
                target="release_blocks", manager=binding, bs=blocks_idx),
            lora_render_cfg=op.lora_render_cfg
        )
        if not isinstance(cache_engine, ScalableCacheEngine):
            # no actual copy
            sp = self.container.sync_point()
            op.set_result(handle, sp)
        else:
            op.set_result(handle)
        op.on_finish()

    async def put_blocks(self, op: OperatePutBlocks):
        decoders_binding, seq_complex = self.decoders_infer_bindings[op.binding]
        cache_engine = self.get_decoders_kvcache_manager(decoders_binding)
        op.on_start()
        self.memory_modeling.put_blocks(op)
        self.memory_modeling.merge_all()
        await infer_binding.put_blocks(
            operate=op,
            seq_complex=seq_complex,
            cache_engine=cache_engine,
            container=self.container
        )
        op.on_finish()

    async def get_blocks(self, op: OperateGetBlocks):
        decoders_binding, seq_complex = self.decoders_infer_bindings[op.binding]
        cache_engine = self.get_decoders_kvcache_manager(decoders_binding)
        op.on_start()
        self.memory_modeling.get_blocks(op)
        self.memory_modeling.merge_all()
        await infer_binding.get_blocks(
            operate=op,
            seq_complex=seq_complex,
            cache_engine=cache_engine,
            container=self.container
        )
        op.on_finish()

    async def dump_train_states(self, op: OperateDumpTrainStates):
        binding = op.binding
        binding_data = self.decoders_train_bindings[binding]
        op.on_start()
        self.memory_modeling.dump_train_states(op)
        self.memory_modeling.merge_all()

        async def _wrap(op: OperateDumpTrainStates, container: Container):
            await train_binding.dump_train_states(binding_data, container, op)
        await self.train_runner.run_wait(op, _wrap, op.on_finish)

    async def load_train_states(self, op: OperateLoadTrainStates, modeling: bool = True):
        binding_data = self.decoders_train_bindings[op.binding]
        op.on_start()
        if modeling:
            self.memory_modeling.load_train_states(op)
        self.memory_modeling.merge_all()

        async def _wrap(op: OperateLoadTrainStates, container: Container):
            await train_binding.load_train_states(binding_data, container, op)
        await self.train_runner.run_wait(op, _wrap, op.on_finish)

    async def infer_tensor_to_embedding_tokens(self, binding: str, ops: List[OperateInferTensorToEmbeddingTokens]):
        model_head = self._base_heads[binding]
        binds: List[Callable[[Any], None]] = []
        for op in ops:
            binds.append(
                self.memory_modeling.infer_tensor_to_embedding_tokens(op)[1])
            op.on_start()
        await head.infer_tensor_to_embedding_tokens(
            model_head,
            self.container,
            ops,
            self.head_decode_graph_runner,
            binding,
        )
        for i, op in enumerate(ops):
            binds[i](op.ret)
            op.on_finish()
            self.barrier_maintainer.remove_managed_op(op.context_id)
        self.memory_modeling.merge_all()

    async def hidden_state_to_sampling_output(self, binding: str, ops: List[OperateHiddenStateToSamplingOutput]):
        model_tail = self._base_tails[binding]
        for op in ops:
            self.memory_modeling.hidden_state_to_sampling_output(op)
            op.on_start()
        await tail.hidden_states_to_sampling_output(
            model_tail,
            self.container,
            ops,
            self.tail_decode_graph_runner,
            binding,
        )
        for op in ops:
            op.on_finish()
            self.barrier_maintainer.remove_managed_op(op.context_id)
        self.memory_modeling.merge_all()

    async def do_prefill(self, binding: str, ops: List[OperateDoPrefill]):
        cache_engine = self.get_decoders_kvcache_manager(binding)
        decoders = self._base_decoders[binding]
        binds: List[Callable[[Any], None]] = []
        for op in ops:
            binds.append(self.memory_modeling.do_prefill(op)[1])
            op.on_start()
        await infer_binding.run_prefilling(
            operates=[(o, self.decoders_infer_bindings[o.binding][1])
                      for o in ops],
            container=self.container,
            cache_engine=cache_engine,
            decoders=decoders
        )
        for i, op in enumerate(ops):
            binds[i](op.ret)
            op.on_finish()
            self.barrier_maintainer.remove_managed_op(op.context_id)
        self.memory_modeling.merge_all()

    async def do_generating(self,  binding: str, ops: List[OperateDoGenerating], binds: Optional[List[Callable[[Any], None]]] = None):
        cache_engine = self.get_decoders_kvcache_manager(binding)
        decoders = self._base_decoders[binding]
        modeling = binds is None
        if modeling:
            binds = []
        if flags.DEBUG_FORCE_SWAPOUT:
            related_blocks: List[int] = []
        for op in ops:
            if modeling:
                binds.append(self.memory_modeling.do_generating(op)[1])
            op.on_start()
            binding, sc, _ = cast(Tuple[str, SeqComplexMemoryModel, int],
                                  self.memory_modeling.decoders_infer_bindings[op.binding])
            self.memory_modeling.decoders_infer_bindings
            if flags.DEBUG_FORCE_SWAPOUT:
                related_blocks.extend(sc.user_model_related_blocks)
                related_blocks.extend(sc.occupied_blocks)
        await infer_binding.run_generating(
            operates=[(o, self.decoders_infer_bindings[o.binding][1])
                      for o in ops],
            container=self.container,
            cache_engine=cache_engine,
            decoders=decoders,
            decode_graph_runner=self.decode_graph_runner,
            decode_graph_binding=binding,
            decoder_backend=self.decoder_implement,
            model_heads=self._base_heads,
            model_tails=self._base_tails,
            head_decode_graph_runner=self.head_decode_graph_runner,
            tail_decode_graph_runner=self.tail_decode_graph_runner,
        )
        if flags.DEBUG_FORCE_SWAPOUT:
            related_blocks = list(set(related_blocks))
            self.swapout("swap_blocks", manager=binding, bs=related_blocks)
        for i, op in enumerate(ops):
            binds[i](op.ret)
            op.on_finish()
            self.barrier_maintainer.remove_managed_op(op.context_id)
        self.memory_modeling.merge_all()

    async def train_tensor_to_embedding_tokens(self, op: OperateTrainTensorToEmbeddingTokens, bind: Optional[Callable[[Any], None]] = None):
        binding = op.binding
        model_head = self._base_heads[binding]
        op.on_start()
        if bind is None:
            _, bind = self.memory_modeling.train_tensor_to_embedding_tokens(op)
        self.memory_modeling.merge_all()

        async def _wrap(op: OperateTrainTensorToEmbeddingTokens, container: Container):
            await head.train_tensor_to_embedding_tokens(model_head, container, op)

        def cb():
            bind(op.ret)
            op.on_finish()
            self.barrier_maintainer.remove_managed_op(op.context_id)
        await self.train_runner.run_wait(op, _wrap, cb)

    async def compute_logits_loss_and_grad(self, op: OperateComputeLogitsLossAndGrad, bind: Optional[Callable[[Any], None]] = None):
        binding = op.binding
        model_tail = self._base_tails[binding]
        op.on_start()
        if bind is None:
            _, bind = self.memory_modeling.compute_logits_loss_and_grad(op)
        self.memory_modeling.merge_all()

        async def _wrap(op: OperateComputeLogitsLossAndGrad, container: Container):
            await tail.compute_logits_loss_and_grad(model_tail, container, op)

        def cb():
            bind(op.ret)
            op.on_finish()
            self.barrier_maintainer.remove_managed_op(op.context_id)

        await self.train_runner.run_wait(op, _wrap, cb)

    async def do_forward(self, op: OperateForward):
        binding = op.binding
        binding_data = self.decoders_train_bindings[binding]
        op.on_start()
        _, bind = self.memory_modeling.forward(op)
        self.memory_modeling.merge_all()

        async def _do_forward(op: OperateForward, container: Container):
            ctx = await train_binding.do_forward(binding_data, container, op)
            self.train_ctx_swappable[(binding, op.context_id)] = ctx

        def cb():
            bind(op.ret)
            op.on_finish()
            self.barrier_maintainer.remove_managed_op(op.context_id)
        await self.train_runner.run_wait(op, _do_forward, cb)

        if flags.DEBUG_FORCE_DOWNGRAD:
            self.swapout(target="downgrad", binding=binding,
                         context_id=op.context_id)
        if flags.DEBUG_FORCE_SWAPOUT:
            self.swapout(target="train_ctx", binding=binding,
                         context_id=op.context_id)

    async def do_backward(self, op: OperateBackward, bind: Optional[Callable[[Any], None]] = None):
        binding = op.binding
        binding_data = self.decoders_train_bindings[binding]
        op.on_start()
        if bind is None:
            _, bind = self.memory_modeling.backward(op)
        self.memory_modeling.merge_all()
        if (binding, op.context_id) in self.train_ctx_swappable:
            del self.train_ctx_swappable[(binding, op.context_id)]

        async def _do_backward(op: OperateBackward, container: Container):
            await train_binding.do_backward(binding_data, container, op)

        def cb():
            bind(op.ret)
            op.on_finish()
            self.barrier_maintainer.remove_managed_op(op.context_id)
        await self.train_runner.run_wait(op, _do_backward, cb)

    async def weight_update(self, op: OperateWeightUpdate):
        binding = op.binding
        binding_data = self.decoders_train_bindings[binding]
        op.on_start()
        self.memory_modeling.weight_update(op)
        self.memory_modeling.merge_all()

        async def _weight_update(op: OperateWeightUpdate, container: Container):
            await train_binding.do_weight_update(binding_data, container, op)

        await self.train_runner.run_wait(op, _weight_update, op.on_finish)

    def swapout(self, target: str, **kwargs: Any) -> None:
        # print(f"swapout: {target} {kwargs}")
        self.memory_modeling.swapout(target=target, **kwargs)
        self.memory_modeling.merge_all()
        if target == "swap_blocks":
            if "manager" in kwargs:
                ce = self.get_decoders_kvcache_manager(kwargs["manager"])
            elif "engine" in kwargs:
                ce = self.kv_cache_engines[kwargs["engine"]]
            assert isinstance(ce, ScalableCacheEngine)
            ce.swap_out(target_blocks=kwargs["bs"])
        elif target == "release_blocks":
            ce = self.get_decoders_kvcache_manager(kwargs["manager"])
            ce.release_blocks(kwargs["bs"])
        elif target == "train_ctx":
            obj = (kwargs["binding"], kwargs["context_id"])
            ctx = self.train_ctx_swappable[obj]
            ctx.swap_out(self.container.stream)
            del self.train_ctx_swappable[obj]
        elif target == "downgrad":
            binding = kwargs["binding"]
            context_id = kwargs["context_id"]
            obj = (binding, context_id)
            ctx = self.train_ctx_swappable[obj]
            assert isinstance(ctx, ForwardCtxEachDecoder)
            dctx = ctx.downgrade()
            self.train_ctx_swappable[obj] = dctx
            self.decoders_train_bindings[binding].forward_ctxs[context_id] = dctx
