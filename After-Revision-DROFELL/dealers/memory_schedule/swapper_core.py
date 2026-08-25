from abc import ABC, abstractmethod
import collections
import random

from dataclasses import dataclass
from operators.assemble.fwd_ctx.memory import ForwardCtxEachDecoderMemoryModeling
from operators.cache_engine.memory import ScalableCacheEngineMemoryModel
from operators.cache_engine.scalable import ScalableCacheEngine
from operators.core.swappable.real import CanSwapObject
from utils.format import format_size
from utils.runtime_metrics import record_sampled_event
from ..real_basic.memory import MemoryBrain, SeqComplexMemoryModel
from ..real_basic.basic_operate import BasicBrainOperates
from .restriction import MemoryConfig, MemoryLeaf, MemoryNode
from .priority import block_eviction_priority

from typing import Any, Dict, Literal, Optional, Set, Tuple, TypeVar, List, Callable, Union, cast
from ..real_basic.basic_operate import BasicBrainOperates
from ..common.brain import (
    ListWithHook,
    OperateDoGenerating,
    OperateDoPrefill,
    OperateForward,
    OperateGetBlocks,
    OperatePutBlocks,
    OperatePutUserModel,
    OperateType,
)


OpPropType = TypeVar("OpPropType", str, int)
T = TypeVar("T")


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


class BasicBrain(BasicBrainOperates, ABC):
    def __init__(self) -> None:
        self.memory_config: MemoryConfig
        self.restriction: MemoryNode
        self.max_concurrency: int
        self.memory_modeling: MemoryBrain

    @abstractmethod
    def check_oom_with_delta(self, delta: Dict[str, int]) -> Dict[str, MemoryLeaf]:
        raise NotImplementedError

    @abstractmethod
    def update_and_check_oom_items(self) -> Dict[str, MemoryLeaf]:
        raise NotImplementedError

    @abstractmethod
    def update_and_check_oom(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def update_restriction(self) -> None:
        raise NotImplementedError


@dataclass
class InferOpBlock:
    related_blocks: List[Tuple[int, Literal['empty', 'device', 'cpu', 'ro_both']]]
    per_block_bytes: int
    cache_engine_desc: str
    memory_cache_engine: ScalableCacheEngineMemoryModel

    @property
    def related_blocks_idx(self):
        return [b[0] for b in self.related_blocks]

    @property
    def non_device_blocks(self):
        return [b[0] for b in self.related_blocks if (b[1] == "empty" or b[1] == "cpu")]

    @property
    def device_blocks(self):
        return [b[0] for b in self.related_blocks if (b[1] == "device" or b[1] == "ro_both")]

    @property
    def total_bytes_req(self):
        return self.memory_cache_engine.per_block_bytes*len(self.related_blocks)

    # @property
    # def non_device_blocks_count(self):
    #     num = 0
    #     for b in self.related_blocks:
    #         if (b[1] == "empty" or b[1] == "cpu"):
    #             num += 1
    #     return num


class SwapperCore:
    def __init__(self, base: BasicBrain) -> None:
        self.base = base
        self.train_ctx_swappable: Dict[Tuple[str, str], CanSwapObject] = {}
        self._active_forward_op_memory: Dict[Tuple[str, str], Dict[str, int]] = {}
        self._active_generating_op_block: Dict[int, InferOpBlock] = {}
        # G is always the next value. Zero is reserved for blocks that have not
        # yet participated in a committed block-access transaction.
        self._next_block_generation = 1

    def _record_infer_batch(
        self,
        operation: str,
        selected: List[Union[OperateDoGenerating, OperateDoPrefill]],
        queue_remaining: int,
        scheduling_path: str,
    ) -> None:
        contexts = []
        for op in selected:
            binding_data = self.base.decoders_infer_bindings.get(op.binding)
            sequence = getattr(
                None if binding_data is None else binding_data[1],
                "seq", None,
            )
            contexts.append({
                "context_id": op.context_id,
                "binding": str(op.binding),
                "cpu_in": op.cpu_in,
                "cpu_out": op.cpu_out,
                "num_tokens": getattr(sequence, "num_tokens", None),
                "output_count": getattr(sequence, "output_count", None),
                "head_fused": bool(
                    getattr(op, "head_task", None) is not None
                ),
                "tail_fused": bool(
                    getattr(op, "tail_task", None) is not None
                ),
            })
        record_sampled_event(
            "batch", "decoder_batch_selected",
            operation=operation,
            scheduling_path=scheduling_path,
            configured_limit=self.base.max_concurrency,
            actual_size=len(selected),
            queue_remaining=queue_remaining,
            contexts=contexts,
        )

    @property
    def next_block_generation(self) -> int:
        return self._next_block_generation

    def _commit_block_accesses(self, accesses: Dict[str, Set[int]]) -> int:
        """Commit one generation to the memory twin and real cache engines."""
        accesses = {desc: blocks for desc, blocks in accesses.items() if blocks}
        if not accesses:
            raise ValueError("a block-access transaction must contain blocks")

        generation = self._next_block_generation
        engines: List[Tuple[
            ScalableCacheEngineMemoryModel, ScalableCacheEngine, Set[int]
        ]] = []
        for desc, blocks in accesses.items():
            memory_engine = self.base.memory_modeling.kv_cache[desc]
            real_engine = self.base.kv_cache_engines[desc]
            if not isinstance(memory_engine, ScalableCacheEngineMemoryModel):
                raise TypeError(f"modeled cache engine {desc} is not scalable")
            if not isinstance(real_engine, ScalableCacheEngine):
                raise TypeError(f"real cache engine {desc} is not scalable")

            # Validate both sides before mutating either one.
            modeled = memory_engine.block_generations(blocks)
            actual = real_engine.block_generations(blocks)
            if modeled != actual:
                raise RuntimeError(
                    f"cache generation mismatch before commit for {desc}: "
                    f"modeled={modeled}, actual={actual}"
                )
            engines.append((memory_engine, real_engine, blocks))

        for memory_engine, real_engine, blocks in engines:
            memory_engine.commit_blocks_generation(blocks, generation)
            real_engine.commit_blocks_generation(blocks, generation)
            if memory_engine.block_generations(blocks) != \
                    real_engine.block_generations(blocks):
                raise RuntimeError("cache generation commit diverged")

        self._next_block_generation += 1
        return generation

    def commit_infer_batch_access(
        self, ops: List[Union[OperateDoGenerating, OperateDoPrefill]]
    ) -> int:
        """Stamp all blocks accessed by one successfully executed infer batch."""
        accesses: Dict[str, Set[int]] = collections.defaultdict(set)
        for op in ops:
            modeled_binding, modeled_sc, _ = cast(
                Tuple[str, SeqComplexMemoryModel, int],
                self.base.memory_modeling.decoders_infer_bindings[op.binding],
            )
            real_binding, real_sc = self.base.decoders_infer_bindings[op.binding]
            modeled_desc = self.base.memory_modeling.decoders_blocks_desc[
                modeled_binding
            ]
            real_desc = self.base.decoders_blocks_desc[real_binding]
            if modeled_desc != real_desc:
                raise RuntimeError(
                    f"cache engine mismatch for {op.binding}: "
                    f"modeled={modeled_desc}, actual={real_desc}"
                )
            modeled_blocks = set(
                modeled_sc.occupied_blocks + modeled_sc.user_model_related_blocks
            )
            real_blocks = set(
                real_sc.occupied_blocks + real_sc.user_model_related_blocks
            )
            if modeled_blocks != real_blocks:
                raise RuntimeError(
                    f"cache block mapping mismatch for {op.binding}: "
                    f"modeled={sorted(modeled_blocks)}, "
                    f"actual={sorted(real_blocks)}"
                )
            accesses[modeled_desc].update(modeled_blocks)
        return self._commit_block_accesses(accesses)

    def commit_transfer_access(
        self, op: Union[OperatePutBlocks, OperateGetBlocks]
    ) -> int:
        """Commit one explicit put/get block transaction."""
        modeled_binding, modeled_sc, _ = cast(
            Tuple[str, SeqComplexMemoryModel, int],
            self.base.memory_modeling.decoders_infer_bindings[op.binding],
        )
        real_binding, real_sc = self.base.decoders_infer_bindings[op.binding]
        modeled_desc = self.base.memory_modeling.decoders_blocks_desc[
            modeled_binding
        ]
        real_desc = self.base.decoders_blocks_desc[real_binding]
        if modeled_desc != real_desc:
            raise RuntimeError(
                f"cache engine mismatch for {op.binding}: "
                f"modeled={modeled_desc}, actual={real_desc}"
            )
        modeled_blocks = {modeled_sc.block_mapping[b] for b in op.seq_blocks}
        real_blocks = {real_sc.block_mapping[b] for b in op.seq_blocks}
        if modeled_blocks != real_blocks:
            raise RuntimeError(
                f"cache block mapping mismatch for {op.binding}: "
                f"modeled={sorted(modeled_blocks)}, actual={sorted(real_blocks)}"
            )
        return self._commit_block_accesses({modeled_desc: modeled_blocks})

    def commit_user_model_access(self, op: OperatePutUserModel) -> int:
        """Commit the block write performed while installing a user model."""
        if op.ret is None:
            raise RuntimeError("put_user_model completed without a block handle")
        desc = self.base.memory_modeling.decoders_blocks_desc[op.binding]
        real_desc = self.base.decoders_blocks_desc[op.binding]
        if desc != real_desc:
            raise RuntimeError(
                f"cache engine mismatch for {op.binding}: "
                f"modeled={desc}, actual={real_desc}"
            )
        return self._commit_block_accesses({desc: set(op.ret.blocks)})

    def _get_forward_memory_change(self, op: OperateForward):
        memory_change = self._active_forward_op_memory.get((op.binding, op.context_id))
        if memory_change is None:
            before_memory = self.base.memory_modeling.env.pools.copy()
            assert self.base.memory_modeling.all_merged
            self.base.memory_modeling.forward(op)[0]()
            after_memory = self.base.memory_modeling.env.pools.copy()
            self.base.memory_modeling.revert_all()
            memory_change = {
                k: current-before_memory[k]
                for k, current in after_memory.items()
                if current-before_memory[k] != 0
            }
            self._active_forward_op_memory[(op.binding, op.context_id)] = memory_change
        return memory_change

    def _swap_train_ctx(self, req_memory: int) -> int:
        # try to downgrad all ctx, then try to swap all
        downgradable_keys: List[Tuple[str, str]] = []
        swappable_keys: List[Tuple[str, str]] = []
        for k, v in self.base.memory_modeling.train_ctx_swappable.childrens.items():
            if isinstance(v, ForwardCtxEachDecoderMemoryModeling):
                downgradable_keys.append(k)
            else:
                swappable_keys.append(k)
        current_memory = self.base.memory_modeling.pools[self.base.memory_config.train_ctx_name]

        def swapped_size():
            return current_memory-self.base.memory_modeling.pools[self.base.memory_config.train_ctx_name]

        downgrad: List[Tuple[str, str]] = []
        swap: List[Tuple[str, str]] = []

        def do_swap():
            self.base.memory_modeling.revert_all()
            for k in downgrad:
                self.base.swapout(target="downgrad", binding=k[0], context_id=k[1])
            for k in swap:
                self.base.swapout(target="train_ctx", binding=k[0], context_id=k[1])

        for k in downgradable_keys:
            downgrad.append(k)
            swappable_keys.append(k)
            print(f"downgrad: binding={k[0]}, context_id={k[1]}")
            self.base.memory_modeling.swapout(target="downgrad", binding=k[0], context_id=k[1])
            if swapped_size() > req_memory:
                do_swap()
                return swapped_size()
        for k in swappable_keys:
            swap.append(k)
            if k in downgrad:
                downgrad.remove(k)
            print(f"swapout: binding={k[0]}, context_id={k[1]}")
            self.base.memory_modeling.swapout(target="train_ctx", binding=k[0], context_id=k[1])
            if swapped_size() > req_memory:
                do_swap()
                return swapped_size()
        do_swap()
        return swapped_size()

    async def _forward(self, op: OperateForward, swap_memory: int):
        if swap_memory > 0:
            self._swap_train_ctx(swap_memory)
        del self._active_forward_op_memory[(op.binding, op.context_id)]
        await self.base.do_forward(op)

    def _maximum_train_swap(self):
        assert self.base.memory_modeling.all_merged
        before_memory = self.base.memory_modeling.pools[self.base.memory_config.train_ctx_name]
        for k in list(self.base.memory_modeling.train_ctx_swappable.childrens.keys()):
            self.base.memory_modeling.swapout(target="train_ctx", binding=k[0], context_id=k[1])
        maximum_swap_memory = before_memory-self.base.memory_modeling.pools[self.base.memory_config.train_ctx_name]
        self.base.memory_modeling.revert_all()
        return maximum_swap_memory

    def rand_enumerate(self, seq: List[OperateType]):
        n = len(seq)
        for idx in random.sample(range(n), n):
            yield idx, seq[idx]

    def _select_forward(self, with_swap: bool, debug: Optional[str] = None):
        maximum_swap_memory = 0
        if with_swap:
            maximum_swap_memory = self._maximum_train_swap()
        for idx, op in self.rand_enumerate(self.base.operates_forward.list):
            memory_change = self._get_forward_memory_change(op)
            oom_items = self.base.check_oom_with_delta(memory_change)

            if len(oom_items) == 0:
                self.base.operates_forward.list.pop(idx)
                return op, 0
            if not with_swap:
                continue
            required_memory = max(-v.available for v in oom_items.values())
            memory_change_if_swap = {
                k: v for k, v in memory_change.items()
            }
            memory_change_if_swap[self.base.memory_config.train_ctx_name] =\
                memory_change_if_swap.get(self.base.memory_config.train_ctx_name, 0)-required_memory
            oom_items_if_swap = self.base.check_oom_with_delta(memory_change_if_swap)
            if len(oom_items_if_swap) != 0:
                for k, v in oom_items_if_swap.items():
                    print(f"{k} requires extra {format_size(-v.available)} memory, pending")
                continue
            # need to make required_memory from train swappable
            if maximum_swap_memory > required_memory:
                self.base.operates_forward.list.pop(idx)
                return op, required_memory
            if debug is not None:
                print(f"fail to operate forward: {op}")
                print(f"maximum swappable: {maximum_swap_memory}")
                print(f"require: {required_memory}")
        return None

    async def do_forward(self, debug: Optional[str] = None) -> bool:
        # first, modeling new memory acquired
        selected = self._select_forward(with_swap=False)
        if selected is None:
            selected = self._select_forward(with_swap=True, debug=debug)
            if selected is None:
                return False
            else:
                print("swap forward")
        op, swap_memory = selected
        await self._forward(op, swap_memory)
        return True

    def _select_prefill_no_swap(self):
        selected: List[OperateDoPrefill] = []
        binding: Optional[str] = None
        recovery_decode: Optional[bool] = None

        def read_fn(op: OperateDoPrefill):
            return self.base.decoders_infer_bindings[op.binding][0]
        while len(selected) < self.base.max_concurrency and len(self.base.operates_do_prefill):
            op = self.base.operates_do_prefill.list[0]
            prop = read_fn(op)
            if binding is not None and binding != prop:
                break
            if recovery_decode is not None \
                    and recovery_decode != op.recovery_decode:
                break
            self.base.memory_modeling.do_prefill(op)[0]()
            if self.base.update_and_check_oom():
                break
            if binding is None:
                binding = prop
                recovery_decode = op.recovery_decode
            self.base.operates_do_prefill.pop(0)
            selected.append(op)
        self.base.memory_modeling.revert_all()
        if len(selected) > 0:
            assert binding is not None
            return binding, selected
        return None

    async def prefill_no_swap(self) -> bool:
        sec = self._select_prefill_no_swap()
        if sec is None:
            return False
        binding, selected = sec
        self._record_infer_batch(
            "decoder_prefill", selected,
            len(self.base.operates_do_prefill), "no_swap",
        )
        await self.base.do_prefill(binding, ops=selected)
        self.commit_infer_batch_access(selected)
        return True

    def _select_generating_no_swap(self):
        selected: List[OperateDoGenerating] = []
        binding: Optional[str] = None
        binds: List[Callable[[Any], None]] = []

        def read_fn(op: OperateDoGenerating):
            return self.base.decoders_infer_bindings[op.binding][0]
        while len(selected) < self.base.max_concurrency and len(self.base.operates_do_generating):
            op = self.base.operates_do_generating.list[0]
            prop = read_fn(op)
            if binding is not None and binding != prop:
                break
            snapshot = self.base.memory_modeling.snapshot()
            vb, b = self.base.memory_modeling.do_generating(op)
            vb()
            if self.base.update_and_check_oom():
                snapshot.apply()
                break
            if binding is None:
                binding = prop
            binds.append(b)
            self.base.operates_do_generating.pop(0)
            selected.append(op)
            if id(op) in self._active_generating_op_block:
                del self._active_generating_op_block[id(op)]
        if len(selected) > 0:
            assert binding is not None
            return binding, selected, binds
        return None

    async def generating_no_swap(self) -> bool:
        sec = self._select_generating_no_swap()
        if sec is None:
            return False
        binding, selected, binds = sec
        self._record_infer_batch(
            "decoder_generating", selected,
            len(self.base.operates_do_generating), "no_swap",
        )
        await self.base.do_generating(binding, ops=selected, binds=binds)
        self.commit_infer_batch_access(selected)
        return True

    def get_infer_op_blocks_info(self, op: Union[OperateDoGenerating, OperateDoPrefill]):
        if id(op) in self._active_generating_op_block:
            return self._active_generating_op_block[id(op)]
        # active prefill task tasks no device block
        binding, sc, _ = cast(
            Tuple[str, SeqComplexMemoryModel, int],
            self.base.memory_modeling.decoders_infer_bindings[op.binding]
        )
        cache_engine_desc = self.base.memory_modeling.decoders_blocks_desc[binding]
        memory_cache_engine: ScalableCacheEngineMemoryModel = self.base.memory_modeling.kv_cache[cache_engine_desc]
        related_blocks = sc.occupied_blocks+sc.user_model_related_blocks
        per_block_bytes = memory_cache_engine.per_block_bytes
        info = InferOpBlock(
            related_blocks=[
                (b, memory_cache_engine.local_blocks[b].state)
                for b in related_blocks
            ],
            per_block_bytes=per_block_bytes,
            cache_engine_desc=cache_engine_desc,
            memory_cache_engine=memory_cache_engine
        )
        if isinstance(op, OperateDoGenerating):
            self._active_generating_op_block[id(op)] = info
        return info

    async def _swap_blocks(
        self,
        req_memory: int,
        non_free_blocks: Tuple[str, List[int]]
    ):

        activate_blocks: Dict[str, Set[int]] = collections.defaultdict(set)
        for op in self.base.operates_do_generating.list:
            info = self.get_infer_op_blocks_info(op)
            for b in info.device_blocks:
                activate_blocks[info.cache_engine_desc].add(b)

        swappable_blocks: List[Tuple[str, int, int]] = []
        cache_engines: Dict[str, ScalableCacheEngineMemoryModel] = {}
        for _d, _c in self.base.memory_modeling.kv_cache.childrens.items():
            desc: str = _d
            cache_engine: ScalableCacheEngineMemoryModel = _c
            cache_engines[desc] = cache_engine
            for b in cache_engine.device_local_blocks:
                if desc == non_free_blocks[0] and b in non_free_blocks[1]:
                    continue
                blk = cache_engine.local_blocks[b]
                if blk.state == "ro_both":
                    swappable_blocks.append((desc, b, 2))
                elif blk.state == "device":
                    swappable_blocks.append((desc, b, (1 if blk.ro else 0)))
                else:
                    raise ValueError(blk.state)
        swappable_blocks = list(set(swappable_blocks))

        def priority(b: Tuple[str, int, int]):
            desc, idx, readonly_level = b
            blk = cache_engines[desc].local_blocks[idx]
            return block_eviction_priority(
                generation=blk.generation,
                next_generation=self._next_block_generation,
                readonly_level=readonly_level,
                active=idx in activate_blocks[desc],
                engine_desc=desc,
                preferred_engine_desc=non_free_blocks[0],
                block_idx=idx,
            )

        swappable_blocks.sort(key=priority)
        current_memory = self.base.memory_modeling.pools[self.base.memory_config.kv_cache_name]

        def swapped_size():
            return current_memory-self.base.memory_modeling.pools[self.base.memory_config.kv_cache_name]

        target_blocks: Dict[str, List[int]] = collections.defaultdict(list)
        swap_ok = False
        for (desc, idx, _) in swappable_blocks:
            cache_engines[desc].swap_out_block(idx)
            target_blocks[desc].append(idx)
            if swapped_size() >= req_memory:
                swap_ok = True
                break
        assert swap_ok, (
            f"unable to release {format_size(req_memory)} from KV blocks; "
            f"released {format_size(swapped_size())}"
        )
        self.base.memory_modeling.revert_all()
        for key, blocks in target_blocks.items():
            self.base.swapout(target="swap_blocks", engine=key, bs=blocks)

    def _select_generating_swap(self):
        self.base.update_restriction()
        maximum_possible = self.base.restriction.leaves[self.base.memory_config.kvcache_memory_key].total
        binding: Optional[str] = None
        selected: List[OperateDoGenerating] = []
        cache_engine_desc = ""
        max_buffer_bytes = 0
        preserve_blocks: List[int] = []

        def read_fn(op: OperateDoGenerating):
            return self.base.decoders_infer_bindings[op.binding][0]
        while len(selected) < self.base.max_concurrency and len(self.base.operates_do_generating):
            op = self.base.operates_do_generating.list[0]
            prop = read_fn(op)
            if binding is not None and binding != prop:
                break
            info = self.get_infer_op_blocks_info(op)
            if info.total_bytes_req > (maximum_possible-info.memory_cache_engine.max_buffer_bytes):
                break
            maximum_possible -= info.total_bytes_req
            if binding is None:
                binding = prop
            cache_engine_desc = info.cache_engine_desc
            max_buffer_bytes = info.memory_cache_engine.max_buffer_bytes
            preserve_blocks.extend(info.device_blocks)
            self.base.operates_do_generating.pop(0)
            selected.append(op)

        for op in selected:
            self.base.memory_modeling.do_generating(op)[0]()
        assert binding is not None

        oom_items = self.base.update_and_check_oom_items()
        required_memory = -oom_items[self.base.memory_config.kvcache_memory_key].available
        # as buffer blocks must be insufficient for required memory,
        # buffer blocks will be refill
        required_memory += max_buffer_bytes
        self.base.memory_modeling.revert_all()
        return binding, selected, required_memory, (cache_engine_desc, preserve_blocks)

    async def generating_swap(self, debug: Optional[str] = None) -> bool:
        binding, selected, required_memory, non_free_blocks = self._select_generating_swap()
        swapout_bytes = self._swap_train_ctx(required_memory)
        required_memory -= swapout_bytes
        if required_memory > 0:
            await self._swap_blocks(
                req_memory=required_memory,
                non_free_blocks=non_free_blocks
            )
        print(f"swap generating {len(selected)} tasks")
        self._record_infer_batch(
            "decoder_generating", selected,
            len(self.base.operates_do_generating), "swap",
        )
        await self.base.do_generating(binding, ops=selected)
        self.commit_infer_batch_access(selected)
        return True

    def _select_prefill_swap(self):
        self.base.update_restriction()
        maximum_possible = self.base.restriction.leaves[self.base.memory_config.kvcache_memory_key].total
        binding: Optional[str] = None
        selected: List[OperateDoPrefill] = []
        recovery_decode: Optional[bool] = None
        cache_engine_desc = ""
        max_buffer_bytes = 0
        preserve_blocks: List[int] = []

        def read_fn(op: OperateDoPrefill):
            return self.base.decoders_infer_bindings[op.binding][0]
        while len(selected) < self.base.max_concurrency and len(self.base.operates_do_prefill):
            op = self.base.operates_do_prefill.list[0]
            prop = read_fn(op)
            if binding is not None and binding != prop:
                break
            if recovery_decode is not None \
                    and recovery_decode != op.recovery_decode:
                break
            info = self.get_infer_op_blocks_info(op)
            if info.total_bytes_req > (maximum_possible-info.memory_cache_engine.max_buffer_bytes):
                break
            maximum_possible -= info.total_bytes_req
            if binding is None:
                binding = prop
                recovery_decode = op.recovery_decode
            cache_engine_desc = info.cache_engine_desc
            max_buffer_bytes = info.memory_cache_engine.max_buffer_bytes
            preserve_blocks.extend(info.device_blocks)
            self.base.operates_do_prefill.pop(0)
            selected.append(op)

        for op in selected:
            self.base.memory_modeling.do_prefill(op)[0]()
        assert binding is not None
        oom_items = self.base.update_and_check_oom_items()
        required_memory = -oom_items[self.base.memory_config.kvcache_memory_key].available
        # as buffer blocks must be insufficient for required memory,
        # buffer blocks will be refill
        required_memory += max_buffer_bytes
        self.base.memory_modeling.revert_all()
        return binding, selected, required_memory, (cache_engine_desc, preserve_blocks)

    async def prefill_swap(self, debug: Optional[str] = None) -> bool:
        binding, selected, required_memory, non_free_blocks = self._select_prefill_swap()
        swapout_bytes = self._swap_train_ctx(required_memory)
        required_memory -= swapout_bytes
        if required_memory > 0:
            await self._swap_blocks(
                req_memory=required_memory,
                non_free_blocks=non_free_blocks
            )
        print(f"swap prefill {len(selected)} tasks")
        self._record_infer_batch(
            "decoder_prefill", selected,
            len(self.base.operates_do_prefill), "swap",
        )
        await self.base.do_prefill(binding, ops=selected)
        self.commit_infer_batch_access(selected)
        return True

    async def do_prefill_or_generating(self, debug: Optional[str] = None) -> bool:
        if len(self.base.operates_do_prefill):
            if await self.prefill_no_swap():
                return True
        if len(self.base.operates_do_generating):
            if await self.generating_no_swap():
                return True
        if len(self.base.operates_do_generating):
            if await self.generating_swap(debug):
                return True
        if len(self.base.operates_do_prefill):
            if await self.prefill_swap(debug):
                return True
        return False
