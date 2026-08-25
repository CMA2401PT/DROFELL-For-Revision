from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, cast

import torch

import extend_asyncio as asyncio
from dealers.common.base_types import DeviceTensorCarrier
from dealers.common.brain import (
    Brain,
    DecodeGeneratingResult,
    OperateCreateDecodersInferBinding,
    OperateDoGenerating,
    OperateDoPrefill,
    OperateGetBlocks,
    OperateHiddenStateToSamplingOutput,
    OperateInferTensorToEmbeddingTokens,
    OperatePutBlocks,
    OperatePutUserModel,
    Profile,
    UserModelHandle,
)
from dealers.real_basic.basic import select_multi_with_group
from dealers.real_basic.memory import SeqComplexMemoryModel
from dealers.memory_schedule.restriction import MemoryConfig, MemoryConfigStatic
from operators import flags as operator_flags
from utils.runtime_metrics import record_event

from .clock import VirtualClock
from .cost import CostModel

META = torch.device("meta")


class _UnderlayDecoders:
    """The attributes production code reads off a real ``DecodersGroup``.

    ``DecodersShim.__init__`` reads ``.decoder_config``; ``Brain.gen_profile``
    awaits ``.gen_profile``. Nothing else on the simulated decode path touches
    the underlay, because ``SimBackend`` supplies execution itself. Exposing
    exactly these keeps any other access loud.
    """

    def __init__(self, memory_model: Any, cost: CostModel, layers: int) -> None:
        self._memory_model = memory_model
        self._cost = cost
        self._layers = layers
        # `DecodersGroupMemoryModel.config` already IS the DecoderConfig (it
        # returns `decoders[0].cfg`), whereas the real `DecodersGroup` exposes
        # `.decoder_config`. Accept either so this keeps working if the twin's
        # accessor is renamed to match.
        self.decoder_config = getattr(
            memory_model.config, "decoder_config", memory_model.config
        )

    @property
    def num_layers(self) -> int:
        return self._layers

    @property
    def block_info(self) -> Any:
        return self._memory_model.block_info

    async def gen_profile(
        self, container: Any = None, **kwargs: Any
    ) -> Tuple[int, int, int]:
        """Per-decoder throughput, derived from the calibrated cost model.

        The real routine times a microbenchmark on the GPU. Deriving it from the
        same service term the simulated schedule will charge keeps the Aux
        directory and the simulated execution consistent by construction; a
        hard-coded number could disagree with the schedule and silently distort
        provider selection.
        """
        per_layer_us = self._cost.per_layer_us()
        tps = max(1, int(1e6 / max(per_layer_us, 1e-6)))
        return tps, max(1, tps // 3), tps


@dataclass
class DeviceModel:
    """One simulated accelerator. Operations on it serialise.

    ``free_at_ns`` is the reservation cursor: the mechanism that makes N logical
    nodes sharing one device contend, which is the whole point of the D-2
    "same compute, more nodes" configuration (docs/00 §2.0).
    """

    device_id: str
    memory_bytes: int
    free_at_ns: int = 0
    copy_free_at_ns: int = 0
    busy_ns: int = 0
    copy_busy_ns: int = 0
    operations: int = 0
    copy_operations: int = 0


@dataclass
class TransferDomain:
    """Host-level PCIe/root-complex and memory-copy contention domain.

    Per-device copy engines are not independent sources of host memory/PCIe
    capacity. All devices mapped to one physical host reserve this domain as
    well. H2D and D2H use separate full-duplex cursors.
    """

    domain_id: str
    free_at_ns: Dict[str, int] = field(default_factory=dict)
    busy_ns: Dict[str, int] = field(default_factory=dict)
    operations: Dict[str, int] = field(default_factory=dict)
    bytes_total: Dict[str, int] = field(default_factory=dict)


@dataclass
class BackendStats:
    operations: int = 0
    by_stage: Dict[str, int] = field(default_factory=dict)
    merged: Dict[str, List[int]] = field(default_factory=dict)
    service_ns: Dict[str, int] = field(default_factory=dict)
    device_contention_ns: int = 0
    transfer_contention_ns: int = 0
    transfer_ns: Dict[str, int] = field(default_factory=dict)
    transfer_bytes: Dict[str, int] = field(default_factory=dict)
    transfer_operations: Dict[str, int] = field(default_factory=dict)
    oom_deferrals: int = 0

    def record(self, stage: str, batch: int, service_ns: int) -> None:
        self.operations += 1
        self.by_stage[stage] = self.by_stage.get(stage, 0) + 1
        self.merged.setdefault(stage, []).append(batch)
        self.service_ns[stage] = self.service_ns.get(stage, 0) + service_ns


class SimBackend(Brain):
    """Production ``Brain`` contract; virtual-time execution."""

    def __init__(
        self,
        node_name: str,
        clock: VirtualClock,
        cost: CostModel,
        device: DeviceModel,
        transfer_domain: TransferDomain,
        models_path: str,
        max_seq_len: int = 4096,
        max_concurrency: int = 8,
        memory_config: Optional[MemoryConfig] = None,
        decoder_implement: str = "marlin",
    ) -> None:
        # A real torch device is never touched; ``torch.device("meta")`` keeps
        # every `with torch.device(...)` block in the production loaders valid
        # while allocating nothing.
        super().__init__(
            models_path,
            META,
            max_seq_len=max_seq_len,
            max_concurrency=max_concurrency,
            decoder_implement=cast(Any, decoder_implement),
        )
        self.node_name = node_name
        self.clock = clock
        self.cost = cost
        self.device_model = device
        self.transfer_domain = transfer_domain
        self.stats = BackendStats()

        if memory_config is None:
            memory_config = MemoryConfigStatic()
        memory_config.root_memory = device.memory_bytes
        self.memory_config = memory_config

        # The digital twin: real admission logic, zero CUDA bytes.
        from dealers.real_basic.memory import MemoryBrain

        self.memory_modeling = MemoryBrain(
            models_path,
            memory_config,
            max_seq_len=max_seq_len,
            decoder_implement=cast(Any, decoder_implement),
            device=None,
        )
        self.restriction = memory_config.restriction(self.memory_modeling.env)
        self.restriction.update(memory_config.root_memory)

        # Mirrors `Brain.__init__` in dealers/memory_schedule/brain.py: the
        # grouping key per queue. Merge sizes then come from the production
        # `select_multi_with_group` rather than from a simulator choice.
        self.operates_create_decoders_infer_binding.compare_fn = \
            lambda op: str(op.binding)
        self.operates_put_user_model.compare_fn = lambda op: str(op.binding)
        self.operates_put_blocks.compare_fn = \
            lambda op: str(self.decoders_infer_bindings[op.binding][0])
        self.operates_get_blocks.compare_fn = \
            lambda op: str(self.decoders_infer_bindings[op.binding][0])
        self.operates_infer_tensor_to_embedding_tokens.compare_fn = \
            lambda op: str(op.binding)
        self.operates_hidden_state_to_sampling_output.compare_fn = \
            lambda op: str(op.binding)
        self.operates_do_generating.compare_fn = \
            lambda op: str(self.decoders_infer_bindings[op.binding][0])
        self.operates_do_prefill.compare_fn = lambda op: (
            f"{self.decoders_infer_bindings[op.binding][0]}:"
            f"recovery={int(op.recovery_decode)}"
        )

        # infer_binding_name -> (decoders_binding, SeqComplexMemoryModel)
        self.decoders_infer_bindings: Dict[str, Tuple[str, Any]] = {}
        self.decoders_blocks_desc: Dict[str, str] = {}
        self._underlay: Dict[str, _UnderlayDecoders] = {}
        self._layers_of: Dict[str, int] = {}
        self._incremental_idx = 0
        # Faked sampling: deterministic so two runs of one scenario agree.
        self._token_counter = 1000
        self._closed = False
        self._loop_task = asyncio.create_task(self.loop())

    # -- model management -------------------------------------------------
    def make_model_head(
        self, base_model: str, adjacent_decoder: Optional[str] = None
    ) -> str:
        self.memory_modeling.make_model_head(base_model)
        self.memory_modeling.merge_all()
        from dealers.common.base_types import ModelDescribe

        name = str(ModelDescribe.new(base_model, is_head=True))
        self._base_heads.setdefault(name, cast(Any, object()))
        self._reinit_restriction()
        return name

    def make_model_tail(
        self, base_model: str, adjacent_decoder: Optional[str] = None
    ) -> str:
        self.memory_modeling.make_model_tail(base_model)
        self.memory_modeling.merge_all()
        from dealers.common.base_types import ModelDescribe

        name = str(ModelDescribe.new(base_model, is_tail=True))
        self._base_tails.setdefault(name, cast(Any, object()))
        self._reinit_restriction()
        return name

    def make_model_decoders(
        self, base_model: str, decoders_range: Tuple[int, int]
    ) -> str:
        self.memory_modeling.make_model_decoders(base_model, decoders_range)
        self.memory_modeling.merge_all()
        from dealers.common.base_types import ModelDescribe

        name = str(
            ModelDescribe.new(
                base_model, start=decoders_range[0], end=decoders_range[1]
            )
        )
        layers = decoders_range[1] - decoders_range[0]
        self._layers_of[name] = layers
        memory_model = self.memory_modeling.base_models[name].item
        self._base_decoders[name] = cast(Any, memory_model)
        self._underlay[name] = _UnderlayDecoders(memory_model, self.cost, layers)
        block_info = memory_model.block_info
        self.decoders_blocks_desc[name] = (
            f"{block_info.num_key_value_heads}*{block_info.head_dim}*{layers}"
        )
        self._reinit_restriction()
        return name

    def get_underlay_decoders(self, name: str) -> Any:
        return self._underlay[name]

    def get_underlay_model_head(self, name: str) -> Any:
        return self._base_heads[name]

    def get_underlay_model_tail(self, name: str) -> Any:
        return self._base_tails[name]

    def remove_model_head(self, binding: str) -> None:
        self.memory_modeling.remove_model_head(binding)
        self.memory_modeling.merge_all()
        self._base_heads.pop(binding, None)

    def remove_model_tail(self, binding: str) -> None:
        self.memory_modeling.remove_model_tail(binding)
        self.memory_modeling.merge_all()
        self._base_tails.pop(binding, None)

    def remove_model_decoders(self, binding: str) -> None:
        self.memory_modeling.remove_model_decoders(binding)
        self.memory_modeling.merge_all()
        self._base_decoders.pop(binding, None)
        self._underlay.pop(binding, None)
        self.decoders_blocks_desc.pop(binding, None)

    def _reinit_restriction(self) -> None:
        self.restriction = self.memory_config.restriction(
            self.memory_modeling.env
        )
        self.restriction.update(self.memory_config.root_memory)

    def _next_unique_id(self) -> str:
        value = self._incremental_idx
        self._incremental_idx += 1
        return str(value)

    # -- Brain contract used by DealerNode --------------------------------
    @property
    def memory(self) -> Tuple[int, int]:
        return (
            self.memory_config.root_memory,
            self.memory_modeling.env.pools[self.memory_config.base_models_name],
        )

    def query_extra_info(self, key: str) -> Any:
        if key == "detail_memory":
            return self.memory_modeling.env.detail_memory()
        if key == "detail_memory_restriction":
            return str(self.restriction)
        return None

    async def gen_profile(
        self,
        model_name: str,
        train_kwargs: Optional[Dict[str, Any]] = None,
        generating_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Profile:
        """Same shape as the production routine: measure one layer, then two.

        Taking the difference of a 1-layer and a 3-layer binding isolates the
        per-decoder memory exactly as ``memory_schedule/brain.py`` does, instead
        of dividing a total and inheriting the head/tail overhead.
        """
        binding = self.make_model_decoders(model_name, (0, 1))
        memory_one = self.memory_modeling.base_models.get(binding).item.memory
        binding_two = self.make_model_decoders(model_name, (1, 3))
        memory_two = self.memory_modeling.base_models.get(binding_two).item.memory
        per_decoder_memory = memory_two - memory_one
        forward, forward_backward, generating = await self.get_underlay_decoders(
            binding
        ).gen_profile(None)
        self.remove_model_decoders(binding)
        self.remove_model_decoders(binding_two)
        return Profile(
            perdecoder_memory=per_decoder_memory,
            perdecoder_forward_tps=forward,
            perdecoder_frowarad_backward_tps=forward_backward,
            perdecoder_generating_tps=generating,
        )

    def capture_decode_graph(self) -> None:
        """CUDA-graph capture has no analogue without a GPU.

        The measured effect of capture is on service time, which the cost model
        already carries (it was fitted on runs WITH capture enabled), so there is
        nothing further to model. Raising here instead would block every caller
        that merely mirrors the production flag.
        """
        return None

    def get_decoders_kvcache_manager(self, binding: str) -> Any:
        return self.memory_modeling.get_decoders_kvcache_manager(binding)

    def swapout(self, target: str, **kwargs: Any) -> None:
        self.memory_modeling.swapout(target, **kwargs)

    def remove_decoders_infer_binding(self, binding: str) -> None:
        # Production shim destructors may run after the short-lived simulated
        # deployment has already completed its explicit remote teardown. Ignore
        # only that post-close duplicate. During an active deployment an unknown
        # binding remains a hard lifecycle error.
        if binding not in self.decoders_infer_bindings:
            if self._closed:
                return
            raise KeyError(f"unknown decoder infer binding: {binding}")
        self.memory_modeling.remove_decoders_infer_binding(binding)
        self.memory_modeling.merge_all()
        self.decoders_infer_bindings.pop(binding)

    def remove_decoders_train_binding(self, binding: str) -> None:
        raise NotImplementedError(
            "DROSIM models the inference path only; a training binding means "
            "the scenario reached code outside the modelled set (docs/02 §4.4)."
        )

    # -- scheduling -------------------------------------------------------
    async def loop(self) -> None:
        """Drain the operate queues, mirroring the production scheduler.

        Structure follows ``memory_schedule/brain.py::loop``: wait for work,
        wait on the barrier, then run exactly one operate. Two production steps
        are absent by construction rather than by omission -- there is no
        ``container.step`` (no CUDA stream to order against) and no swapper
        (swap-out is only reachable under real memory pressure the twin reports
        but the simulator never provokes).
        """
        while not self._closed:
            await self.wait_task()
            if self._closed:
                return
            await self.barrier_maintainer.wait()
            if await self._operate():
                continue
            # Nothing was admissible: the twin says every pending operate would
            # OOM. Production prints and parks on `wait_new_operate`; the same
            # park is needed here, and under a virtual clock it MUST be
            # event-driven -- a timeout poll would advance the clock forever
            # (docs/06 P10, the hang we hit).
            self.stats.oom_deferrals += 1
            record_event(
                "sim_memory_deferral",
                node=self.node_name,
                pending=self._pending_summary(),
            )
            await self._wait()

    def _pending_summary(self) -> Dict[str, int]:
        return {
            name: len(queue.list)
            for name, queue in self.all_queues
            if len(queue.list)
        }

    async def _operate(self) -> bool:
        """One scheduling decision, in the production priority order."""
        if len(self.operates_put_blocks):
            op = self.operates_put_blocks.pop(0)
            await self._run_put_blocks(op)
            return True
        if len(self.operates_put_user_model):
            op = self.operates_put_user_model.pop(0)
            await self._run_put_user_model(op)
            return True
        if len(self.operates_get_blocks):
            op = self.operates_get_blocks.pop(0)
            await self._run_get_blocks(op)
            return True
        if len(self.operates_create_decoders_infer_binding):
            op = self._select_admissible(
                self.operates_create_decoders_infer_binding,
                self.memory_modeling.create_decoders_infer_binding,
            )
            if op is not None:
                await self._run_create_infer_binding(op)
                return True
            return False
        if len(self.operates_infer_tensor_to_embedding_tokens):
            binding, ops = select_multi_with_group(
                pendings=self.operates_infer_tensor_to_embedding_tokens,
                read_fn=lambda op: str(op.binding),
                max_concurrency=self.max_concurrency,
            )
            record_event(
                "batch_merge",
                operation="infer_head",
                configured_limit=self.max_concurrency,
                actual_size=len(ops),
            )
            await self._run_head(binding, ops)
            return True
        if len(self.operates_hidden_state_to_sampling_output):
            binding, ops = select_multi_with_group(
                pendings=self.operates_hidden_state_to_sampling_output,
                read_fn=lambda op: str(op.binding),
                max_concurrency=self.max_concurrency,
            )
            record_event(
                "batch_merge",
                operation="infer_tail",
                configured_limit=self.max_concurrency,
                actual_size=len(ops),
            )
            await self._run_tail(binding, ops)
            return True
        if len(self.operates_do_prefill):
            _group_key, ops = select_multi_with_group(
                pendings=self.operates_do_prefill,
                read_fn=lambda op: (
                    f"{self.decoders_infer_bindings[op.binding][0]}:"
                    f"recovery={int(op.recovery_decode)}"
                ),
                max_concurrency=self.max_concurrency,
            )
            record_event(
                "batch_merge",
                operation="decoder_prefill",
                configured_limit=self.max_concurrency,
                actual_size=len(ops),
            )
            # The grouping key contains a recovery-mode suffix so normal
            # prefill and recovery decode never merge. It is not a model
            # binding. Passing it to `_run_prefill` used the one-layer fallback
            # and undercharged every prefill instead of using the real shard.
            binding = self.decoders_infer_bindings[ops[0].binding][0]
            await self._run_prefill(binding, ops)
            return True
        if len(self.operates_do_generating):
            binding, ops = select_multi_with_group(
                pendings=self.operates_do_generating,
                read_fn=lambda op: str(
                    self.decoders_infer_bindings[op.binding][0]
                ),
                max_concurrency=self.max_concurrency,
            )
            record_event(
                "batch_merge",
                operation="decoder_generating",
                configured_limit=self.max_concurrency,
                actual_size=len(ops),
            )
            await self._run_generating(binding, ops)
            return True
        for name, queue in self.all_queues:
            if len(queue.list):
                raise NotImplementedError(
                    f"SimBackend does not model queue {name!r}: the scenario "
                    f"reached a path outside the infer set measured in "
                    f"probes/p10b. Returning without executing it would hang "
                    f"the awaiting shim silently (docs/02 §4.4)."
                )
        return False

    def _select_admissible(self, queue: Any, model_fn: Any) -> Any:
        """Pick the first operate the twin admits, exactly as production does.

        The twin is speculatively applied and reverted on OOM, so admission
        walks the same constraint tree with the same quotas as hardware.
        """
        for index, op in enumerate(queue.list):
            model_fn(op)
            self.restriction.update(self.memory_config.root_memory)
            if self.restriction.oom:
                self.memory_modeling.revert_all()
                continue
            queue.list.pop(index)
            return op
        return None

    # -- timing ------------------------------------------------------------
    def _reserve_device(
        self,
        stage: str,
        batch: int,
        layers: int,
        *,
        ready_ns: Optional[int] = None,
    ) -> int:
        """Enqueue work and return its completion timestamp.

        Production submits CUDA work asynchronously: the Brain can dequeue and
        submit the next operate before the previous result's sync point fires.
        Waiting here would collapse `exec->done` and `done->async`, destroy the
        dual-queue overlap, and change recovery event order. Reserve the device
        cursor immediately; callers attach a completion future only where the
        production result requires synchronization.
        """
        duration_ns = self.cost.duration_ns(stage, batch, layers=layers)
        ready = self.clock.now_ns if ready_ns is None else ready_ns
        start = max(self.device_model.free_at_ns, ready)
        contention_ns = start - ready
        if contention_ns > 0:
            self.stats.device_contention_ns += contention_ns
        finish = start + duration_ns
        self.device_model.free_at_ns = finish
        self.device_model.busy_ns += duration_ns
        self.device_model.operations += 1
        self.stats.record(stage, batch, duration_ns)
        return finish

    async def _charge_bookkeeping(self, stage: str, batch: int) -> None:
        """Advance time for work that costs CPU but does not occupy the device."""
        duration_ns = self.cost.duration_ns(stage, batch, layers=0)
        await asyncio.sleep(duration_ns / 1e9)
        self.stats.record(stage, batch, duration_ns)

    def _reserve_transfer(
        self,
        direction: str,
        size_bytes: int,
        transfers: int,
        *,
        ready_ns: Optional[int] = None,
    ) -> int:
        """Charge an explicit host/device copy on the device copy engine.

        Copies use a resource cursor separate from compute so different logical
        nodes sharing a physical device may overlap a copy with compute, but
        copies in the same direction-independent engine serialize. The caller's
        await preserves the production critical path: D2H must reach its sync
        point before Orbit can serialize the CPU tensor, and H2D is enqueued
        before the consuming kernel.
        """
        if size_bytes <= 0 or transfers <= 0:
            return self.clock.now_ns if ready_ns is None else ready_ns
        duration_ns = self.cost.transfer_duration_ns(
            direction, size_bytes, transfers
        )
        ready = self.clock.now_ns if ready_ns is None else ready_ns
        domain_ready = self.transfer_domain.free_at_ns.get(direction, 0)
        start = max(
            self.device_model.copy_free_at_ns,
            domain_ready,
            ready,
        )
        contention_ns = start - ready
        if contention_ns > 0:
            self.stats.transfer_contention_ns += contention_ns
        finish = start + duration_ns
        self.device_model.copy_free_at_ns = finish
        self.transfer_domain.free_at_ns[direction] = finish
        self.device_model.copy_busy_ns += duration_ns
        self.device_model.copy_operations += transfers
        self.transfer_domain.busy_ns[direction] = (
            self.transfer_domain.busy_ns.get(direction, 0) + duration_ns
        )
        self.transfer_domain.operations[direction] = (
            self.transfer_domain.operations.get(direction, 0) + transfers
        )
        self.transfer_domain.bytes_total[direction] = (
            self.transfer_domain.bytes_total.get(direction, 0) + size_bytes
        )
        self.stats.transfer_ns[direction] = (
            self.stats.transfer_ns.get(direction, 0) + duration_ns
        )
        self.stats.transfer_bytes[direction] = (
            self.stats.transfer_bytes.get(direction, 0) + size_bytes
        )
        self.stats.transfer_operations[direction] = (
            self.stats.transfer_operations.get(direction, 0) + transfers
        )
        return finish

    def _completion_future(self, when_ns: int) -> "asyncio.Future[None]":
        future: asyncio.Future[None] = asyncio.Future()
        if when_ns <= self.clock.now_ns:
            future.set_result(None)
            return future

        def complete() -> None:
            if not future.done():
                future.set_result(None)

        asyncio.get_running_loop().call_at(when_ns / 1e9, complete)
        return future

    def _layers(self, decoders_binding: str) -> int:
        return self._layers_of.get(decoders_binding, 1)

    def _meta_hidden(self, rows: int, cpu: bool = False) -> DeviceTensorCarrier:
        """A hidden-state carrier with real shape and dtype, no storage.

        Exact shape matters: ``SimFabric`` pickles the payload to get its wire
        size, so a wrong shape would silently corrupt every bandwidth result.

        ``cpu`` is not cosmetic. Production honours ``op.cpu_out``: a result that
        is about to leave the node is copied to host memory, and
        ``InferNode._on_drop`` asserts ``infer_task.is_cpu`` -- an invariant that
        a task crossing the network carries host-resident data. A meta tensor on
        the "cuda" device therefore trips that assert as soon as any task is
        dropped or reordered. Honouring cpu_out keeps the invariant intact.
        """
        hidden = self._hidden_size()
        size = (hidden,) if rows == 1 else (rows, hidden)
        # A meta tensor reports `is_cpu == False`, so an explicitly host-side
        # result must be built on the real CPU device. It stays empty (no data
        # is ever written), so this costs shape-sized allocation only.
        device = torch.device("cpu") if cpu else META
        return DeviceTensorCarrier(
            torch.empty(size, device=device, dtype=torch.float16), None
        )

    def _hidden_size(self) -> int:
        for underlay in self._underlay.values():
            return int(underlay.decoder_config.hidden_size)
        return 4096

    def _hidden_bytes(self, rows: int = 1) -> int:
        return max(1, rows) * self._hidden_size() * 2

    @staticmethod
    def _carrier_bytes(carrier: DeviceTensorCarrier) -> int:
        tensor = carrier.tensor
        return int(tensor.numel() * tensor.element_size())

    @staticmethod
    def _rows(carrier: Any) -> int:
        """How many token rows a carrier holds.

        A multi-token activation is 2-D ``(rows, hidden)``, but a single decode
        step is 1-D ``(hidden,)`` -- production does `t.view(-1)` for exactly that
        case (dealers/real_basic/head.py). So `shape[0]` is the row count only
        when the tensor is 2-D; on a 1-D carrier it returns the HIDDEN SIZE.
        Reading it blindly charged prefill for 4096 tokens instead of 1 and
        tripped the cost model's extrapolation warning, which is how this was
        caught.
        """
        tensor = carrier.tensor
        return int(tensor.shape[0]) if tensor.dim() > 1 else 1

    def _next_token(self) -> int:
        self._token_counter += 1
        return self._token_counter

    # -- operate implementations ------------------------------------------
    async def _run_create_infer_binding(
        self, op: OperateCreateDecodersInferBinding
    ) -> None:
        op.on_start()
        self.memory_modeling.merge_all()
        binding = op.binding
        cache_engine = self.get_decoders_kvcache_manager(binding)
        seq_complex = SeqComplexMemoryModel(
            env=self.memory_modeling.env,
            block_manager=cache_engine,
            seq=op.seq_length_data.clone(),
            user_model=op.user_model,
        )
        if op.allocate_slots:
            remaining = op.seq_length_data.num_tokens
            while remaining > 1:
                remaining -= 1
                seq_complex.append_slot()
        binding_name = f"{op.binding_name},{self._next_unique_id()}"
        self.decoders_infer_bindings[binding_name] = (binding, seq_complex)
        await self._charge_bookkeeping("create_infer_binding", 1)
        op.set_result(binding_name)
        op.on_finish()
        self.memory_modeling.merge_all()

    async def _run_put_user_model(self, op: OperatePutUserModel) -> None:
        op.on_start()
        self.memory_modeling.put_user_model(op)
        self.memory_modeling.merge_all()
        cache_engine = self.get_decoders_kvcache_manager(op.binding)
        blocks = cache_engine.allocate_blocks(len(op.user_model_blocks[0]))
        await self._charge_bookkeeping("put_user_model", 1)
        op.set_result(
            UserModelHandle(
                blocks=blocks,
                lora_render_cfg=op.lora_render_cfg,
                _free=lambda: None,
            )
        )
        op.on_finish()

    async def _run_put_blocks(self, op: OperatePutBlocks) -> None:
        op.on_start()
        self.memory_modeling.put_blocks(op)
        self.memory_modeling.merge_all()
        await self._charge_bookkeeping("put_blocks", len(op.seq_blocks) or 1)
        op.set_result(None)
        op.on_finish()

    async def _run_get_blocks(self, op: OperateGetBlocks) -> None:
        op.on_start()
        self.memory_modeling.get_blocks(op)
        self.memory_modeling.merge_all()
        decoders_binding, _ = self.decoders_infer_bindings[op.binding]
        layers = self._layers(decoders_binding)
        await self._charge_bookkeeping("get_blocks", len(op.seq_blocks) or 1)
        # KV for the requested blocks: one (k, v) meta pair per layer per block.
        result = [
            [
                (self._meta_hidden(operator_flags.BLOCK_SIZE),
                 self._meta_hidden(operator_flags.BLOCK_SIZE))
                for _ in op.seq_blocks
            ]
            for _ in range(layers)
        ]
        op.set_result(result)
        op.on_finish()

    async def _run_head(
        self, binding: str, ops: List[OperateInferTensorToEmbeddingTokens]
    ) -> None:
        # The twin returns (vbind, bind); `bind(result)` is what registers the
        # produced carrier in `env.actuals`. Dropping it is not harmless: the
        # NEXT stage on this node calls `env.retrack(carrier)` and raises
        # KeyError, because from the twin's point of view that tensor was never
        # accounted for. Production calls it the same way (basic_operate.py).
        binds = []
        for op in ops:
            binds.append(self.memory_modeling.infer_tensor_to_embedding_tokens(op)[1])
            op.on_start()
        self.memory_modeling.merge_all()
        # Token IDs are CPU int64 inputs to the embedding kernel.
        h2d_done = self._reserve_transfer(
            "h2d", sum(len(op.tokens) * 8 for op in ops), len(ops)
        )
        compute_done = self._reserve_device(
            "infer_head", len(ops), layers=0, ready_ns=h2d_done
        )
        outgoing_rows = sum(len(op.tokens) for op in ops if op.cpu_out)
        d2h_done = self._reserve_transfer(
            "d2h",
            self._hidden_bytes(outgoing_rows) if outgoing_rows else 0,
            sum(1 for op in ops if op.cpu_out),
            ready_ns=compute_done,
        )
        completion = self._completion_future(d2h_done)
        for op, bind in zip(ops, binds):
            rows = len(op.tokens)
            op.set_result(self._meta_hidden(rows, cpu=op.cpu_out))
            if op.cpu_out:
                op.sync = completion
            bind(op.ret)
            op.on_finish()
            self.barrier_maintainer.remove_managed_op(op.context_id)
        self.memory_modeling.merge_all()

    async def _run_tail(
        self, binding: str, ops: List[OperateHiddenStateToSamplingOutput]
    ) -> None:
        for op in ops:
            self.memory_modeling.hidden_state_to_sampling_output(op)
            op.on_start()
        self.memory_modeling.merge_all()
        incoming = [op for op in ops if op.cpu_in]
        h2d_done = self._reserve_transfer(
            "h2d",
            sum(self._carrier_bytes(op.carrier) for op in incoming),
            len(incoming),
        )
        compute_done = self._reserve_device(
            "infer_tail", len(ops), layers=0, ready_ns=h2d_done
        )
        completion = self._completion_future(compute_done)
        for op in ops:
            token = self._next_token()
            # The production tail appends to the sequence before returning; the
            # shim asserts on that afterwards, so it must happen here too.
            op.carrier.free()
            op.seq_core.append_token_id(token)
            op.set_result((token, {}))
            op.sync = completion
            op.on_finish()
            self.barrier_maintainer.remove_managed_op(op.context_id)
        self.memory_modeling.merge_all()

    async def _run_prefill(
        self, binding: str, ops: List[OperateDoPrefill]
    ) -> None:
        binds = []
        for op in ops:
            binds.append(self.memory_modeling.do_prefill(op)[1])
            op.on_start()
        self.memory_modeling.merge_all()
        layers = self._layers(binding)
        # Prefill processes the whole prompt, so the batch dimension is tokens,
        # not merged requests: charging per merged request would under-count it
        # by the prompt length.
        tokens = sum(self._rows(op.carrier) for op in ops)
        incoming = [op for op in ops if op.cpu_in]
        h2d_done = self._reserve_transfer(
            "h2d",
            sum(self._carrier_bytes(op.carrier) for op in incoming),
            len(incoming),
        )
        compute_done = self._reserve_device(
            "decoder_prefill", tokens, layers=layers, ready_ns=h2d_done
        )
        outgoing_rows = sum(
            1 if op.tail_in else self._rows(op.carrier)
            for op in ops if op.cpu_out
        )
        d2h_done = self._reserve_transfer(
            "d2h",
            self._hidden_bytes(outgoing_rows) if outgoing_rows else 0,
            sum(1 for op in ops if op.cpu_out),
            ready_ns=compute_done,
        )
        completion = self._completion_future(d2h_done)
        for op, bind in zip(ops, binds):
            _, seq_complex = self.decoders_infer_bindings[op.binding]
            squeezed = 1
            seq_complex.seq.output_count += squeezed
            rows = 1 if op.tail_in else self._rows(op.carrier)
            carrier = self._meta_hidden(rows, cpu=op.cpu_out)
            op.carrier.free()
            op.set_result(carrier)
            if op.cpu_out:
                op.sync = completion
            bind(carrier)
            op.on_finish()
            self.barrier_maintainer.remove_managed_op(op.context_id)
        self.memory_modeling.merge_all()

    async def _run_generating(
        self, binding: str, ops: List[OperateDoGenerating]
    ) -> None:
        binds = []
        for op in ops:
            binds.append(self.memory_modeling.do_generating(op)[1])
            op.on_start()
        self.memory_modeling.merge_all()
        layers = self._layers(binding)
        stage = "decoder_generating"
        # A fused operate carries head and/or tail work inside the same device
        # occupancy. Charging those stages here (rather than as separate
        # operates) is exactly what consolidation does on hardware.
        fused_head = sum(1 for op in ops if op.head_task is not None)
        fused_tail = sum(1 for op in ops if op.tail_task is not None)
        incoming = [
            op for op in ops
            if op.cpu_in and op.carrier is not None
        ]
        h2d_done = self._reserve_transfer(
            "h2d",
            sum(self._carrier_bytes(op.carrier) for op in incoming)
            + fused_head * 8,
            len(incoming) + fused_head,
        )
        compute_done = self._reserve_device(
            stage, len(ops), layers=layers, ready_ns=h2d_done
        )
        if fused_head:
            compute_done = self._reserve_device(
                "infer_head", fused_head, layers=0, ready_ns=compute_done
            )
        if fused_tail:
            compute_done = self._reserve_device(
                "infer_tail", fused_tail, layers=0, ready_ns=compute_done
            )

        outgoing = [
            op for op in ops
            if op.cpu_out and op.tail_task is None
        ]
        d2h_done = self._reserve_transfer(
            "d2h", self._hidden_bytes(len(outgoing)), len(outgoing),
            ready_ns=compute_done,
        )
        cross_completion = self._completion_future(d2h_done)
        tail_completion = self._completion_future(compute_done)

        for op, bind in zip(ops, binds):
            _, seq_complex = self.decoders_infer_bindings[op.binding]
            seq_complex.seq.output_count += 1
            if op.tail_task is not None:
                token = self._next_token()
                op.tail_task.seq_core.append_token_id(token)
                if op.tail_task.finish_metric is not None:
                    op.tail_task.finish_metric()
                result = DecodeGeneratingResult(sampling_output=(token, {}))
            else:
                result = DecodeGeneratingResult(
                    carrier=self._meta_hidden(1, cpu=op.cpu_out)
                )
            if op.head_task is not None and op.head_task.finish_metric is not None:
                op.head_task.finish_metric()
            if op.carrier is not None:
                op.carrier.free()
            op.set_result(result)
            if op.tail_task is not None:
                op.sync = tail_completion
            elif op.cpu_out:
                op.sync = cross_completion
            bind(result)
            op.on_finish()
            self.barrier_maintainer.remove_managed_op(op.context_id)
        self.memory_modeling.merge_all()

    # -- teardown / reporting ---------------------------------------------
    def close(self) -> None:
        self._closed = True
        if self._wait_operate_fut is not None and not self._wait_operate_fut.done():
            fut, self._wait_operate_fut = self._wait_operate_fut, None
            fut.set_result(None)
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()

    def report(self) -> Dict[str, Any]:
        merged = {
            stage: sum(sizes) / len(sizes)
            for stage, sizes in self.stats.merged.items()
            if sizes
        }
        merged_histograms = {
            stage: {
                str(size): sizes.count(size) for size in sorted(set(sizes))
            }
            for stage, sizes in self.stats.merged.items()
            if sizes
        }
        return {
            "node": self.node_name,
            "operations": self.stats.operations,
            "operations_by_stage": dict(self.stats.by_stage),
            "mean_merged_batch": merged,
            "merged_batch_histogram": merged_histograms,
            "service_ns_by_stage": dict(self.stats.service_ns),
            "device_id": self.device_model.device_id,
            "device_busy_ns": self.device_model.busy_ns,
            "device_operations": self.device_model.operations,
            "device_contention_ns": self.stats.device_contention_ns,
            "copy_busy_ns": self.device_model.copy_busy_ns,
            "copy_operations": self.device_model.copy_operations,
            "transfer_contention_ns": self.stats.transfer_contention_ns,
            "transfer_ns_by_direction": dict(self.stats.transfer_ns),
            "transfer_bytes_by_direction": dict(self.stats.transfer_bytes),
            "transfer_operations_by_direction": dict(
                self.stats.transfer_operations
            ),
            "transfer_domain_id": self.transfer_domain.domain_id,
            "transfer_domain_busy_ns": dict(self.transfer_domain.busy_ns),
            "transfer_domain_operations": dict(
                self.transfer_domain.operations
            ),
            "transfer_domain_bytes": dict(
                self.transfer_domain.bytes_total
            ),
            "memory_deferrals": self.stats.oom_deferrals,
            "twin_total_bytes": int(self.memory_modeling.total),
            "twin_pools": {
                name: int(value)
                for name, value in self.memory_modeling.pools.items()
            },
        }
