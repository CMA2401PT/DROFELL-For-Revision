"""CUDA graph runner for decoder-only generating forward."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from operators import flags
from operators.assemble.decoders.common import LayersCacheProvider, SimpleLayersCacheProvider
from operators.assemble.decoders.real_decoders_group import DecodersGroup
from operators.container.container import Container
from operators.core.head.real import ModelHead
from operators.core.sampler.real import stable_greedy_argmax
from operators.core.tail.real import ModelTail


StageGraphKey = Tuple[str, int]
HiddenInput = Union[torch.Tensor, Sequence[torch.Tensor]]


@dataclass(frozen=True)
class GraphKey:
    """Semantic identity of one decoder or fused decode graph variant."""

    decoder_binding: str
    batch_size: int
    cache_engine_id: int
    head_binding: Optional[str] = None
    tail_binding: Optional[str] = None

    def as_tuple(self) -> Tuple[str, int, int, Optional[str], Optional[str]]:
        return (
            self.decoder_binding,
            self.batch_size,
            self.cache_engine_id,
            self.head_binding,
            self.tail_binding,
        )


@dataclass
class HeadDecodeGraphInputBuffer:
    token_ids_cpu: torch.Tensor
    token_ids: torch.Tensor

    @staticmethod
    def create(batch_size: int, device: torch.device) -> "HeadDecodeGraphInputBuffer":
        return HeadDecodeGraphInputBuffer(
            token_ids_cpu=torch.empty(
                batch_size,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            ),
            token_ids=torch.empty(
                batch_size,
                dtype=torch.int32,
                device=device,
            ),
        )

    def update(self, token_ids: List[int]) -> None:
        self.token_ids_cpu.numpy()[:] = token_ids
        self.token_ids.copy_(self.token_ids_cpu, non_blocking=True)


@dataclass
class TailDecodeGraphInputBuffer:
    hidden_states: torch.Tensor

    @staticmethod
    def create(
        batch_size: int,
        hidden_size: int,
        device: torch.device,
    ) -> "TailDecodeGraphInputBuffer":
        return TailDecodeGraphInputBuffer(
            hidden_states=torch.empty(
                (batch_size, hidden_size),
                dtype=torch.float16,
                device=device,
            )
        )

    def update(self, hidden_states: HiddenInput) -> None:
        if isinstance(hidden_states, torch.Tensor):
            self.hidden_states.copy_(hidden_states.view_as(self.hidden_states))
        else:
            torch.stack(
                [hidden.view(-1) for hidden in hidden_states],
                out=self.hidden_states,
            )


@dataclass
class CapturedStageGraph:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    module: Union[ModelHead, ModelTail]
    requested_bytes: int
    pool: Any
    output_cpu: Optional[torch.Tensor] = None


class HeadDecodeGraphRunner:
    def __init__(
        self,
        *,
        enabled: bool,
        batch_sizes: Tuple[int, ...],
        on_capture: Optional[Callable[[StageGraphKey, int], None]] = None,
        on_remove: Optional[Callable[[StageGraphKey], None]] = None,
    ) -> None:
        if not batch_sizes or any(size <= 0 for size in batch_sizes):
            raise ValueError("graph batch sizes must contain positive values")
        self.enabled = enabled
        self.batch_sizes = frozenset(batch_sizes)
        self.on_capture = on_capture
        self.on_remove = on_remove
        self._buffers: Dict[StageGraphKey, HeadDecodeGraphInputBuffer] = {}
        self._graphs: Dict[StageGraphKey, CapturedStageGraph] = {}
        self.replays = 0
        self.fallbacks: Dict[str, int] = {}

    def fallback_reason(self, batch_size: int) -> Optional[str]:
        if not self.enabled:
            return "disabled"
        if batch_size not in self.batch_sizes:
            return "batch_size"
        return None

    def record_fallback(self, reason: str) -> None:
        if self.enabled:
            self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1

    @staticmethod
    def _forward(
        model_head: ModelHead,
        input_buffer: HeadDecodeGraphInputBuffer,
    ) -> torch.Tensor:
        with torch.inference_mode():
            return model_head.embed_tokens(input_buffer.token_ids).to(
                torch.float16
            )

    def capture_binding(
        self,
        *,
        binding: str,
        model_head: ModelHead,
        stream: torch.cuda.Stream,
        device: torch.device,
    ) -> None:
        """Capture every configured head graph during model initialization."""
        if not self.enabled:
            return
        with torch.cuda.stream(stream), torch.device(device):
            for batch_size in sorted(self.batch_sizes):
                key = (binding, batch_size)
                if key in self._graphs:
                    continue
                requested_before = torch.cuda.memory_stats(device)[
                    "requested_bytes.all.current"
                ]
                input_buffer = HeadDecodeGraphInputBuffer.create(
                    batch_size, device
                )
                input_buffer.update([0] * batch_size)
                self._buffers[key] = input_buffer
                self._forward(model_head, input_buffer)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                graph_pool = torch.cuda.graph_pool_handle()
                with torch.cuda.graph(graph, pool=graph_pool, stream=stream):
                    output = self._forward(model_head, input_buffer)
                stream.synchronize()
                requested_bytes = torch.cuda.memory_stats(device)[
                    "requested_bytes.all.current"
                ] - requested_before
                self._graphs[key] = CapturedStageGraph(
                    graph=graph,
                    output=output,
                    module=model_head,
                    requested_bytes=requested_bytes,
                    pool=graph_pool,
                )
                if self.on_capture is not None:
                    self.on_capture(key, requested_bytes)

    def _run_sync(
        self,
        *,
        key: StageGraphKey,
        model_head: ModelHead,
        token_ids: List[int],
    ) -> torch.Tensor:
        input_buffer = self._buffers.get(key)
        captured = self._graphs.get(key)
        if input_buffer is None or captured is None:
            raise RuntimeError(
                f"decode head graph {key} was not captured during initialization"
            )
        input_buffer.update(token_ids)
        if captured.module is not model_head:
            raise RuntimeError("decode head graph module changed")

        captured.graph.replay()
        self.replays += 1
        # A head result may wait in the scheduler while another head batch is
        # executed. Do not expose the graph-owned output buffer to a carrier.
        return captured.output.clone()

    async def run(
        self,
        *,
        key: StageGraphKey,
        model_head: ModelHead,
        token_ids: List[int],
        container: Container,
    ) -> torch.Tensor:
        output = await container.step(
            "decode_head_graph_replay",
            self._run_sync,
            key=key,
            model_head=model_head,
            token_ids=token_ids,
        )
        return output

    def remove_binding(self, binding: str) -> None:
        keys = [key for key in self._graphs if key[0] == binding]
        for key in keys:
            self._graphs.pop(key)
            self._buffers.pop(key, None)
            if self.on_remove is not None:
                self.on_remove(key)

    def stats(self) -> Mapping[str, Any]:
        return {
            "enabled": self.enabled,
            "configured_batch_sizes": sorted(self.batch_sizes),
            "captured": [list(key) for key in sorted(self._graphs)],
            "replays": self.replays,
            "fallbacks": dict(self.fallbacks),
        }


class TailDecodeGraphRunner:
    def __init__(
        self,
        *,
        enabled: bool,
        batch_sizes: Tuple[int, ...],
        on_capture: Optional[Callable[[StageGraphKey, int], None]] = None,
        on_remove: Optional[Callable[[StageGraphKey], None]] = None,
    ) -> None:
        if not batch_sizes or any(size <= 0 for size in batch_sizes):
            raise ValueError("graph batch sizes must contain positive values")
        self.enabled = enabled
        self.batch_sizes = frozenset(batch_sizes)
        self.on_capture = on_capture
        self.on_remove = on_remove
        self._buffers: Dict[StageGraphKey, TailDecodeGraphInputBuffer] = {}
        self._graphs: Dict[StageGraphKey, CapturedStageGraph] = {}
        self.replays = 0
        self.fallbacks: Dict[str, int] = {}

    def fallback_reason(
        self,
        *,
        batch_size: int,
        use_greedy_fast_path: bool,
    ) -> Optional[str]:
        if not self.enabled:
            return "disabled"
        if batch_size not in self.batch_sizes:
            return "batch_size"
        if not use_greedy_fast_path:
            return "sampling"
        return None

    def record_fallback(self, reason: str) -> None:
        if self.enabled:
            self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1

    @staticmethod
    def _forward(
        model_tail: ModelTail,
        input_buffer: TailDecodeGraphInputBuffer,
    ) -> torch.Tensor:
        with torch.inference_mode():
            hidden_states = input_buffer.hidden_states
            output_dtype = model_tail.lm_head.weight.dtype
            if hidden_states.dtype != output_dtype:
                hidden_states = hidden_states.to(dtype=output_dtype)
            logits = model_tail.lm_head(model_tail.norm(hidden_states))
            return stable_greedy_argmax(logits)

    def capture_binding(
        self,
        *,
        binding: str,
        model_tail: ModelTail,
        stream: torch.cuda.Stream,
        device: torch.device,
    ) -> None:
        """Capture every configured tail graph during model initialization."""
        if not self.enabled:
            return
        with torch.cuda.stream(stream), torch.device(device):
            for batch_size in sorted(self.batch_sizes):
                key = (binding, batch_size)
                if key in self._graphs:
                    continue
                requested_before = torch.cuda.memory_stats(device)[
                    "requested_bytes.all.current"
                ]
                input_buffer = TailDecodeGraphInputBuffer.create(
                    batch_size, model_tail.cfg.hidden_size, device
                )
                input_buffer.hidden_states.zero_()
                self._buffers[key] = input_buffer
                self._forward(model_tail, input_buffer)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                graph_pool = torch.cuda.graph_pool_handle()
                with torch.cuda.graph(graph, pool=graph_pool, stream=stream):
                    output = self._forward(model_tail, input_buffer)
                stream.synchronize()
                requested_bytes = torch.cuda.memory_stats(device)[
                    "requested_bytes.all.current"
                ] - requested_before
                self._graphs[key] = CapturedStageGraph(
                    graph=graph,
                    output=output,
                    output_cpu=torch.empty(
                        output.shape,
                        dtype=output.dtype,
                        device="cpu",
                        pin_memory=True,
                    ),
                    module=model_tail,
                    requested_bytes=requested_bytes,
                    pool=graph_pool,
                )
                if self.on_capture is not None:
                    self.on_capture(key, requested_bytes)

    def _run_sync(
        self,
        *,
        key: StageGraphKey,
        model_tail: ModelTail,
        hidden_states: HiddenInput,
    ) -> CapturedStageGraph:
        input_buffer = self._buffers.get(key)
        captured = self._graphs.get(key)
        if input_buffer is None or captured is None:
            raise RuntimeError(
                f"decode tail graph {key} was not captured during initialization"
            )
        input_buffer.update(hidden_states)
        if captured.module is not model_tail:
            raise RuntimeError("decode tail graph module changed")

        captured.graph.replay()
        if captured.output_cpu is None:
            raise RuntimeError("decode tail graph has no CPU output buffer")
        captured.output_cpu.copy_(captured.output, non_blocking=True)
        self.replays += 1
        return captured

    async def run(
        self,
        *,
        key: StageGraphKey,
        model_tail: ModelTail,
        hidden_states: HiddenInput,
        container: Container,
    ) -> List[int]:
        captured = await container.step(
            "decode_tail_graph_replay",
            self._run_sync,
            key=key,
            model_tail=model_tail,
            hidden_states=hidden_states,
        )
        await container.sync_point()
        if captured.output_cpu is None:
            raise RuntimeError("decode tail graph has no CPU output buffer")
        return captured.output_cpu.tolist()

    def remove_binding(self, binding: str) -> None:
        keys = [key for key in self._graphs if key[0] == binding]
        for key in keys:
            self._graphs.pop(key)
            self._buffers.pop(key, None)
            if self.on_remove is not None:
                self.on_remove(key)

    def stats(self) -> Mapping[str, Any]:
        return {
            "enabled": self.enabled,
            "configured_batch_sizes": sorted(self.batch_sizes),
            "captured": [list(key) for key in sorted(self._graphs)],
            "replays": self.replays,
            "fallbacks": dict(self.fallbacks),
        }


@dataclass
class DecodeGraphInputBuffer:
    host_staging: torch.Tensor
    device_staging: torch.Tensor
    block_tables_cpu: torch.Tensor
    context_lens_cpu: torch.Tensor
    positions_cpu: torch.Tensor
    slots_cpu: torch.Tensor
    token_ids_cpu: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    token_ids: torch.Tensor
    hidden_states: torch.Tensor

    @staticmethod
    def create(
        batch_size: int,
        hidden_size: int,
        max_context_len: int,
        device: torch.device,
        position_ids_dtype: torch.dtype,
    ) -> "DecodeGraphInputBuffer":
        max_blocks_per_seq = math.ceil(max_context_len / flags.BLOCK_SIZE)
        fields = {
            "block_tables": (
                (batch_size, max_blocks_per_seq), torch.int32),
            "context_lens": ((batch_size,), torch.int32),
            "positions": ((batch_size,), position_ids_dtype),
            "slots": ((batch_size,), torch.int64),
            "token_ids": ((batch_size,), torch.int32),
        }
        layouts: Dict[str, Tuple[int, int, Tuple[int, ...], torch.dtype]] = {}
        offset = 0
        for name, (shape, dtype) in fields.items():
            element_size = dtype.itemsize
            offset = math.ceil(offset / element_size) * element_size
            num_bytes = math.prod(shape) * element_size
            layouts[name] = (offset, num_bytes, shape, dtype)
            offset += num_bytes

        host_staging = torch.empty(
            offset, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        device_staging = torch.empty(offset, dtype=torch.uint8, device=device)

        def view(staging: torch.Tensor, name: str) -> torch.Tensor:
            field_offset, num_bytes, shape, dtype = layouts[name]
            return staging[field_offset:field_offset + num_bytes].view(
                dtype
            ).view(shape)

        return DecodeGraphInputBuffer(
            host_staging=host_staging,
            device_staging=device_staging,
            block_tables_cpu=view(host_staging, "block_tables"),
            context_lens_cpu=view(host_staging, "context_lens"),
            positions_cpu=view(host_staging, "positions"),
            slots_cpu=view(host_staging, "slots"),
            token_ids_cpu=view(host_staging, "token_ids"),
            block_tables=view(device_staging, "block_tables"),
            context_lens=view(device_staging, "context_lens"),
            positions=view(device_staging, "positions"),
            slots=view(device_staging, "slots"),
            token_ids=view(device_staging, "token_ids"),
            hidden_states=torch.empty(
                (batch_size, 1, hidden_size),
                dtype=torch.float16,
                device=device,
            ),
        )

    def update(
        self,
        hidden_states: Optional[HiddenInput],
        token_ids: Optional[List[int]],
        block_tables: List[List[int]],
        context_lens: List[int],
        positions: List[int],
        slots: List[int],
    ) -> None:
        block_tables_np = self.block_tables_cpu.numpy()
        block_tables_np.fill(0)
        for row, values in zip(block_tables_np, block_tables):
            if len(values) > len(row):
                raise ValueError("decode block table exceeds graph capacity")
            row[:len(values)] = values
        self.context_lens_cpu.numpy()[:] = context_lens
        self.positions_cpu.numpy()[:] = positions
        self.slots_cpu.numpy()[:] = slots
        if token_ids is not None:
            self.token_ids_cpu.numpy()[:] = token_ids
        self.device_staging.copy_(self.host_staging, non_blocking=True)

        if hidden_states is None:
            return
        if isinstance(hidden_states, torch.Tensor):
            self.hidden_states.copy_(hidden_states.view_as(self.hidden_states))
        else:
            torch.stack(
                [hidden.view(-1) for hidden in hidden_states],
                out=self.hidden_states[:, 0],
            )


@dataclass
class CapturedDecodeGraph:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    token_ids: Optional[torch.Tensor]
    token_ids_cpu: Optional[torch.Tensor]
    model_head: Optional[ModelHead]
    model_tail: Optional[ModelTail]
    cache_holders: SimpleLayersCacheProvider
    requested_bytes: int
    pool: Any


@dataclass(frozen=True)
class DecodeGraphResult:
    hidden_states: torch.Tensor
    token_ids: Optional[List[int]] = None


class DecodeGraphRunner:
    def __init__(
        self,
        *,
        enabled: bool,
        max_context_len: int,
        batch_sizes: Tuple[int, ...],
        on_capture: Optional[Callable[[GraphKey, int], None]] = None,
        on_remove: Optional[Callable[[GraphKey], None]] = None,
    ) -> None:
        if max_context_len <= 0:
            raise ValueError("graph max context length must be positive")
        if not batch_sizes or any(size <= 0 for size in batch_sizes):
            raise ValueError("graph batch sizes must contain positive values")
        self.enabled = enabled
        self.max_context_len = max_context_len
        self.batch_sizes = frozenset(batch_sizes)
        self.on_capture = on_capture
        self.on_remove = on_remove
        self._buffers: Dict[GraphKey, DecodeGraphInputBuffer] = {}
        self._graphs: Dict[GraphKey, CapturedDecodeGraph] = {}
        self.replays = 0
        self.decoder_only_replays = 0
        self.fused_replays = 0
        self.fallbacks: Dict[str, int] = {}

    def fallback_reason(
        self,
        *,
        batch_size: int,
        context_len_max: int,
        has_lora: bool,
        decoder_backend: str,
    ) -> Optional[str]:
        if not self.enabled:
            return "disabled"
        if decoder_backend != "marlin":
            return "unsupported_decoder_backend"
        if batch_size not in self.batch_sizes:
            return "batch_size"
        if context_len_max > self.max_context_len:
            return "context_capacity"
        if has_lora:
            return "lora"
        return None

    def record_fallback(self, reason: str) -> None:
        if self.enabled:
            self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1

    def binding_captured(
        self,
        binding: str,
        cache_engine_id: int,
        head_binding: Optional[str] = None,
        tail_binding: Optional[str] = None,
    ) -> bool:
        return all(
            GraphKey(
                decoder_binding=binding,
                batch_size=batch_size,
                cache_engine_id=cache_engine_id,
                head_binding=head_binding,
                tail_binding=tail_binding,
            ) in self._graphs
            for batch_size in self.batch_sizes
        )

    def has_graph(self, key: GraphKey) -> bool:
        return key in self._graphs

    def variants_for_binding(
        self,
        binding: str,
        cache_engine_id: int,
    ) -> set[Tuple[Optional[str], Optional[str]]]:
        return {
            (key.head_binding, key.tail_binding)
            for key in self._graphs
            if key.decoder_binding == binding
            and key.cache_engine_id == cache_engine_id
        }

    @staticmethod
    def _decode_forward(
        decoders: DecodersGroup,
        cache_holders: SimpleLayersCacheProvider,
        input_buffer: DecodeGraphInputBuffer,
        context_len_max: int,
        model_head: Optional[ModelHead],
        model_tail: Optional[ModelTail],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        with torch.inference_mode():
            hidden_states = input_buffer.hidden_states
            if model_head is not None:
                hidden_states = model_head.embed_tokens(
                    input_buffer.token_ids
                ).to(torch.float16).unsqueeze(1)
            for layer_i, decoder in enumerate(decoders.decoders):
                cacher = cache_holders.get_layer_with_event_ready(layer_i)
                hidden_states = decoder.generating_forward(
                    hidden_states=hidden_states,
                    position_ids=input_buffer.positions,
                    kv_cacher=cacher,
                    context_len_max=context_len_max,
                    context_lens=input_buffer.context_lens,
                    block_tables=input_buffer.block_tables,
                    slot_mapping=input_buffer.slots,
                    lora_hook=None,
                )
            token_ids = None
            if model_tail is not None:
                tail_input = hidden_states.squeeze(1)
                output_dtype = model_tail.lm_head.weight.dtype
                if tail_input.dtype != output_dtype:
                    tail_input = tail_input.to(dtype=output_dtype)
                logits = model_tail.lm_head(model_tail.norm(tail_input))
                token_ids = stable_greedy_argmax(logits)
        return hidden_states, token_ids

    def _capture_sync(
        self,
        *,
        key: GraphKey,
        decoders: DecodersGroup,
        cache_provider: LayersCacheProvider,
        stream: torch.cuda.Stream,
        device: torch.device,
        position_ids_dtype: torch.dtype,
        block_tables: List[List[int]],
        slots: List[int],
        model_head: Optional[ModelHead],
        model_tail: Optional[ModelTail],
    ) -> int:
        if key in self._graphs:
            return 0
        requested_before = torch.cuda.memory_stats(device)[
            "requested_bytes.all.current"
        ]
        buffer = DecodeGraphInputBuffer.create(
            batch_size=key.batch_size,
            hidden_size=decoders.decoder_config.hidden_size,
            max_context_len=self.max_context_len,
            device=device,
            position_ids_dtype=position_ids_dtype,
        )
        dummy_hidden_states = torch.zeros_like(buffer.hidden_states)
        context_lens = [1] * key.batch_size
        positions = [0] * key.batch_size
        buffer.update(
            hidden_states=(
                None if model_head is not None else dummy_hidden_states
            ),
            token_ids=(
                [0] * key.batch_size if model_head is not None else None
            ),
            block_tables=block_tables,
            context_lens=context_lens,
            positions=positions,
            slots=slots,
        )

        holders = [
            cache_provider.get_layer_with_event_ready(layer_i)
            for layer_i in range(decoders.num_layers)
        ]
        static_provider = SimpleLayersCacheProvider(holders, events=None)
        self._decode_forward(
            decoders,
            static_provider,
            buffer,
            self.max_context_len,
            model_head,
            model_tail,
        )
        stream.synchronize()
        # Marlin's fused residual RMSNorm mutates the first layer's residual,
        # which aliases the fixed graph input. Restore it after warmup.
        buffer.update(
            hidden_states=(
                None if model_head is not None else dummy_hidden_states
            ),
            token_ids=(
                [0] * key.batch_size if model_head is not None else None
            ),
            block_tables=block_tables,
            context_lens=context_lens,
            positions=positions,
            slots=slots,
        )
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        graph_pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(graph, pool=graph_pool, stream=stream):
            output, token_ids = self._decode_forward(
                decoders,
                static_provider,
                buffer,
                self.max_context_len,
                model_head,
                model_tail,
            )
        stream.synchronize()
        requested_bytes = torch.cuda.memory_stats(device)[
            "requested_bytes.all.current"
        ] - requested_before
        self._graphs[key] = CapturedDecodeGraph(
            graph=graph,
            output=output,
            token_ids=token_ids,
            token_ids_cpu=(
                torch.empty(
                    token_ids.shape,
                    dtype=token_ids.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                if token_ids is not None else None
            ),
            model_head=model_head,
            model_tail=model_tail,
            cache_holders=static_provider,
            requested_bytes=requested_bytes,
            pool=graph_pool,
        )
        self._buffers[key] = buffer
        return requested_bytes

    def capture_binding(
        self,
        *,
        binding: str,
        cache_engine_id: int,
        decoders: DecodersGroup,
        cache_provider: LayersCacheProvider,
        stream: torch.cuda.Stream,
        device: torch.device,
        position_ids_dtype: torch.dtype,
        block_tables: List[List[int]],
        slots: List[int],
        head_binding: Optional[str] = None,
        tail_binding: Optional[str] = None,
        model_head: Optional[ModelHead] = None,
        model_tail: Optional[ModelTail] = None,
    ) -> None:
        """Capture every configured decoder graph during model initialization."""
        if not self.enabled:
            return
        if (head_binding is None) != (model_head is None):
            raise ValueError("head binding and module must be provided together")
        if (tail_binding is None) != (model_tail is None):
            raise ValueError("tail binding and module must be provided together")
        max_batch_size = max(self.batch_sizes)
        if len(block_tables) < max_batch_size or len(slots) < max_batch_size:
            raise ValueError("insufficient temporary KV slots for graph capture")
        with torch.cuda.stream(stream), torch.device(device):
            for batch_size in sorted(self.batch_sizes):
                key = GraphKey(
                    decoder_binding=binding,
                    batch_size=batch_size,
                    cache_engine_id=cache_engine_id,
                    head_binding=head_binding,
                    tail_binding=tail_binding,
                )
                if key in self._graphs:
                    continue
                requested_bytes = self._capture_sync(
                    key=key,
                    decoders=decoders,
                    cache_provider=cache_provider,
                    stream=stream,
                    device=device,
                    position_ids_dtype=position_ids_dtype,
                    block_tables=block_tables[:batch_size],
                    slots=slots[:batch_size],
                    model_head=model_head,
                    model_tail=model_tail,
                )
                if self.on_capture is not None:
                    self.on_capture(key, requested_bytes)

    def _run_sync(
        self,
        *,
        key: GraphKey,
        decoders: DecodersGroup,
        cache_provider: LayersCacheProvider,
        hidden_states: Optional[HiddenInput],
        token_ids: Optional[List[int]],
        block_tables: List[List[int]],
        context_lens: List[int],
        positions: List[int],
        slots: List[int],
        model_head: Optional[ModelHead],
        model_tail: Optional[ModelTail],
    ) -> CapturedDecodeGraph:
        buffer = self._buffers.get(key)
        captured = self._graphs.get(key)
        if buffer is None or captured is None:
            raise RuntimeError(
                f"decode graph {key} was not captured during initialization"
            )
        buffer.update(
            hidden_states=hidden_states,
            token_ids=token_ids,
            block_tables=block_tables,
            context_lens=context_lens,
            positions=positions,
            slots=slots,
        )

        # Swap-in readiness is dynamic and must be observed before every
        # replay. The captured graph itself uses the stable pointer tables.
        holders = [
            cache_provider.get_layer_with_event_ready(layer_i)
            for layer_i in range(decoders.num_layers)
        ]
        static_provider = SimpleLayersCacheProvider(holders, events=None)

        for old, new in zip(
            captured.cache_holders.layers_blocks_holder,
            static_provider.layers_blocks_holder,
        ):
            if old is not new:
                raise RuntimeError("decode graph cache provider changed")
        if captured.model_head is not model_head:
            raise RuntimeError("decode graph head module changed")
        if captured.model_tail is not model_tail:
            raise RuntimeError("decode graph tail module changed")

        captured.graph.replay()
        if captured.token_ids is not None:
            if captured.token_ids_cpu is None:
                raise RuntimeError("decode graph has no CPU token buffer")
            captured.token_ids_cpu.copy_(
                captured.token_ids, non_blocking=True
            )
        self.replays += 1
        if key.head_binding is None and key.tail_binding is None:
            self.decoder_only_replays += 1
        else:
            self.fused_replays += 1
        return captured

    async def run(
        self,
        *,
        key: GraphKey,
        container: Container,
        decoders: DecodersGroup,
        cache_provider: LayersCacheProvider,
        hidden_states: Optional[HiddenInput],
        token_ids: Optional[List[int]],
        block_tables: List[List[int]],
        context_lens: List[int],
        positions: List[int],
        slots: List[int],
        model_head: Optional[ModelHead] = None,
        model_tail: Optional[ModelTail] = None,
    ) -> DecodeGraphResult:
        if (key.head_binding is None) != (model_head is None):
            raise ValueError("graph key and head module disagree")
        if (key.tail_binding is None) != (model_tail is None):
            raise ValueError("graph key and tail module disagree")
        if model_head is None:
            if hidden_states is None or token_ids is not None:
                raise ValueError("decoder graph requires hidden-state input")
        elif hidden_states is not None or token_ids is None:
            raise ValueError("head-fused graph requires token input")
        captured = await container.step(
            "decode_graph_replay",
            self._run_sync,
            key=key,
            decoders=decoders,
            cache_provider=cache_provider,
            hidden_states=hidden_states,
            token_ids=token_ids,
            block_tables=block_tables,
            context_lens=context_lens,
            positions=positions,
            slots=slots,
            model_head=model_head,
            model_tail=model_tail,
        )
        output_tokens = None
        if captured.token_ids_cpu is not None:
            await container.sync_point()
            output_tokens = captured.token_ids_cpu.tolist()
        return DecodeGraphResult(
            hidden_states=captured.output,
            token_ids=output_tokens,
        )

    def remove_binding(self, binding: str) -> None:
        keys = [
            key for key in self._buffers
            if key.decoder_binding == binding
        ]
        self._remove_keys(keys)

    def remove_head_binding(self, binding: str) -> None:
        self._remove_keys([
            key for key in self._buffers if key.head_binding == binding
        ])

    def remove_tail_binding(self, binding: str) -> None:
        self._remove_keys([
            key for key in self._buffers if key.tail_binding == binding
        ])

    def remove_variant(
        self,
        binding: str,
        cache_engine_id: int,
        head_binding: Optional[str],
        tail_binding: Optional[str],
    ) -> None:
        self._remove_keys([
            key for key in self._buffers
            if key.decoder_binding == binding
            and key.cache_engine_id == cache_engine_id
            and key.head_binding == head_binding
            and key.tail_binding == tail_binding
        ])

    def _remove_keys(self, keys: List[GraphKey]) -> None:
        for key in keys:
            captured = self._graphs.pop(key, None)
            self._buffers.pop(key, None)
            if captured is not None and self.on_remove is not None:
                self.on_remove(key)

    def stats(self) -> Mapping[str, Any]:
        return {
            "enabled": self.enabled,
            "max_context_len": self.max_context_len,
            "configured_batch_sizes": sorted(self.batch_sizes),
            "captured": [
                list(key.as_tuple())
                for key in sorted(
                    self._graphs,
                    key=lambda item: (
                        item.decoder_binding,
                        item.batch_size,
                        item.cache_engine_id,
                        item.head_binding or "",
                        item.tail_binding or "",
                    ),
                )
            ],
            "replays": self.replays,
            "decoder_only_replays": self.decoder_only_replays,
            "fused_replays": self.fused_replays,
            "fallbacks": dict(self.fallbacks),
        }
