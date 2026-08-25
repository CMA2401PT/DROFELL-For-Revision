from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, List, Mapping, Optional, Tuple, TypeVar, Union, cast
import torch
from operators import flags
from operators.assemble.decoders.common import SimpleLayersCacheProvider, SimpleLayersPrefillCacheProvider
from operators.assemble.decoders.real_decoders_group import DecodersGroup
from operators.core.lora import LoraRenderConfig
from operators.container import Container
from operators.core.lora.layers_lora_provider import SimpleLayersLoraHookProvider
from operators.core.lora.real import Lora
from operators.core.lora.common import LoraConfig
from operators.core.tokenizer.common import Sequence
from operators.core.tokenizer.real import ExtendTokenizer
from operators.core.tail.real import ModelTail
from operators.core.head.real import ModelHead
from operators.core.block.real import LayerBlocksHolder
from operators.core.sampler.common import SamplingParams
from operators.core.sampler.real import (
    batch_sample,
    batch_sample_greedy_no_logprobs,
    stable_greedy_argmax,
)


@dataclass
class CacheAllocator:
    allocate_block: Callable[[], int]
    occupied_blocks: List[int] = cast(List[int], field(default_factory=list))
    occupied_slots: List[int] = cast(List[int], field(default_factory=list))

    @property
    def current_slots_count(self) -> int:
        return len(self.occupied_slots)

    def append_slot(self) -> int:
        current_slots = len(self.occupied_slots)
        num_blocks_req = (
            current_slots+1+(flags.BLOCK_SIZE-1))//flags.BLOCK_SIZE
        if num_blocks_req > len(self.occupied_blocks):
            b = self.allocate_block()
            self.occupied_blocks.append(b)
        b = self.occupied_blocks[-1]
        slot_offset = current_slots % flags.BLOCK_SIZE
        slot = b*flags.BLOCK_SIZE+slot_offset
        self.occupied_slots.append(slot)
        return slot


@dataclass
class SeqComplex:
    seq: Sequence
    allocator: CacheAllocator
    sampling_params: SamplingParams
    user_model: Union[None, List[int], Tuple[LoraRenderConfig, List[int]]]


@dataclass
class DecodeInputBuffer:
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

    @staticmethod
    def create(
        batch_size: int,
        max_blocks_per_seq: int,
        device: torch.device,
        position_ids_dtype: torch.dtype,
    ) -> "DecodeInputBuffer":
        fields = {
            "block_tables": ((batch_size, max_blocks_per_seq), torch.int32),
            "context_lens": ((batch_size,), torch.int32),
            "positions": ((batch_size,), position_ids_dtype),
            "slots": ((batch_size,), torch.int64),
            "token_ids": ((batch_size,), torch.int32),
        }
        layouts: dict[str, tuple[int, int, Tuple[int, ...], torch.dtype]] = {}
        offset = 0
        for name, (shape, dtype) in fields.items():
            element_size = torch.empty(
                (), dtype=dtype, device="cpu"
            ).element_size()
            offset = (offset + element_size - 1) // element_size * element_size
            num_bytes = math.prod(shape) * element_size
            layouts[name] = (offset, num_bytes, shape, dtype)
            offset += num_bytes

        host_staging = torch.empty(
            offset, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        device_staging = torch.empty(offset, dtype=torch.uint8, device=device)

        def view(
            staging: torch.Tensor,
            name: str,
        ) -> torch.Tensor:
            field_offset, num_bytes, shape, dtype = layouts[name]
            return staging[field_offset:field_offset + num_bytes].view(
                dtype
            ).view(shape)

        return DecodeInputBuffer(
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
        )

    def update(
        self,
        block_tables: List[List[int]],
        context_lens: List[int],
        positions: List[int],
        slots: List[int],
        token_ids: List[int],
    ) -> None:
        block_tables_np = self.block_tables_cpu.numpy()
        block_tables_np.fill(0)
        for row, values in zip(block_tables_np, block_tables):
            row[:len(values)] = values
        self.context_lens_cpu.numpy()[:] = context_lens
        self.positions_cpu.numpy()[:] = positions
        self.slots_cpu.numpy()[:] = slots
        self.token_ids_cpu.numpy()[:] = token_ids

        self.device_staging.copy_(self.host_staging, non_blocking=True)


@dataclass
class CapturedDecodeGraph:
    graph: torch.cuda.CUDAGraph
    token_ids: torch.Tensor
    token_ids_cpu: torch.Tensor
    pool: Any


E = TypeVar('E')


def pad(mutli_lists: List[List[E]], default: E) -> List[List[E]]:
    pad_len = max([len(l) for l in mutli_lists])
    ls = [l+[default]*(pad_len-len(l))
          for l in mutli_lists]
    return ls


class InferRunner:
    def __init__(self, head: ModelHead, tail: ModelTail,
                 decoders: DecodersGroup, layer_cachers: List[LayerBlocksHolder],
                 tokenizer: ExtendTokenizer,
                 enable_cuda_graph: bool,
                 max_seqs_num: int,
                 max_seq_len: int,
                 ) -> None:
        if max_seqs_num <= 0:
            raise ValueError("max_seqs_num must be positive")
        self.layer_cachers = SimpleLayersCacheProvider(
            layers_blocks_holder=layer_cachers, events=None)
        self.head = head
        self.decoders = decoders
        self.tail = tail
        self.tokenizer = tokenizer
        # The checkpoint is FP16. ModelHead/ModelTail are constructed using
        # PyTorch's FP32 default, so restore the intended inference dtype here.
        self.head.half()
        self.tail.half()
        self.position_ids_dtype = (
            self.decoders.decoders[0].rotary_emb.position_ids_dtype
        )
        self.tail.norm.use_fused_cuda_ops = (
            self.decoders.decoders[0].input_layernorm.use_fused_cuda_ops
        )
        self._decode_input_buffers: dict[int, DecodeInputBuffer] = {}
        self.enable_cuda_graph = enable_cuda_graph
        self.max_seqs_num = max_seqs_num
        self.max_seq_len = max_seq_len
        self.graph_batch_sizes = frozenset(
            range(1, self.max_seqs_num + 1)
        )
        self._decode_graphs: dict[int, CapturedDecodeGraph] = {}
        self.decode_graph_replays = 0
        self.decode_graph_fallbacks: dict[str, int] = {}
        if self.enable_cuda_graph:
            if not self.decoders.decoders[0].input_layernorm.use_fused_cuda_ops:
                raise ValueError("CUDA graph requires a fused decoder backend")
            self._capture_decode_graphs()

    def _record_graph_fallback(self, reason: str) -> None:
        self.decode_graph_fallbacks[reason] = (
            self.decode_graph_fallbacks.get(reason, 0) + 1
        )

    @staticmethod
    def _can_use_greedy_no_logprobs(params: SamplingParams) -> bool:
        return (
            params.temperature < 1e-5
            and params.logprobs is None
            and params.presence_penalty == 0.0
            and params.frequency_penalty == 0.0
            and params.repetition_penalty == 1.0
        )

    async def _sample(
        self,
        container: Container,
        logits: torch.Tensor,
        seqs: List[SeqComplex],
    ) -> List[int]:
        sampling_params = [seq.sampling_params for seq in seqs]
        if all(self._can_use_greedy_no_logprobs(p) for p in sampling_params):
            return await batch_sample_greedy_no_logprobs(container, logits)
        tokens, _log_probs = await batch_sample(
            container=container,
            seqs_logits=logits,
            sampling_params=sampling_params,
            output_token_ids=[seq.seq.core.output_token_ids for seq in seqs],
            prompt_token_ids=[seq.seq.core.prompt_token_ids for seq in seqs],
        )
        return tokens

    def _graph_fallback_reason(
        self,
        seqs: List[SeqComplex],
        context_len_max: int,
    ) -> Optional[str]:
        if not self.enable_cuda_graph:
            return "disabled"
        if len(seqs) not in self.graph_batch_sizes:
            return "batch_size"
        if not self.decoders.decoders[0].input_layernorm.use_fused_cuda_ops:
            return "unsupported_decoder_backend"
        if context_len_max > self.max_seq_len:
            return "context_capacity"
        if any(isinstance(seq.user_model, tuple) for seq in seqs):
            return "lora"
        if not all(
            self._can_use_greedy_no_logprobs(seq.sampling_params)
            for seq in seqs
        ):
            return "sampling"
        return None

    def _decode_device_greedy(
        self,
        input_buffer: DecodeInputBuffer,
        context_len_max: int,
    ) -> torch.Tensor:
        hidden_states = self.head.embed_tokens(input_buffer.token_ids).to(
            torch.float16
        ).unsqueeze(1)
        if self.decoders.fuse_cache is not None:
            self.decoders.fuse_cache.make_fuse_forward_temp(
                batch_size=input_buffer.token_ids.shape[0],
                seq_len=1,
            )
        try:
            for layer_i, decoder in enumerate(self.decoders.decoders):
                hidden_states = decoder.generating_forward(
                    hidden_states=hidden_states,
                    position_ids=input_buffer.positions,
                    kv_cacher=self.layer_cachers.get_layer_with_event_ready(
                        layer_i
                    ),
                    context_len_max=context_len_max,
                    context_lens=input_buffer.context_lens,
                    block_tables=input_buffer.block_tables,
                    slot_mapping=input_buffer.slots,
                    lora_hook=None,
                )
        finally:
            if self.decoders.fuse_cache is not None:
                self.decoders.fuse_cache.remove_fuse_forward_temp()
        hidden_states = hidden_states.squeeze(1)
        output_dtype = self.tail.lm_head.weight.dtype
        if hidden_states.dtype != output_dtype:
            hidden_states = hidden_states.to(dtype=output_dtype)
        logits = self.tail.lm_head(self.tail.norm(hidden_states))
        return stable_greedy_argmax(logits)

    def _capture_decode_graph(
        self,
        input_buffer: DecodeInputBuffer,
        stream: torch.cuda.Stream,
    ) -> CapturedDecodeGraph:
        # Warm all kernels and materialize allocator state on the same stream
        # before capture. Rewriting the same KV slots is idempotent.
        self._decode_device_greedy(
            input_buffer,
            self.max_seq_len,
        )
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        graph_pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(
            graph,
            pool=graph_pool,
            stream=stream,
        ):
            token_ids = self._decode_device_greedy(
                input_buffer,
                self.max_seq_len,
            )
        stream.synchronize()
        captured = CapturedDecodeGraph(
            graph=graph,
            token_ids=token_ids,
            token_ids_cpu=torch.empty(
                token_ids.shape,
                dtype=token_ids.dtype,
                device="cpu",
                pin_memory=True,
            ),
            pool=graph_pool,
        )
        self._decode_graphs[input_buffer.token_ids.shape[0]] = captured
        return captured

    def _capture_decode_graphs(self) -> None:
        device = self.decoders.device
        max_blocks_per_seq = len(
            self.layer_cachers.layers_blocks_holder[0].layer_key_ptr_table_list
        )
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream), torch.inference_mode():
            for batch_size in sorted(self.graph_batch_sizes):
                input_buffer = DecodeInputBuffer.create(
                    batch_size=batch_size,
                    max_blocks_per_seq=max_blocks_per_seq,
                    device=device,
                    position_ids_dtype=self.position_ids_dtype,
                )
                input_buffer.update(
                    block_tables=[[] for _ in range(batch_size)],
                    context_lens=[1] * batch_size,
                    positions=[0] * batch_size,
                    slots=[0] * batch_size,
                    token_ids=[0] * batch_size,
                )
                self._decode_input_buffers[batch_size] = input_buffer
                self._capture_decode_graph(input_buffer, stream)

    async def _decode_graph(
        self,
        container: Container,
        input_buffer: DecodeInputBuffer,
    ) -> List[int]:
        batch_size = input_buffer.token_ids.shape[0]
        captured = self._decode_graphs.get(batch_size)
        if captured is None:
            raise RuntimeError(
                f"decode graph for batch size {batch_size} was not captured "
                "during InferRunner initialization"
            )
        # CUDA capture records launches but its output buffer is not a valid
        # decode result on all supported PyTorch/CUDA combinations. Replay
        # once even on the capture turn before exposing token_ids.
        with torch.cuda.stream(container.stream):
            captured.graph.replay()
        self.decode_graph_replays += 1
        with torch.cuda.stream(container.stream):
            captured.token_ids_cpu.copy_(
                captured.token_ids,
                non_blocking=True,
            )
        await container.sync_point()
        return captured.token_ids_cpu.tolist()

    def graph_stats(self) -> Mapping[str, Any]:
        return {
            "enabled": self.enable_cuda_graph,
            "max_context_len": self.max_seq_len,
            "configured_batch_sizes": sorted(self.graph_batch_sizes),
            "captured_batch_sizes": sorted(self._decode_graphs),
            "replays": self.decode_graph_replays,
            "fallbacks": dict(self.decode_graph_fallbacks),
        }

    async def prefilling_step(self, container: Container, seqs: List[SeqComplex]):
        prompts = [seq.seq.core.prompt_token_ids for seq in seqs]
        prompt_lens = [len(p) for p in prompts]  # 8

        input_lens = prompt_lens
        slot_mapping: List[int] = []
        last_indices: List[int] = []
        offset = 0
        for prompt_len, s in zip(prompt_lens, seqs):
            for _ in range(prompt_len):
                slot_mapping.append(s.allocator.append_slot())
            last_indices.append(offset + prompt_len - 1)
            offset += prompt_len
        prompts = [token for tokens in prompts for token in tokens]
        input_positions = [
            i for prompt_len in prompt_lens for i in range(prompt_len)]

        def gen_input(prompts: List[int], slot_mapping: List[int], input_positions: List[int], input_lens: List[int], user_models: List[Union[None, List[int], Tuple[LoraRenderConfig, List[int]]]]):
            hidden_states = torch.tensor(prompts, dtype=torch.int32)
            slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int64)
            input_positions_tensor = torch.tensor(
                input_positions, dtype=self.position_ids_dtype
            )
            ptunings = [(p if isinstance(p, list) else None)
                        for p in user_models]
            loras = [(l if isinstance(l, tuple) else None)
                     for l in user_models]
            layers_cache_provider = SimpleLayersPrefillCacheProvider(
                seq_ptunings=ptunings, cache_provider=self.layer_cachers
            )
            layers_lora_hook_provider = SimpleLayersLoraHookProvider(
                seq_loras=loras,
                seq_lens=input_lens,
                cache_provider=self.layer_cachers
            )

            return hidden_states, slot_mapping_tensor, input_positions_tensor, layers_cache_provider, layers_lora_hook_provider

        hidden_states, slot_mapping_tensor, input_positions_tensor, layers_cache_provider, layers_lora_hook_provider = \
            await container.step("gen_input", gen_input,
                                 prompts=prompts, slot_mapping=slot_mapping, input_positions=input_positions, input_lens=input_lens, user_models=[s.user_model for s in seqs])
        hidden_states = await self.head.infer_to_embedding_tokens(container=container, input_tokens_tensor=hidden_states, dtype=torch.float16)
        hidden_states = await self.decoders.multi_seqs_prefill_forward(
            container=container,
            hidden_states=hidden_states, position_ids=input_positions_tensor, input_lens=input_lens,
            layers_cache_provider=layers_cache_provider,
            slot_mapping=slot_mapping_tensor,
            layers_lora_hook_provider=layers_lora_hook_provider
        )

        def take_last(hidden_states: torch.Tensor, last_indices: List[int]):
            last_indices_tensor: torch.Tensor = torch.tensor(last_indices)
            hidden_states = hidden_states[last_indices_tensor]
            return hidden_states
        hidden_states = await container.step("take_last", take_last, hidden_states=hidden_states, last_indices=last_indices)
        logits = await self.tail.to_logits(container=container, hidden_states=hidden_states)

        return await self._sample(container, logits, seqs)

    async def generating_step(self, container: Container, seqs: List[SeqComplex], next_tokens: List[int]):
        prompt_lens = [s.seq.core.length_data.prompt_len for s in seqs]
        generating_lens = [s.seq.core.length_data.output_count for s in seqs]
        # latest generated token (this input token) has not already been in block cache
        cached_lens = [p+g-1 for (p, g) in zip(prompt_lens, generating_lens)]
        positions = cached_lens  # position for rope of this input token

        user_models = [sc.user_model for sc in seqs]
        ptunings = [(p if isinstance(p, list) else None) for p in user_models]
        loras = [(l if isinstance(l, tuple) else None) for l in user_models]

        context_lens = [(len(ptuning or [])*flags.BLOCK_SIZE)+position+1 for position, ptuning in zip(
            positions, ptunings)]  # when attention, this input token is already been cached
        slots = [s.allocator.append_slot() for s in seqs]
        block_tables = pad(
            [(ptuning or []) + s.allocator.occupied_blocks for s, ptuning in zip(seqs, ptunings)], 0)
        context_lens_max = max(context_lens)

        def gen_input(block_tables: List[List[int]], context_lens: List[int], positions: List[int], slots: List[int],
                      next_tokens: List[int], loras: List[Optional[Tuple[LoraRenderConfig, List[int]]]]):
            batch_size = len(next_tokens)
            input_buffer = self._decode_input_buffers.get(batch_size)
            if input_buffer is None:
                max_blocks_per_seq = len(
                    self.layer_cachers.layers_blocks_holder[0].layer_key_ptr_table_list
                )
                input_buffer = DecodeInputBuffer.create(
                    batch_size=batch_size,
                    max_blocks_per_seq=max_blocks_per_seq,
                    device=container.device,
                    position_ids_dtype=self.position_ids_dtype,
                )
                self._decode_input_buffers[batch_size] = input_buffer
            input_buffer.update(
                block_tables=block_tables,
                context_lens=context_lens,
                positions=positions,
                slots=slots,
                token_ids=next_tokens,
            )
            layers_lora_provider = SimpleLayersLoraHookProvider(
                seq_loras=loras, seq_lens=[1]*len(next_tokens), cache_provider=self.layer_cachers)
            return (
                input_buffer.block_tables,
                input_buffer.context_lens,
                input_buffer.positions,
                input_buffer.slots,
                input_buffer.token_ids,
                layers_lora_provider,
            )

        block_tables_t, context_lens_t, positions_t, slot_mapping_t, input_tokens_tensor, layers_lora_hook_provider = \
            await container.step("gen_input", gen_input,
                                 block_tables=block_tables, context_lens=context_lens, positions=positions, slots=slots, next_tokens=next_tokens, loras=loras)

        input_buffer = self._decode_input_buffers[len(next_tokens)]
        graph_fallback = self._graph_fallback_reason(seqs, context_lens_max)
        if graph_fallback is None:
            return await self._decode_graph(container, input_buffer)
        if self.enable_cuda_graph:
            self._record_graph_fallback(graph_fallback)

        hidden_states = (await self.head.infer_to_embedding_tokens(container=container, input_tokens_tensor=input_tokens_tensor, dtype=torch.float16)).unsqueeze(1)
        hidden_states = await self.decoders.multi_seqs_generating_forward(
            container=container,
            hidden_states=hidden_states,
            position_ids=positions_t,
            layers_cache_provider=self.layer_cachers,
            context_len_max=context_lens_max,  # 9
            context_lens=context_lens_t,  # [9],device, int32
            block_tables=block_tables_t,  # [[]], device, int32
            slot_mapping=slot_mapping_t,  # [], device int64
            layers_lora_hook_provider=layers_lora_hook_provider,
        )

        logits = await self.tail.to_logits(container=container, hidden_states=hidden_states.squeeze(1))

        return await self._sample(container, logits, seqs)

    async def run_seqs(self, container: Container, seqs: List[SeqComplex], print_output: bool):
        with container.stage("prefilling_step"):
            next_tokens = await self.prefilling_step(container=container, seqs=seqs)
            await container.notify("prefilling_step_end")
        for seq_i, s in enumerate(seqs):
            new_token = next_tokens[seq_i]
            s.seq.core.append_token_id(new_token)
            self.tokenizer.incrementally_decode_seq(s.seq)

        if print_output:
            for seq_i, s in enumerate(seqs):
                assert s.seq.prompt is not None
                print((s.seq.prompt+s.seq.output_text).replace("\n", "\\n"))
            print("", end=f"\033[{len(seqs)}F")

        total_generated = 0
        start_time = time.time()
        all_seqs = seqs
        seqs = [s for s in seqs]
        # generating
        while len(seqs):
            with container.stage("generating_step"):
                generated_tokens = await self.generating_step(container=container, seqs=seqs, next_tokens=next_tokens)
                await container.notify("generating_step_end")
            if print_output:
                print("\x1b[2J\x1b[H")
            for seq_i, s in enumerate(seqs):
                new_token = generated_tokens[seq_i]
                s.seq.core.append_token_id(new_token)
                self.tokenizer.incrementally_decode_seq(s.seq)
                total_generated += 1
            seqs, next_tokens = [], []
            for seq_i, s in enumerate(all_seqs):
                if not s.seq.core.stopped:
                    seqs.append(s)
                    next_tokens.append(s.seq.core.output_token_ids[-1])
                if print_output:
                    assert s.seq.prompt is not None
                    print(f"SEQ-{seq_i} > ", (s.seq.prompt+s.seq.output_text))
            if not print_output:
                print(
                    f"total gen: {total_generated/1000:.2f}K {total_generated/(time.time()-start_time):.2f}TPS", end="\r")
        print()


if __name__ == "__main__":
    from operators.assemble.decoders.real import make_decoders_from_files
    from operators.core.ptuning.common import PtuningConfig
    from operators.core.ptuning.real import Ptuning
    from operators.helpers.model_lib import ModelsLib
    from operators.container import NonAsyncContainer
    import split_qllm_models
    import data
    import extend_asyncio as asyncio

    # RUNNER CONFIG STAGE
    target_model = "teared-Llama-2-7B-GPTQ"
    decoders_impl_name = "marlin"
    device = torch.zeros((0,), device="cuda:0").device
    num_blocks = 128
    max_seq_len = 32*1024
    max_seq_num = 8
    models = ModelsLib(split_qllm_models.lib_path)
    model = models.models_storage[target_model]
    tokenizer = ExtendTokenizer.load_tokenier(
        model.tokenizer_path,
        model.config.tokenizer_special_tokens)
    block_info = model.config.block_info
    free_blocks = [i for i in range(num_blocks)]

    with torch.device(device):
        head = ModelHead(model.config.model_head_config)
        head.load_from_data_dir(model.model_head_path)
        tail = ModelTail(model.config.model_tail_config)
        tail.load_from_data_dir(model.model_tail_path)
        decoders = make_decoders_from_files(
            implement=decoders_impl_name,
            config=model.config.quant_decoder_config,
            state_files=[model.decoder_path(
                i) for i in range(model.config.num_decoders)],
        )
        layer_cachers = [
            LayerBlocksHolder.create_contigious_cacher(
                block_info=block_info,
                num_blocks=num_blocks,
                device=device,
                dtype=torch.float16
            )
            for _ in range(model.config.num_decoders)
        ]
    runner = InferRunner(
        head,
        tail,
        decoders,
        layer_cachers,
        tokenizer,
        enable_cuda_graph=True,
        max_seqs_num=max_seq_num,
        max_seq_len=max_seq_len,
    )

    # RUNNER CONFIG STAGE FINISH

    # SIMPLE EXAMPLE INFER
    ptuning_cfg = PtuningConfig(
        num_layers=model.config.num_decoders, head_dim=model.config.head_dim, num_key_value_heads=model.config.num_key_value_heads,
        prefix_len=128, dropout_p=0.1, build_prompt_str="[instruction]{}\n[generation]"
    )

    lora_cfg = LoraConfig(
        r_q=-1,
        r_k=0,
        r_v=-1,
        build_prompt_str="[instruction]{}\n[generation]",
        dropout_p=0.0,
        len_limit=128,
    ).check_or_set_decoders_config(
        num_layers=model.config.num_decoders, head_dim=model.config.head_dim,
        num_key_value_heads=model.config.num_key_value_heads, num_query_heads=model.config.num_query_heads
    )
    use_ptuning = True
    user_model_cfg = ptuning_cfg if use_ptuning else lora_cfg
    user_model_data = \
        data.trained_model_path_of("ptuning_7B.pt") \
        if use_ptuning else \
        data.trained_model_path_of("lora_7B.pt")

    def make_user_model(
        cfg: Union[LoraConfig, PtuningConfig],
        state_dict: Mapping[str, Any],
        get_block: Callable[[], int],
        layer_cachers: List[LayerBlocksHolder],
    ):
        is_lora = isinstance(cfg, LoraConfig)
        user_model = Lora(cfg) if is_lora else Ptuning(cfg)
        user_model.load_state_dict(state_dict)
        user_data =\
            cast(Lora, user_model).get_inference_lora_blocks(torch.float16) \
            if is_lora else \
            cast(Ptuning, user_model).get_inference_ptuning_kv_cache(torch.float16)
        required_blocks_num = 0
        if is_lora:
            required_blocks_num = (cfg.index.total_saperate_blocks+1)//2
        else:
            assert cfg.prefix_len % flags.BLOCK_SIZE == 0, f"{cfg.prefix_len}, {flags.BLOCK_SIZE}"
            required_blocks_num = cfg.prefix_len//flags.BLOCK_SIZE
        cache_blocks = [get_block() for _ in range(required_blocks_num)]
        if is_lora:
            for lora_block, cacher in zip(user_data, layer_cachers):
                assert isinstance(lora_block, list)
                cacher.swap_in_from_mapping({
                    idx: (lora_block[i][0], lora_block[i][1])
                    for i, idx in enumerate(cache_blocks)
                })
            return (cfg.render_config, cache_blocks)
        else:
            for (k, v), cacher in zip(user_data, layer_cachers):
                assert not isinstance(k, tuple)
                assert not isinstance(v, tuple)
                bk, bv = cacher.block_info.tensor_kv_to_block_kv(k, v)
                cacher.swap_in_from_mapping({
                    idx: (bk[i], bv[i])
                    for i, idx in enumerate(cache_blocks)
                })
            return cache_blocks

    with torch.device(device):
        user_model = make_user_model(
            cfg=user_model_cfg,
            state_dict=torch.load(user_model_data, map_location=device),
            get_block=free_blocks.pop, layer_cachers=layer_cachers
        )

    def make_sample_seq_complex():
        seq_complexs: List[SeqComplex] = []
        available_blocks = free_blocks.copy()

        seq0 = tokenizer.init_seq(
            prompt=f"once upon a time", max_tokens=128, ignore_eos=False)
        allocator0 = CacheAllocator(available_blocks.pop)
        sampling_params0 = SamplingParams(
            temperature=0.0, logprobs=1, presence_penalty=1, frequency_penalty=1, repetition_penalty=2)
        seq_complexs.append(SeqComplex(
            seq0, allocator0, sampling_params0, user_model=None))

        built_prompt = user_model_cfg.build_prompt_str.format(
            "类型#裤*风格#英伦*风格#简约")
        seq1 = tokenizer.init_seq(
            prompt=built_prompt, max_tokens=128, ignore_eos=False)
        allocator1 = CacheAllocator(available_blocks.pop)
        sampling_params1 = SamplingParams(temperature=0.0, logprobs=1)
        seq_complexs.append(SeqComplex(
            seq1, allocator1, sampling_params1, user_model=user_model))

        return seq_complexs

    def make_bench_complex(seq_count: int, per_seq_tokens: int):
        seq_complexs: List[SeqComplex] = []
        available_blocks = free_blocks.copy()
        for _ in range(seq_count):
            seq = tokenizer.init_seq(
                prompt=f"", max_tokens=per_seq_tokens, ignore_eos=True, stop_str=None)
            allocator = CacheAllocator(available_blocks.pop)
            sampling_params = SamplingParams(temperature=0.0, logprobs=1)
            seq_complexs.append(SeqComplex(
                seq, allocator, sampling_params, user_model=user_model))
        return seq_complexs

    stream = torch.cuda.Stream()

    async def amain():
        loop = asyncio.get_event_loop()

        container = NonAsyncContainer(stream)
        # from operators.metrics.base_container.containers.fast import FastContainer
        # container = FastContainer(stream)
        # loop.set_blocking_scheduler(container)

        sample_seq = make_sample_seq_complex()

        with container.mode('inference'):
            await runner.run_seqs(container, sample_seq,  print_output=True)

        torch.cuda.synchronize()
        loop.clear_profile()

        bench_seqs = make_bench_complex(1, 128)
        with container.mode('inference'):
            await runner.run_seqs(container, bench_seqs,  print_output=False)

        # will be 0 if NonAsyncContainer is config, since there is no yield happen
        print(f"blocking time: {loop.blocking_calltime_ms:.4f}ms")
        print(f"nonblocking time: {loop.non_blocking_calltime_ms:.4f}ms")

    asyncio.config_impl("golang", True)
    asyncio.run(amain())
