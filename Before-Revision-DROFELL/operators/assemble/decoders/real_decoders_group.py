import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple
import torch
from torch import nn
from operators import flags
from ...core.decoder_wrap.common import LoraHook

from ...core.block.real import LayerBlocksHolder
from ...core.block.common import BlockInfo

from ...container.container import Container, NonAsyncContainer
from ...core.decoder_wrap.real_cache import FuseCache
from ..decoder.real import Decoder
from .common import LayersCacheProvider, LayersLoraHookProvider, LayersPrefillCacheProvider, SimpleLayersCacheProvider, SimpleLayersPrefillCacheProvider


class DecodersGroup:
    def __init__(self,
                 decoders: List[Decoder],
                 device: torch.device,
                 fuse_cache: Optional[FuseCache],
                 attached_no_gc: Any = None
                 ) -> None:
        self.decoders: List[Decoder] = decoders
        self.device = device
        self.fuse_cache = fuse_cache
        self.attached_no_gc = attached_no_gc
        self.non_free_addrs: Set[int] = set()
        self._block_info = BlockInfo(
            head_dim=decoders[0].config.head_dim,
            num_key_value_heads=decoders[0].config.num_key_value_heads,
            dtype=torch.float16
        )
        self.decoder_config = decoders[0].config

    @property
    def block_info(self):
        return self._block_info

    @property
    def num_layers(self):
        return len(self.decoders)

    def freeze(self):
        def freeze(module: nn.Module):
            for _, parameter in module.named_parameters():
                parameter.requires_grad_(False)
                self.non_free_addrs.add(parameter.data_ptr())
            for _, buffer in module.named_buffers():
                buffer.requires_grad_(False)
                self.non_free_addrs.add(buffer.data_ptr())
            for _, submodule in module.named_children():
                freeze(submodule)
        for decoder_i, decoder in enumerate(self.decoders):
            decoder = decoder.eval()
            freeze(decoder)
            self.decoders[decoder_i] = decoder

    # def multi_seqs_finetuning_forward(
    #     self,
    #     hidden_states: torch.Tensor, position_ids: torch.Tensor, input_lens: list[int],
    #     ptuning_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    #     grad: bool
    # ):
    #     with torch.inference_mode(not grad):
    #         assert len(ptuning_kv_caches) == len(self.decoders)
    #         for (decoder, ptuning_kv_cache) in zip(self.decoders, ptuning_kv_caches):
    #             hidden_states = decoder.multi_seqs_finetuning_forward(hidden_states=hidden_states, position_ids=position_ids, input_lens=input_lens, ptuning_kv_cache=ptuning_kv_cache)
    #         return hidden_states

    async def multi_seqs_prefill_forward(
        self, container: Container,
        hidden_states: torch.Tensor, position_ids: torch.Tensor,
        input_lens: list[int],
        layers_cache_provider: LayersPrefillCacheProvider,
        slot_mapping: torch.Tensor,
        layers_lora_hook_provider: Optional[LayersLoraHookProvider] = None,
    ):
        def decoder_multi_seqs_prefill_forward(
            decoder: Decoder,
            layer_i: int, layers_cache_provider: LayersPrefillCacheProvider,
            hidden_states: torch.Tensor, position_ids: torch.Tensor,
            input_lens: list[int],
        ):
            cacher, ptuning_cache_getter = \
                layers_cache_provider.get_layer_with_event_ready(
                    layer_i=layer_i,
                )
            hook_fn: Optional[LoraHook] = None
            if layers_lora_hook_provider is not None:
                hook_fn = layers_lora_hook_provider.get_lora_hook(layer_i)
            return decoder.multi_seqs_prefill_forward(
                hidden_states=hidden_states, position_ids=position_ids, input_lens=input_lens,
                ptunings_getter=ptuning_cache_getter,
                kv_cacher=lambda k, v: cacher.kv_tensor_to_slots_with_convert(k, v, slot_mapping),
                lora_hook=hook_fn,
            )
        for (layer_i, decoder) in enumerate(self.decoders):
            hidden_states = await container.step(
                "decoder_multi_seqs_prefill_forward", decoder_multi_seqs_prefill_forward,
                decoder=decoder, layer_i=layer_i, layers_cache_provider=layers_cache_provider,
                hidden_states=hidden_states, position_ids=position_ids, input_lens=input_lens
            )
        return hidden_states

    async def multi_seqs_generating_forward(
        self,
        container: Container,
        hidden_states: torch.Tensor, position_ids: torch.Tensor,
        layers_cache_provider: LayersCacheProvider,
        context_len_max: int,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        slot_mapping: torch.Tensor,
        layers_lora_hook_provider: Optional[LayersLoraHookProvider] = None,
    ):
        """
        hidden_states=[bsz,1,hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        assert seq_len == 1

        if self.fuse_cache is not None:
            self.fuse_cache.make_fuse_forward_temp(batch_size=batch_size, seq_len=seq_len)

        def decoder_generating_forward(
            decoder: Decoder,
            layer_i: int, layers_cache_provider: LayersCacheProvider,
            hidden_states: torch.Tensor, position_ids: torch.Tensor,
            context_len_max: int,
            context_lens: torch.Tensor,
            block_tables: torch.Tensor,
            slot_mapping: torch.Tensor,
        ):
            cacher = layers_cache_provider.get_layer_with_event_ready(
                layer_i=layer_i,
            )
            hook_fn: Optional[LoraHook] = None
            if layers_lora_hook_provider is not None:
                hook_fn = layers_lora_hook_provider.get_lora_hook(layer_i)
            return decoder.generating_forward(
                hidden_states=hidden_states, position_ids=position_ids,
                kv_cacher=cacher,
                context_len_max=context_len_max, context_lens=context_lens,
                block_tables=block_tables, slot_mapping=slot_mapping,
                lora_hook=hook_fn,
            )

        for (layer_i, decoder) in enumerate(self.decoders):
            hidden_states = await container.step(
                "decoder_generating_forward", decoder_generating_forward,
                decoder=decoder, layer_i=layer_i, layers_cache_provider=layers_cache_provider,
                hidden_states=hidden_states, position_ids=position_ids,
                context_len_max=context_len_max, context_lens=context_lens,
                block_tables=block_tables, slot_mapping=slot_mapping
            )
        if self.fuse_cache is not None:
            self.fuse_cache.remove_fuse_forward_temp()
        return hidden_states

    async def get_forward_backward_throughoutput(
        self,
        per_batch_size: Tuple[int, int] = (1, 1024),
        is_bfloat16: bool = False,
        batch_count: int = 10,
        time_limit: float = -1,
        user_model: Optional[Any] = None,
        container: Optional[Container] = None,
    ) -> Tuple[float, float, float, float]:
        from ..fwd_ctx.trainable import PlaceHolderTrainableUserModel
        from ..fwd_ctx import forward
        from ..fwd_ctx import backward
        if container is None:
            container = NonAsyncContainer(torch.cuda.current_stream())
        dtype: torch.dtype = torch.bfloat16 if is_bfloat16 else torch.float16
        qheads = self.decoders[0].config.num_query_heads
        hidden_states = torch.zeros(
            (
                per_batch_size[0]*per_batch_size[1],
                self.block_info.head_dim*qheads
            ),
            device=self.device,
            dtype=dtype,
            requires_grad=False
        )

        position_ids = torch.tensor([
            i
            for _ in range(per_batch_size[0])
            for i in range(per_batch_size[1])
        ],
            device=self.device,
            dtype=torch.int32,
        )

        input_lens = [per_batch_size[1]]*per_batch_size[0]
        if user_model is None:
            placeholder = torch.zeros((
                2,
                0,
                self.block_info.num_key_value_heads,
                self.block_info.head_dim,
            ), dtype=dtype, device=self.device, requires_grad=False)
            user_model = PlaceHolderTrainableUserModel([
                placeholder for _ in range(self.num_layers)
            ])

        async def forward_once():
            return await forward(
                container=container,
                hidden_states=hidden_states.detach(),
                position_ids=position_ids,
                input_lens=input_lens,
                user_model=user_model,
                decoders=self,
            )

        async def forward_backward_once():
            t, ctx = await forward_once()
            return await backward(
                container=container,
                grad=torch.empty_like(t),
                ctx=ctx,
                user_model=user_model,
                decoders=self,
            )

        # warm up
        await forward_backward_once()
        await container.sync_point()

        async def test(do: Callable[[], Coroutine[Any, Any, Any]]):
            st = time.time_ns()
            ddl = time.time_ns()+int(time_limit*1e9)
            total_count = 0
            for _ in range(batch_count):
                await do()
                total_count += 1
                if ddl > st and time.time_ns() > ddl:
                    break
            await container.sync_point()
            total_time = time.time_ns()-st
            total_tokens = total_count*per_batch_size[0]*per_batch_size[1]
            overall_tps = (total_tokens/total_time)*1e9
            perdecoder_tps = overall_tps*self.num_layers
            return overall_tps, perdecoder_tps
        overall_forward_tps, perdecoder_forward_tps = await test(forward_once)
        overall_forwardbackward_tps, perdecoder_forwardbackward_tps = await test(forward_backward_once)
        return (overall_forward_tps, perdecoder_forward_tps, overall_forwardbackward_tps, perdecoder_forwardbackward_tps)

    async def get_prefill_throughoutput(
        self,
        per_batch_size: Tuple[int, int] = (1, 1024),
        batch_count: int = 10,
        time_limit: float = -1,
        layers_cache_provider: Optional[LayersCacheProvider] = None,
        container: Optional[Container] = None,
    ) -> Tuple[float, float]:
        num_blocks = (per_batch_size[1]+flags.BLOCK_SIZE-1)//flags.BLOCK_SIZE
        if container is None:
            container = NonAsyncContainer(torch.cuda.current_stream())
        if layers_cache_provider is None:
            layer_blocks_holder = LayerBlocksHolder.create_contigious_cacher(
                self.block_info,
                num_blocks=num_blocks,
                device=self.device,
                dtype=torch.float16,
            )
            layers_cache_provider = SimpleLayersCacheProvider(
                layers_blocks_holder=[layer_blocks_holder]*self.num_layers,
                events=None,
            )
        prefill_cache_provider = SimpleLayersPrefillCacheProvider(
            seq_ptunings=[None]*per_batch_size[0],
            cache_provider=layers_cache_provider,
        )
        input_lens = [per_batch_size[1]]*per_batch_size[0]
        qheads = self.decoders[0].config.num_query_heads
        hidden_states = torch.zeros(
            (
                per_batch_size[0]*per_batch_size[1],
                self.block_info.head_dim*qheads
            ),
            device=self.device,
            dtype=torch.float16,
        )
        position_ids = torch.tensor([
            i
            for _ in range(per_batch_size[0])
            for i in range(per_batch_size[1])
        ],
            device=self.device,
            dtype=torch.int32,
        )
        slot_mapping = position_ids.to(torch.int64)

        async def run_once():
            await self.multi_seqs_prefill_forward(
                container=container,
                hidden_states=hidden_states,
                position_ids=position_ids,
                input_lens=input_lens,
                layers_cache_provider=prefill_cache_provider,
                slot_mapping=slot_mapping,
                layers_lora_hook_provider=None,
            )
        # warm up
        await run_once()
        await container.sync_point()

        st = time.time_ns()
        ddl = time.time_ns()+int(time_limit*1e9)
        total_count = 0
        for _ in range(batch_count):
            await run_once()
            total_count += 1
            if ddl > st and time.time_ns() > ddl:
                break
        await container.sync_point()
        total_time = time.time_ns()-st
        total_tokens = total_count*per_batch_size[0]*per_batch_size[1]
        overall_tps = (total_tokens/total_time)*1e9
        perdecoder_tps = overall_tps*self.num_layers
        return (overall_tps, perdecoder_tps)

    async def get_generating_throughoutput(
        self,
        parallel_count: int = 8,
        seq_len: int = 128,
        repeat_times: int = 1,
        time_limit: float = -1,
        layers_cache_provider: Optional[LayersCacheProvider] = None,
        container: Optional[Container] = None,
    ) -> Tuple[float, float]:

        per_seq_blocks = ((seq_len+flags.BLOCK_SIZE)//flags.BLOCK_SIZE)
        num_blocks = per_seq_blocks*parallel_count
        if container is None:
            container = NonAsyncContainer(torch.cuda.current_stream())
        if layers_cache_provider is None:
            layes_blocks_holder = \
                LayerBlocksHolder.create_contigious_cacher(
                    block_info=self.block_info,
                    num_blocks=num_blocks,
                    device=self.device,
                    dtype=torch.float16,
                )
            layers_cache_provider = SimpleLayersCacheProvider(
                layers_blocks_holder=[layes_blocks_holder]*self.num_layers,
                events=None,
            )
        qheads = self.decoders[0].config.num_query_heads
        hidden_states = torch.zeros(
            (
                parallel_count,
                1,
                self.block_info.head_dim*qheads
            ),
            device=self.device,
            dtype=torch.float16,
        )

        async def run_once():
            for i in range(seq_len):
                def gen_input():
                    positions_t = torch.tensor([i]*parallel_count, dtype=torch.int32, device=self.device)
                    context_lens_t = torch.tensor([i+1]*parallel_count, dtype=torch.int32, device=self.device)
                    slot_mapping = torch.tensor([i+(p*seq_len) for p in range(parallel_count)], dtype=torch.int64, device=self.device)
                    num_blocks = (i+flags.BLOCK_SIZE)//flags.BLOCK_SIZE
                    block_tables = torch.tensor(
                        data=[
                            [b+p*per_seq_blocks for b in range(num_blocks)]
                            for p in range(parallel_count)
                        ],
                        dtype=torch.int32, device=self.device).view(parallel_count, -1)
                    return positions_t, context_lens_t, slot_mapping, block_tables
                positions_t, context_lens_t, slot_mapping, block_tables =\
                    await container.step("gen_input", gen_input)
                await self.multi_seqs_generating_forward(
                    container=container,
                    hidden_states=hidden_states,
                    position_ids=positions_t,
                    layers_cache_provider=layers_cache_provider,
                    context_len_max=i+1,
                    context_lens=context_lens_t,
                    block_tables=block_tables,
                    slot_mapping=slot_mapping,
                    layers_lora_hook_provider=None
                )

        # warm up
        await run_once()
        await container.sync_point()

        st = time.time_ns()
        ddl = time.time_ns()+int(time_limit*1e9)
        total_count = 0
        for _ in range(repeat_times):
            await run_once()
            total_count += 1
            if ddl > st and time.time_ns() > ddl:
                break
        await container.sync_point()
        total_time = time.time_ns()-st
        total_tokens = total_count*parallel_count*seq_len
        overall_tps = (total_tokens/total_time)*1e9
        perdecoder_tps = overall_tps*self.num_layers
        return (overall_tps, perdecoder_tps)

    async def gen_profile(
        self,
        container: Container,
        train_kwargs: Optional[Dict[str, Any]] = None,
        generating_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if train_kwargs is None:
            train_kwargs = {}
        _, perdecoder_forward_tps, _, perdecoder_forwardbackward_tps =\
            await self.get_forward_backward_throughoutput(container=container, **train_kwargs)
        if generating_kwargs is None:
            generating_kwargs = {}
        _, perdecoder_tps =\
            await self.get_generating_throughoutput(container=container, **generating_kwargs)
        return int(perdecoder_forward_tps), int(perdecoder_forwardbackward_tps), int(perdecoder_tps)

    def __del__(self):
        for i, d in enumerate(self.decoders):
            if d.fuse_wrap is not None:
                d.fuse_wrap.base = None  # type:ignore
            d.fuse_wrap = None
            self.decoders[i] = None  # type:ignore
