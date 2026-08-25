from dataclasses import dataclass, field
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
from operators.core.sampler.real import batch_sample


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
        num_blocks_req = (current_slots+1+(flags.BLOCK_SIZE-1))//flags.BLOCK_SIZE
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


E = TypeVar('E')


def pad(mutli_lists: List[List[E]], default: E) -> List[List[E]]:
    pad_len = max([len(l) for l in mutli_lists])
    ls = [l+[default]*(pad_len-len(l))
          for l in mutli_lists]
    return ls


class InferRunner:
    def __init__(self, head: ModelHead, tail: ModelTail,
                 decoders: DecodersGroup, layer_cachers: List[LayerBlocksHolder],
                 tokenizer: ExtendTokenizer
                 ) -> None:
        self.layer_cachers = SimpleLayersCacheProvider(layers_blocks_holder=layer_cachers, events=None)
        self.head = head
        self.decoders = decoders
        self.tail = tail
        self.tokenizer = tokenizer

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
        input_positions = [i for prompt_len in prompt_lens for i in range(prompt_len)]

        def gen_input(prompts: List[int], slot_mapping: List[int], input_positions: List[int], input_lens: List[int], user_models: List[Union[None, List[int], Tuple[LoraRenderConfig, List[int]]]]):
            hidden_states = torch.tensor(prompts, dtype=torch.int32)
            slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int64)
            input_positions_tensor = torch.tensor(input_positions, dtype=torch.int32)
            ptunings = [(p if isinstance(p, list) else None) for p in user_models]
            loras = [(l if isinstance(l, tuple) else None) for l in user_models]
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

        next_tokens, log_probs = await batch_sample(container=container, seqs_logits=logits,
                                                    sampling_params=[s.sampling_params for s in seqs],
                                                    output_token_ids=[s.seq.core.prompt_token_ids for s in seqs],
                                                    prompt_token_ids=[s.seq.core.output_token_ids for s in seqs]
                                                    )
        return next_tokens

    async def generating_step(self, container: Container, seqs: List[SeqComplex], next_tokens: List[int]):
        prompt_lens = [s.seq.core.length_data.prompt_len for s in seqs]
        generating_lens = [s.seq.core.length_data.output_count for s in seqs]
        cached_lens = [p+g-1 for (p, g) in zip(prompt_lens, generating_lens)]  # latest generated token (this input token) has not already been in block cache
        positions = cached_lens  # position for rope of this input token

        user_models = [sc.user_model for sc in seqs]
        ptunings = [(p if isinstance(p, list) else None) for p in user_models]
        loras = [(l if isinstance(l, tuple) else None) for l in user_models]

        context_lens = [(len(ptuning or [])*flags.BLOCK_SIZE)+position+1 for position, ptuning in zip(positions, ptunings)]  # when attention, this input token is already been cached
        slots = [s.allocator.append_slot() for s in seqs]
        block_tables = pad([(ptuning or []) + s.allocator.occupied_blocks for s, ptuning in zip(seqs, ptunings)], 0)
        context_lens_max = max(context_lens)

        def gen_input(block_tables: List[List[int]], context_lens: List[int], positions: List[int], slots: List[int],
                      next_tokens: List[int], loras: List[Optional[Tuple[LoraRenderConfig, List[int]]]]):
            block_tables_t = torch.tensor(block_tables, dtype=torch.int32)
            context_lens_t = torch.tensor(context_lens, dtype=torch.int32)
            positions_t = torch.tensor(positions, dtype=torch.int32)
            slot_mapping_t = torch.tensor(slots, dtype=torch.int64)
            input_tokens_tensor = torch.tensor(next_tokens, dtype=torch.int32)
            layers_lora_provider = SimpleLayersLoraHookProvider(seq_loras=loras, seq_lens=[1]*len(next_tokens), cache_provider=self.layer_cachers)
            return block_tables_t, context_lens_t, positions_t, slot_mapping_t, input_tokens_tensor, layers_lora_provider

        block_tables_t, context_lens_t, positions_t, slot_mapping_t, input_tokens_tensor, layers_lora_hook_provider = \
            await container.step("gen_input", gen_input,
                                 block_tables=block_tables, context_lens=context_lens, positions=positions, slots=slots, next_tokens=next_tokens, loras=loras)

        hidden_states = (await self.head.infer_to_embedding_tokens(container=container, input_tokens_tensor=input_tokens_tensor, dtype=torch.float16)).unsqueeze(1)
        hidden_states = await self.decoders.multi_seqs_generating_forward(
            container=container,
            hidden_states=hidden_states,
            position_ids=positions_t,  # [8],device, int32
            layers_cache_provider=self.layer_cachers,
            context_len_max=context_lens_max,  # 9
            context_lens=context_lens_t,  # [9],device, int32
            block_tables=block_tables_t,  # [[]], device, int32
            slot_mapping=slot_mapping_t,  # [], device int64
            layers_lora_hook_provider=layers_lora_hook_provider,
        )

        logits = await self.tail.to_logits(container=container, hidden_states=hidden_states.squeeze(1))

        generated_tokens, log_probs = await batch_sample(container=container, seqs_logits=logits,
                                                         sampling_params=[s.sampling_params for s in seqs],
                                                         output_token_ids=[s.seq.core.prompt_token_ids for s in seqs],
                                                         prompt_token_ids=[s.seq.core.output_token_ids for s in seqs]
                                                         )
        return generated_tokens

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
                print(f"total gen: {total_generated/1000:.2f}K {total_generated/(time.time()-start_time):.2f}TPS", end="\r")
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

    target_model = "teared-Llama-2-7B-GPTQ"
    decoders_impl_name = "our_fuse"
    models = ModelsLib(split_qllm_models.lib_path)
    model = models.models_storage[target_model]
    tokenizer = ExtendTokenizer.load_tokenier(
        model.tokenizer_path,
        model.config.tokenizer_special_tokens)
    device = torch.zeros((0,), device="cuda:0").device
    block_info = model.config.block_info
    num_blocks = 128
    free_blocks = [i for i in range(num_blocks)]

    with torch.device(device):
        head = ModelHead(model.config.model_head_config)
        head.load_from_data_dir(model.model_head_path)
        tail = ModelTail(model.config.model_tail_config)
        tail.load_from_data_dir(model.model_tail_path)
        decoders = make_decoders_from_files(
            implement=decoders_impl_name,
            config=model.config.quant_decoder_config,
            state_files=[model.decoder_path(i) for i in range(model.config.num_decoders)],
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
    runner = InferRunner(head, tail, decoders, layer_cachers, tokenizer)

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

    with torch.device(device):
        user_model = make_user_model(
            cfg=user_model_cfg,
            state_dict=torch.load(user_model_data, map_location=device),
            get_block=free_blocks.pop, layer_cachers=layer_cachers
        )

    def make_sample_seq_complex():
        seq_complexs: List[SeqComplex] = []
        available_blocks = free_blocks.copy()

        seq0 = tokenizer.init_seq(prompt=f"once upon a time", max_tokens=128, ignore_eos=False)
        allocator0 = CacheAllocator(available_blocks.pop)
        sampling_params0 = SamplingParams(temperature=0.0, logprobs=1, presence_penalty=1, frequency_penalty=1, repetition_penalty=2)
        seq_complexs.append(SeqComplex(seq0, allocator0, sampling_params0, user_model=None))

        built_prompt = user_model_cfg.build_prompt_str.format("类型#裤*风格#英伦*风格#简约")
        seq1 = tokenizer.init_seq(prompt=built_prompt, max_tokens=128, ignore_eos=False)
        allocator1 = CacheAllocator(available_blocks.pop)
        sampling_params1 = SamplingParams(temperature=0.0, logprobs=1)
        seq_complexs.append(SeqComplex(seq1, allocator1, sampling_params1, user_model=user_model))

        return seq_complexs

    def make_bench_complex(seq_count: int, per_seq_tokens: int):
        seq_complexs: List[SeqComplex] = []
        available_blocks = free_blocks.copy()
        for _ in range(seq_count):
            seq = tokenizer.init_seq(prompt=f"", max_tokens=per_seq_tokens, ignore_eos=True, stop_str=None)
            allocator = CacheAllocator(available_blocks.pop)
            sampling_params = SamplingParams(temperature=0.0, logprobs=1)
            seq_complexs.append(SeqComplex(seq, allocator, sampling_params, user_model=user_model))
        return seq_complexs

    stream = cast(torch.cuda.Stream, torch.cuda.Stream())

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
