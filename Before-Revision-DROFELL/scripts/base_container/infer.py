import json
import os
from typing import Callable, List, cast

import torch
from operators import flags
from operators.core.ptuning.common import PtuningConfig
from operators.core.ptuning.real import Ptuning
from operators.helpers.model_lib import ModelsLib
from operators.core.tokenizer.real import ExtendTokenizer
from operators.assemble.decoders.real import make_decoders_from_files
from operators.core.tail.real import ModelTail
from operators.core.head.real import ModelHead
from operators.core.block.real import LayerBlocksHolder
from operators.core.sampler.common import SamplingParams
from scripts.base.infer import SeqComplex, CacheAllocator, InferRunner
import extend_asyncio as asyncio
from containers.fast import FastContainer
from containers.measure import MeasureContainer
from containers.pressure import PressureReleaseContainer

models = ModelsLib("split_qllm_models")
model = models.models_storage["teared-Llama-2-7B-GPTQ"]
tokenizer = ExtendTokenizer.load_tokenier(model.tokenizer_path, model.config.tokenizer_special_tokens)


device = torch.zeros((0,), device="cuda").device
block_info = model.config.block_info
num_blocks = 128
free_blocks = [i for i in range(num_blocks)]
ptuning_cfg = PtuningConfig(
    num_layers=model.config.num_decoders, head_dim=model.config.head_dim, num_key_value_heads=model.config.num_key_value_heads,
    prefix_len=128, dropout_p=0.1, build_prompt_str="[instruction]{}\n[generation]"
)


def make_ptuning(get_block: Callable[[], int]):
    ptuning = Ptuning(ptuning_cfg)
    ptuning.load_state_dict(torch.load("data/trained_model/ptuning_7B.pt", map_location=device))
    ptuning_kv_cache = ptuning.get_inference_ptuning_kv_cache(torch.float16)
    ptuning_len = ptuning.cfg.prefix_len
    assert ptuning_len % flags.BLOCK_SIZE == 0
    ptuning_required_blocks_num = ptuning_len//flags.BLOCK_SIZE
    ptuning_blocks = [get_block() for _ in range(ptuning_required_blocks_num)]
    # ptuning_slots = [b*flags.BLOCK_SIZE+i for b in ptuning_blocks for i in range(flags.BLOCK_SIZE)]
    # ptuning_shape = (ptuning_len, ptuning.cfg.num_key_value_heads, ptuning.cfg.head_dim)
    for (k, v), cacher in zip(ptuning_kv_cache, layer_cachers):
        bk, bv = cacher.block_info.tensor_kv_to_block_kv(k, v)
        cacher.swap_in_from_mapping({
            idx: (bk[i], bv[i])
            for i, idx in enumerate(ptuning_blocks)
        })
    return ptuning_blocks


with torch.device(device):
    head = ModelHead(model.config.model_head_config)
    head.load_from_data_dir(model.model_head_path)
    tail = ModelTail(model.config.model_tail_config)
    tail.load_from_data_dir(model.model_tail_path)
    decoders = make_decoders_from_files(
        implement="our_fuse",
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
    ptuning_info = make_ptuning(free_blocks.pop)

runner = InferRunner(head, tail, decoders, layer_cachers, tokenizer)


def make_sample_seq_complex():
    seq_complexs: List[SeqComplex] = []
    available_blocks = free_blocks.copy()

    seq0 = tokenizer.init_seq(prompt=f"once upon a time", max_tokens=128, ignore_eos=False)
    allocator0 = CacheAllocator(available_blocks.pop)
    sampling_params0 = SamplingParams(temperature=0.0, logprobs=1)
    seq_complexs.append(SeqComplex(seq0, allocator0, sampling_params0, ptuning=None))

    built_prompt = ptuning_cfg.build_prompt_str.format("类型#裤*风格#英伦*风格#简约")
    seq1 = tokenizer.init_seq(prompt=built_prompt, max_tokens=128, ignore_eos=False)
    allocator1 = CacheAllocator(available_blocks.pop)
    sampling_params1 = SamplingParams(temperature=0.0, logprobs=1)
    seq_complexs.append(SeqComplex(seq1, allocator1, sampling_params1, ptuning=ptuning_info))

    return seq_complexs


def make_bench_complex(count: int):
    seq_complexs: List[SeqComplex] = []
    available_blocks = free_blocks.copy()
    for _ in range(count):
        seq = tokenizer.init_seq(prompt=f"", max_tokens=128, ignore_eos=True, stop_str=None)
        allocator = CacheAllocator(available_blocks.pop)
        sampling_params = SamplingParams(temperature=0.0, logprobs=1)
        seq_complexs.append(SeqComplex(seq, allocator, sampling_params, ptuning=None))
    return seq_complexs


stream = cast(torch.cuda.Stream, torch.cuda.Stream())


async def amain():
    mode = "pressure"
    if mode == "fast":
        container = FastContainer(stream)
    elif mode == "measure":
        container = MeasureContainer(stream)
    elif mode == "pressure":
        operate_load_file = os.path.join(os.path.dirname(__file__), "infer_profile.json")
        with open(operate_load_file, "r") as f:
            operate_load = json.load(f)
        container = PressureReleaseContainer(stream, operate_load)
    else:
        raise ValueError(mode)

    with container.mode('inference'):
        loop = asyncio.get_event_loop()
        loop.set_blocking_scheduler(container)
        sample_seq = make_sample_seq_complex()
        await runner.run_seqs(container, sample_seq,  print_output=True)
        torch.cuda.synchronize()
        container.reset()
        loop.clear_profile()
        bench_seqs = make_bench_complex(1)
        await runner.run_seqs(container, bench_seqs,  print_output=False)
        print(f"blocking time: {loop.blocking_calltime_ms:.4f}ms")
        print(f"nonblocking time: {loop.non_blocking_calltime_ms:.4f}ms")
        profile_file = os.path.join(os.path.dirname(__file__), "results", f"{mode}_infer.json")
        container.dump_profile(profile_file)
        if isinstance(container, MeasureContainer):
            profile_out = os.path.join(os.path.dirname(__file__), "infer_profile.json")
            with open(profile_out, "w") as f:
                json.dump(container.compute_stats(), f)
asyncio.config_impl("golang", True)
asyncio.run(amain())
