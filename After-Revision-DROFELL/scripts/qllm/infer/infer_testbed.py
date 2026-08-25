from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple,
import torch

from operators.cache_engine.vllm_fix import VllmFixCacheEngine
from operators.container import Container
from operators.core.tokenizer.real import ExtendTokenizer
from operators.core.tail.real import ModelTail
from operators.core.head.real import ModelHead
from operators.assemble.decoders.real import DecodersImplement, make_decoders_from_files
from operators.core.ptuning.common import PtuningConfig
from operators.core.ptuning.real import Ptuning
from operators.helpers.cuda_utils import memory
from operators.helpers.model_lib import ModelsLib

from vllm_algo.interface.block_space_io import VirtualBlockSpaceIO
from vllm_algo.interface.block_space import VirtualBlockSpaceHolder
from vllm_algo.interface.infer_request_outputs import RequestOutput
from vllm_algo.interface.sequence import SequenceGroup
from vllm_algo.interface.sampling_params import SamplingParams
from vllm_algo.interface.basic_types import TokenID, VirtualDevice
from vllm_algo.implements.virtual_block_space_io import VllmLikeVirtualBlockSpaceIO
from vllm_algo.implements.virtual_block_space_holder import VllmLikeVirtualBlockSpaceHolder
from vllm_algo.implements.scheduler import SchedulerConfig, VllmScheduler
from vllm_algo.implements.infer_manager import VllmInferManager
from vllm_algo.implements.infer_engine import LocalMachineInferEngine

import split_qllm_models
import data
# from vllm_like.implements.scheduler import VllmScheduler, SchedulerConfig
# from vllm_like.interface.block_space import VirtualBlockSpaceHolder
# from vllm_like.interface.sequence import SequenceGroup
# from vllm_like.interface.infer_request_outputs import RequestOutput
# from vllm_like.implements.scheduler import VllmScheduler, SchedulerConfig
# from vllm_like.implements.infer_manager import VllmInferManager
# from vllm_like.implements.virtual_block_space_io import VirtualBlockSpaceIO, VllmLikeVirtualBlockSpaceIO
# from vllm_like.implements.virtual_block_space_holder import VllmLikeVirtualBlockSpaceHolder
# from vllm_like.implements.infer_engine import LocalMachineInferEngine


@dataclass
class InferConfig:
    model_name: str = "teared-Llama-2-7B-GPTQ"
    max_layers_to_load: int = -1
    max_allowed_len: int = 128*1024
    layers: int = 0
    perfix_encoder_confg: PtuningConfig = PtuningConfig(
        build_prompt_str="[instruction]{}\n[generation]",
        dropout_p=0.1,
        prefix_len=128
    )
    encoder_data: str = data.trained_model_path_of("ptuning_7B.pt")
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=128*1024,
        max_num_seqs=256,
        max_model_len=128*1024,
        max_paddings=128*1024,
    )
    decoder_implement: DecodersImplement = "exllama_fuse"
    cache_kernel_implement: Literal['vllm', 'dual_table'] = "dual_table"
    default_sampling_params: SamplingParams = SamplingParams(
        temperature=0.0, logprobs=1, ignore_eos=False, max_tokens=128)
    blocks_count: int = 640

    def __post_init__(self):
        # process model config
        models = ModelsLib(split_qllm_models.lib_path)
        self.model_info = models.models_storage[self.model_name]
        self.base_model_config = self.model_info.config
        self.block_info = self.base_model_config.block_info
        self.base_model_config.max_seq_len = self.max_allowed_len
        if self.max_layers_to_load > 0:
            assert self.max_layers_to_load <= self.base_model_config.num_decoders
            self.base_model_config.num_decoders = self.max_layers_to_load
            self.layers = self.base_model_config.num_decoders
            print(f"load only {self.layers} layers")
        else:
            self.layers = self.base_model_config.num_decoders
        self.tokenizer_path = self.model_info.tokenizer_path

        self.perfix_encoder_confg.num_layers = self.layers
        self.perfix_encoder_confg.head_dim = self.base_model_config.head_dim
        self.perfix_encoder_confg.num_key_value_heads = self.base_model_config.num_key_value_heads


class InferTestBed:
    def __init__(self, cfg: InferConfig) -> None:
        self.cfg = cfg
        self.device = torch.zeros((0,), device="cuda").device
        self.tokenizer = self.load_tonkenizer()
        self.load_base_model()
        self.block_io, self.cache_engine = self.load_cache_engine()
        self.engine = self.load_execute_engine()
        self.scheduler = VllmScheduler(self.cfg.scheduler_config)
        self.manager = VllmInferManager(scheduler=self.scheduler,
                                        block_space_io=self.block_io,
                                        tokenizer=self.tokenizer)
        self.ptuning_handler = self.load_user_model()
        self.request_offset = 0
        self.request_counter = 0
        self.current_request_count = 0

    def reset(self):
        self.scheduler.reset()
        self.block_io.reset()
        self.request_offset = 0
        self.request_counter = 0
        self.current_request_count = 0

    def load_tonkenizer(self):
        tokenizer = ExtendTokenizer.load_tokenier(
            path=self.cfg.tokenizer_path,
            cfg=self.cfg.base_model_config.tokenizer_special_tokens
        )
        return tokenizer

    def load_base_model(self):
        with self.device:
            self.model_head = ModelHead(
                config=self.cfg.base_model_config.model_head_config)
            self.model_tail = ModelTail(
                config=self.cfg.base_model_config.model_tail_config)
            self.model_head.load_from_data_dir(
                self.cfg.model_info.model_head_path)
            self.model_tail.load_from_data_dir(
                self.cfg.model_info.model_tail_path)
            self.decoders = make_decoders_from_files(
                implement=self.cfg.decoder_implement,
                config=self.cfg.base_model_config.quant_decoder_config,
                state_files=[
                    self.cfg.model_info.decoder_path(i)
                    for i in range(self.cfg.base_model_config.num_decoders)
                ],
            )

    def load_cache_engine(self):
        virtual_cpu_block_space_holder: VirtualBlockSpaceHolder = VllmLikeVirtualBlockSpaceHolder(
            device=VirtualDevice.CPU, block_size=32, num_blocks=self.cfg.blocks_count)
        virtual_gpu_block_space_holder: VirtualBlockSpaceHolder = VllmLikeVirtualBlockSpaceHolder(
            device=VirtualDevice.GPU, block_size=32, num_blocks=self.cfg.blocks_count)
        virtual_block_space_io: VirtualBlockSpaceIO = VllmLikeVirtualBlockSpaceIO(
            gpu_space_holder=virtual_gpu_block_space_holder,
            cpu_space_holder=virtual_cpu_block_space_holder
        )
        # load actual cache engine
        cache_engine = VllmFixCacheEngine(
            block_info=self.cfg.block_info,
            num_cpu_blocks=virtual_cpu_block_space_holder.get_total_blocks(),
            num_device_blocks=virtual_gpu_block_space_holder.get_total_blocks(),
            num_layers=self.cfg.layers,
            device=self.device,
            mode=self.cfg.cache_kernel_implement,
            translate=False,
        )
        return virtual_block_space_io, cache_engine

    def free_blocks_count(self) -> int:
        return self.block_io.gpu_space_holder.get_num_free_blocks()

    def load_execute_engine(self):
        return LocalMachineInferEngine(
            model_head=self.model_head,
            model_tail=self.model_tail,
            all_decoders=self.decoders,
            cache_engine=self.cache_engine,
            device=self.device
        )

    def load_user_model(self):
        with self.device:
            encoder = Ptuning(config=self.cfg.perfix_encoder_confg)
        if self.cfg.encoder_data != "":
            encoder.load_from_file(self.cfg.encoder_data)
        encoder = encoder.eval()
        ptuning_kv_cache = encoder.get_inference_ptuning_kv_cache(
            torch.float16)
        ptuning_kv_cache = [t.cpu().contiguous() for t in ptuning_kv_cache]
        ptuning_kv_cache = [
            self.cfg.block_info.tensor_kv_to_block_kv(
                key_states=k, val_states=v
            )
            for k, v in ptuning_kv_cache
        ]
        handler = self.manager.add_ptuning(
            ptuning_id="user_model",
            build_prompt_str=encoder.build_prompt_str,
            ptuning_kv_cache_cpu=[(t[0], t[1]) for t in ptuning_kv_cache],
            engine=self.cache_engine
        )
        return handler

    def fill_user_model(self, len: int):
        self.manager.ptuning_handlers = {}
        with self.device:
            cfg = PtuningConfig(
                prefix_len=len,
                build_prompt_str="{}",
                num_layers=self.cfg.layers,
                head_dim=self.cfg.base_model_config.head_dim,
                num_key_value_heads=self.cfg.base_model_config.num_key_value_heads
            )
            encoder = Ptuning(cfg)
        encoder = encoder.eval()
        ptuning_kv_cache = encoder.get_inference_ptuning_kv_cache(
            torch.float16)
        ptuning_kv_cache = [t.cpu().contiguous() for t in ptuning_kv_cache]
        ptuning_kv_cache = [
            self.cfg.block_info.tensor_kv_to_block_kv(
                key_states=k, val_states=v
            )
            for k, v in ptuning_kv_cache
        ]
        handler = self.manager.add_ptuning(
            ptuning_id="user_model",
            build_prompt_str=encoder.build_prompt_str,
            ptuning_kv_cache_cpu=[(t[0], t[1]) for t in ptuning_kv_cache],
            engine=self.cache_engine
        )
        self.ptuning_handler = handler
        return handler

    def add_request(
        self,
            user_model: bool,
            sampling_parameters: Optional[SamplingParams] = None,
            prompt: Optional[str] = None,
            prompt_token_ids: Optional[List[TokenID]] = None
    ):
        request_id = f"{self.request_counter}"
        self.request_counter += 1
        if sampling_parameters is None:
            sampling_parameters = self.cfg.default_sampling_params
        assert self.manager.add_request(
            request_id=request_id,
            sampling_parameters=sampling_parameters,
            ptuning_id=(
                self.ptuning_handler.ptuning_id if user_model else None),
            prompt=prompt,
            prompt_token_ids=prompt_token_ids
        )
        self.current_request_count += 1

    async def do_prefill(self, conatiner: Container, prefill_hook: Optional[Callable[[List[Tuple[int, List[int]]]], None]] = None):
        prefill_count = 0
        prefill_step = 0
        while True:
            target_seqs_id_and_prefill_metas = self.manager.step_pre_prefill()
            if len(target_seqs_id_and_prefill_metas) == 0:
                assert prefill_count == self.current_request_count, f"{prefill_count} != {self.current_request_count}"
                assert prefill_step == 1
                return
            prefill_step += 1
            seq_groups: List[SequenceGroup] = [meta.seq_group for (
                _, meta) in target_seqs_id_and_prefill_metas]
            sampler_output = await self.engine.infer_prefill(conatiner, target_seqs_id_and_prefill_metas)
            outputs: List[RequestOutput] = self.manager.step_post_process(
                sampler_output=sampler_output, seq_groups=seq_groups)
            if prefill_hook is not None:
                prefill_hook([(int(o.request_id), o.prompt_token_ids)
                             for o in outputs])
            prefill_count += len(outputs)

    async def do_infer(self,
                       container: Container,
                       prefill_hook: Optional[Callable[[
                           List[Tuple[int, List[int]]]], None]] = None,
                       prefill_finish_hook: Optional[Callable[[
                       ], None]] = None,
                       generating_hook: Optional[Callable[[List[Tuple[int, RequestOutput]]], None]] = None):
        await self.do_prefill(container, prefill_hook)
        if prefill_finish_hook is not None:
            prefill_finish_hook()

        generated: Dict[int, str] = {}

        async def do_generating():
            nonlocal generated
            while True:
                generating_meta = self.manager.step_pre_generating()
                if generating_meta is None:
                    break
                if generating_meta.memory_operations and not generating_meta.seq_group_metas:
                    raise NotImplementedError
                    # self.engine.start_and_wait_memory_operations(
                    #     generating_meta.memory_operations)
                else:
                    seq_groups: List[SequenceGroup] = []
                    for (_, meta) in generating_meta.seq_group_metas:
                        seq_groups.append(meta.seq_group)
                    sampler_output = await self.engine.infer_generating(
                        container=container,
                        memory_operations=generating_meta.memory_operations,
                        seq_group_metas=generating_meta.seq_group_metas
                    )
                    outputs: List[RequestOutput] = self.manager.step_post_process(
                        sampler_output=sampler_output, seq_groups=seq_groups)
                    for o in outputs:
                        seq_i = int(o.request_id)
                        assert len(o.outputs) == 1
                        if o.outputs[0].finished():
                            generated[seq_i] = o.outputs[0].text
                    if generating_hook is not None:
                        generating_hook([(int(o.request_id), o)
                                        for o in outputs])
        await do_generating()
        self.request_offset = self.request_counter
        self.current_request_count = 0
        return generated


async def run_prefill(container: Container, test_bed: InferTestBed, count: int, memory: bool = False):
    test_bed.add_request(user_model=True, sampling_parameters=SamplingParams(
        temperature=0.0, logprobs=1, max_tokens=128),
        prompt="类型#裤*风格#英伦*风格#简约",
    )
    # warm up
    await test_bed.do_prefill(container)
    torch.cuda.synchronize()
    test_bed.reset()
    # warm up complete

    async def test_once(fill_fn: Callable[[], None]):
        prefill_tokens = 0

        def on_prefill(seqs: List[Tuple[int, List[int]]]):
            nonlocal prefill_tokens
            for _, s in seqs:
                prefill_tokens += len(s)
        fill_fn()
        st = time.time_ns()
        await test_bed.do_prefill(container, on_prefill)
        time_comsuming = (time.time_ns()-st)/(10**9)
        print(
            f"complete {prefill_tokens} @ {time_comsuming} {prefill_tokens/time_comsuming}")
        print(f"{test_bed.free_blocks_count()}")
        torch.cuda.synchronize()
        test_bed.reset()
        return prefill_tokens, time_comsuming

    async def test_memory(fill_fn: Callable[[], None]):
        torch.cuda.memory._record_memory_history()  # type: ignore
        fill_fn()
        await test_bed.do_prefill(container)
        record = torch.cuda.memory._snapshot(
        )["device_traces"][test_bed.device.index]  # type: ignore
        torch.cuda.memory._record_memory_history(enabled=None)  # type: ignore
        torch.cuda.synchronize()
        test_bed.reset()
        return record

    def add_request(count: int, prompt_len: int):
        for _ in range(count):
            test_bed.add_request(user_model=True, sampling_parameters=SamplingParams(
                temperature=0.0, logprobs=1, max_tokens=128),
                prompt_token_ids=[3]*prompt_len,
            )
    report = {
        f"256 {count}": [await test_once(lambda: add_request(count, 256)) for _ in range(3)],
        f"512 {count}": [await test_once(lambda: add_request(count, 512)) for _ in range(3)],
        f"1024 {count}": [await test_once(lambda: add_request(count, 1024)) for _ in range(3)]
    }
    if memory:
        report["memory"] = await test_memory(lambda: add_request(count, 256))
    return report


@dataclass
class Report:
    info: str = ""
    memory_record: Optional[List[Any]] = None
    prefill_time: float = 0
    prefill_tokens: int = 0
    generating_time: float = 0
    generating_tokens: int = 0
    generating_time_record: List[Tuple[int, float]
                                 ] = field(default_factory=list)

    def __str__(self) -> str:
        return "" +\
            f"prefill: {self.prefill_time*(10**3):.2f}ms of {self.prefill_tokens}, speed: {self.prefill_tokens/(self.prefill_time):.2f} tokens/s\n" +\
            f"generating: {self.generating_time*(10**3):.2f}ms of {self.generating_tokens}, speed: {self.generating_tokens/(self.generating_time):.2f} tokens/s"


async def run(cfg: InferConfig, container: Container, total_tokens: int = 20*1000, do_memory: bool = False):
    test_bed = InferTestBed(cfg)

    def add_std_request(count: int):
        for _ in range(count):
            test_bed.add_request(user_model=True, sampling_parameters=SamplingParams(
                temperature=0.0, logprobs=1, max_tokens=128),
                prompt="类型#裤*风格#英伦*风格#简约",
            )

    # current check
    add_std_request(1)
    await test_bed.do_infer(container)
    start_memory = memory.current_memory_status()
    add_std_request(8)
    output = await test_bed.do_infer(container)
    for seq_i, out in output.items():
        print(f"{seq_i}: {out}")
    current_memory = memory.current_memory_status()
    for k, v in current_memory.items():
        v0 = start_memory[k]
        assert v0.requested == v.requested, f"{v0.requested}!={v.requested}@{k}"
    torch.cuda.synchronize()

    async def run_single_test(fill_fn: Callable[[], None], memory: bool = False):
        # warm up:
        # fill_fn()
        # test_bed.do_infer()
        # torch.cuda.synchronize()

        report = Report()

        # memory test
        if memory:
            torch.cuda.memory._record_memory_history()  # type: ignore
            fill_fn()
            await test_bed.do_infer(container)
            record = torch.cuda.memory._snapshot(
            )["device_traces"][test_bed.device.index]  # type: ignore
            torch.cuda.memory._record_memory_history(
                enabled=None)  # type: ignore
            torch.cuda.synchronize()
            report.memory_record = record

        # speed test
        fill_fn()
        start_time = time.time_ns()
        prefill_end_time = 0
        last_free_blocks_count = 0

        def on_prefill(seqs: List[Tuple[int, List[int]]]):
            nonlocal report
            for _, s in seqs:
                report.prefill_tokens += len(s)

        def on_prefill_finish():
            nonlocal prefill_end_time
            prefill_end_time = time.time_ns()

        def on_generating_hook(seqs: List[Tuple[int, RequestOutput]]):
            nonlocal report, last_free_blocks_count
            report.generating_tokens += len(seqs)
            t = time.time_ns()
            for (si, _) in seqs:
                report.generating_time_record.append((si, t))
                # if len(report.generating_time_record) % 128 == 0:
                print(f"{len(report.generating_time_record)/1000}K", end="\r")
            free_blocks_count = test_bed.free_blocks_count()
            if free_blocks_count != last_free_blocks_count:
                # print(f"free blocks count: {free_blocks_count}")
                last_free_blocks_count = free_blocks_count

        await test_bed.do_infer(
            container=container,
            prefill_hook=on_prefill,
            prefill_finish_hook=on_prefill_finish,
            generating_hook=on_generating_hook
        )
        report.prefill_time = (prefill_end_time-start_time)/(10**9)
        report.generating_time = (time.time_ns()-prefill_end_time)/(10**9)
        return report

    def add_request(count: int, user_model: bool, prompt_len: int, generating_len: int):
        for _ in range(count):
            test_bed.add_request(user_model=user_model, sampling_parameters=SamplingParams(
                temperature=0.0, logprobs=1, max_tokens=generating_len),
                prompt_token_ids=[3]*prompt_len,
            )
    report_mem = Report()
    if do_memory:
        print("do memory")
        report_mem: Report = await run_single_test(lambda: add_request(count=1, user_model=False, prompt_len=1, generating_len=total_tokens), memory=True)
        return {"memory": report_mem}
    else:
        print("do 1x")
        report1x = await run_single_test(lambda: add_request(count=1, user_model=False, prompt_len=1, generating_len=total_tokens), memory=False)
        print("do 2x")
        report2x = await run_single_test(lambda: add_request(count=2, user_model=False, prompt_len=1, generating_len=total_tokens//2), memory=False)
        print("do 4x")
        report4x = await run_single_test(lambda: add_request(count=4, user_model=False, prompt_len=1, generating_len=total_tokens//4), memory=False)
        print("do 8x")
        report8x = await run_single_test(lambda: add_request(count=8, user_model=False, prompt_len=1, generating_len=total_tokens//8), memory=False)

        return {
            "memory": report_mem,
            "1x": report1x,
            "2x": report2x,
            "4x": report4x,
            "8x": report8x,
        }


if __name__ == "__main__":
    import extend_asyncio as asyncio
    from operators.container.container import NonAsyncContainer
    # from operators.container.container import LeakDetectContainer
    cfg = InferConfig()
    device = torch.zeros((0,), device="cuda").device
    with torch.device(device):
        test_bed = InferTestBed(cfg)
    asyncio.config_impl("golang", True)
    stream = torch.cuda.Stream()
    # stream = torch.cuda.current_stream(device)

    async def amain():
        loop = asyncio.get_event_loop()
        container = NonAsyncContainer(stream)
        # from operators.metrics.base_container.containers.fast import FastContainer
        # container = FastContainer(stream)

        def on_generating(seqs: List[Tuple[int, RequestOutput]]):
            print("\x1b[2J\x1b[H")
            for seq_i, (_, out) in enumerate(seqs):
                text = (out.prompt+out.outputs[0].text).replace("\n", "\\n")
                print(f"SEQ-{seq_i}> {text}")

        with container.mode('inference'):
            test_bed.add_request(
                user_model=False,
                sampling_parameters=None,
                prompt="once upon a time"
            )
            test_bed.add_request(
                user_model=True,
                sampling_parameters=None,
                prompt="类型#裤*风格#英伦*风格#简约"
            )
            await test_bed.do_infer(
                container=container,
                generating_hook=on_generating
            )

    asyncio.run(amain())
