import argparse
from dataclasses import asdict, dataclass
from functools import partial
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast
from uuid import uuid4

import torch

from node import flags
from node.aux_node.protocols import req_providers
from operators.core.ptuning.real import Ptuning

from operators.core.lora.real import Lora

from operators.core.lora.common import LoraConfig
from operators.core.lora.render import LoraRenderConfig

from dealers.common.base_types import DeviceTensorCarrier
import extend_asyncio as asyncio
from node.aux_node.protocols import req_extra_info
from node.base import BaseNode
from node.request import InferControlLike, RequestNode
from operators.core.ptuning.common import PtuningConfig
from operators.core.sampler.common import SamplingParams
from operators.core.tokenizer.common import Sequence, SequenceCore, StopConfig
from operators.core.tokenizer.real import ExtendTokenizer
from operators.helpers.configs.quant_llama import LlamaConfig
from orbit import Satellite
from orbit.defines import Context
from orbit.launch_pad import LaunchPad
from orbit.link.extender import ContextNetExtender
from scripts.online import models
from scripts.online.benchmark.analysis_infer import Analyzer
from scripts.online.benchmark.launch_network import initialize_process
from scripts.online.benchmark.protocols import DecodeBenchmarkConfig
from scripts.online.runtime_config import API_DECODE_GRAPH_STATS


@dataclass(frozen=True)
class InferBenchmarkConfig:
    model_name: str
    prompt: str
    max_tokens: int
    warmup_tokens: int
    parallel_seqs: int
    num_paths: int
    allow_provider_reuse: bool
    retrieve: bool
    detail_trace: bool
    logprobs: Optional[int]
    progress_interval: float
    disable_gc: bool
    adapter: str
    adapter_name: str
    adapter_state: str
    adapter_prompt_template: str
    prefix_len: int
    lora_len_limit: int


@dataclass(frozen=True)
class FailureBenchmarkConfig:
    request: InferBenchmarkConfig
    probabilities: tuple[float, ...]
    repeats: int
    inspection_postpone: float


Measurement = Literal["request", "decode", "failure"]
BenchmarkConfig = Union[
    InferBenchmarkConfig,
    DecodeBenchmarkConfig,
    FailureBenchmarkConfig,
]


def gen_user_model(
        user_model_name: str,
        base_model_config: LlamaConfig,
        user_model_config: Union[LoraConfig, PtuningConfig, None],
        user_model_data_file: str
    ) -> Tuple[
    Optional[Tuple[
        str, Callable[[Tuple[int, int]], Tuple[
            List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]],
            Optional[LoraRenderConfig]
        ]]
    ]],
    Callable[[str], str]
]:
    if user_model_config is None:
        assert user_model_name == ""
        return None, lambda prompt=str: prompt
    assert user_model_name != ""
    state_file = user_model_data_file
    lora_cfg = None
    cfg = user_model_config
    is_lora = isinstance(cfg, LoraConfig)
    if is_lora:
        lora_cfg = cfg.render_config
        cfg.check_or_set_decoders_config(
            num_layers=base_model_config.num_decoders,
            head_dim=base_model_config.head_dim,
            num_key_value_heads=base_model_config.num_key_value_heads,
            num_query_heads=base_model_config.num_query_heads
        )
    else:
        cfg.check_or_set_decoders_config(
            num_layers=base_model_config.num_decoders,
            num_key_value_heads=base_model_config.num_key_value_heads,
            head_dim=base_model_config.head_dim
        )
    user_model = Lora(cfg) if is_lora else Ptuning(cfg)
    state_dict = torch.load(state_file, map_location="cpu")
    user_model.load_state_dict(state_dict)
    user_data =\
        cast(Lora, user_model).get_inference_lora_blocks(torch.float16) \
        if is_lora else \
        cast(Ptuning, user_model).get_inference_ptuning_kv_cache_with_convert(
            torch.float16)
    user_data = [
        [
            (DeviceTensorCarrier.from_data(k.cpu()),
             DeviceTensorCarrier.from_data(v.cpu()))
            for (k, v) in layer_blocks
        ]
        for layer_blocks in user_data
    ]

    def slice(range: Tuple[int, int]) ->\
        Tuple[
            List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]],
            Optional[LoraRenderConfig]
    ]:
        return user_data[range[0]:range[1]], lora_cfg
    return (user_model_name, slice), cfg.build_prompt


class InferInstances:
    def __init__(self,
                 instances: List[InferControlLike],
                 prompt_token_ids: List[List[int]],
                 with_detail_traces: bool,
                 ) -> None:
        self.instances = instances
        self.prompt_token_ids = prompt_token_ids
        self.with_detail_traces = with_detail_traces
        self.detail_traces_record: List[List[List[Tuple[str, int]]]] = [
            [] for _ in range(len(instances))]
        self.output_time_record: List[List[int]] = [[]
                                                    for _ in range(len(instances))]

    async def run_prefill(self):
        outs = await asyncio.gather(*[
            instance.infer_step(
                inp=prompt_token_ids.copy(),
                is_prefill=True,
                detail_trace=self.with_detail_traces
            )
            for instance, prompt_token_ids in zip(self.instances, self.prompt_token_ids)
        ])
        self._init_tokens = [token for _, token, _, _ in outs]
        return outs

    async def run_generating(self):
        queue: asyncio.Queue[Tuple[int, bool, int, Dict[int, float],
                                   Optional[List[Tuple[str, int]]]]] = asyncio.Queue()

        async def run_each(
            seq_i: int,
            instance: InferControlLike,
            next_token: int,
        ) -> None:
            stopped = False
            while not stopped:
                stopped, next_token, logprobs, detail_traces =\
                    await instance.infer_step(
                        inp=next_token,
                        is_prefill=False,
                        detail_trace=self.with_detail_traces
                    )
                queue.put_nowait(
                    (seq_i,  stopped, next_token, logprobs, detail_traces))
        running_tasks = [
            asyncio.create_task(run_each(i, instance, init_token))
            for i, (instance, init_token) in enumerate(zip(self.instances, self._init_tokens))
        ]
        unfinished_count = len(self.instances)
        while unfinished_count > 0:
            out = await queue.get()
            if out[1]:
                unfinished_count -= 1
            yield out
        del running_tasks


def record_time_to_current_speed(record_time: List[int]):
    if len(record_time) < 31:
        return 0
    dt = record_time[-1]-record_time[-31]
    return 30/(dt/(10**9))


async def run_benchmark_multi(
    tokenizer: ExtendTokenizer,
    create_instance: Callable[[str, List[int], int], InferControlLike],
    remove_instance: Callable[[InferControlLike], None],
    sample_string: str = "once upon a time",
    max_len: int = 20*1000,
    multi_task_count: int = 8,
    with_detail_trace: bool = False,
    warmup_tokens: int = 32,
    progress_interval: float = 0.0,
    disable_gc: bool = True,
):
    progress_interval_ns = int(progress_interval * 10**9)

    # do warm up
    print("warm up")
    prompt_tokens = tokenizer.encode(sample_string)
    instances = [
        create_instance(f"warmup-{i}#", prompt_tokens.copy(), warmup_tokens)
        for i in range(multi_task_count)
    ]
    multi_instances = InferInstances(instances, [prompt_tokens.copy() for _ in range(
        multi_task_count)], with_detail_traces=with_detail_trace)
    seqs = [
        Sequence(SequenceCore(StopConfig(max_tokens=warmup_tokens),
                 prompt_token_ids=prompt_tokens.copy()))
        for _ in range(multi_task_count)
    ]

    outs = await multi_instances.run_prefill()
    for seq_i, (_, token_id, _, trace) in enumerate(outs):
        seqs[seq_i].core.append_token_id(token_id)
        if with_detail_trace:
            assert trace is not None
    for seq in seqs:
        tokenizer.incrementally_decode_seq(seq)

    time_record: List[int] = []
    async for out in multi_instances.run_generating():
        seq_i, _, token_id, _, _ = out
        time_record.append(time.time_ns())
        seqs[seq_i].core.append_token_id(token_id)
        tokenizer.incrementally_decode_seq(seqs[seq_i])
        for seq in seqs:
            print(seq.output_text)
        print(
            f"TPS quick glance: total {record_time_to_current_speed(time_record):.2f}", end=f"\033[{multi_task_count}F")
    print("\n"*multi_task_count)

    for instance in instances:
        remove_instance(instance)
    await asyncio.sleep(0.2)

    print("benchmarking")
    detail_time_traces: List[List[List[Tuple[str, int]]]] = [
        [] for _ in range(multi_task_count)]
    instances = [
        create_instance(f"bench-{i}#", prompt_tokens.copy(), max_len)
        for i in range(multi_task_count)
    ]
    multi_instances = InferInstances(instances, [
        prompt_tokens.copy()
        for _ in range(multi_task_count)
    ], with_detail_traces=with_detail_trace)
    request_start_ns = time.time_ns()
    outs = await multi_instances.run_prefill()
    prefill_end_ns = time.time_ns()
    for seq_i, (_, _, _, trace) in enumerate(outs):
        if with_detail_trace and trace is not None:
            assert trace is not None
            detail_time_traces[seq_i].append(trace)

    time_record_multi: List[Tuple[int, int]] = []
    output_tokens: List[List[int]] = [[out[1]] for out in outs]
    first_token_ns: List[Optional[int]] = [
        prefill_end_ns for _ in range(multi_task_count)]
    t_window: List[int] = []

    gc_was_enabled = gc.isenabled()
    if disable_gc:
        gc.collect()
        gc.disable()
    st = time.time_ns()
    last_progress_time = st
    async for out in multi_instances.run_generating():
        seq_i = out[0]
        t = time.time_ns()
        output_tokens[seq_i].append(out[2])
        if first_token_ns[seq_i] is None:
            first_token_ns[seq_i] = t
        time_record_multi.append((seq_i, t))
        t_window.append(t)
        trace = out[-1]
        if with_detail_trace and trace is not None:
            detail_time_traces[seq_i].append(trace)
        if progress_interval_ns > 0 and len(t_window) > 8 and t - last_progress_time >= progress_interval_ns:
            print(
                f"\r{8/((t_window[-1]-t_window[0])/(10**9)):.2f}; {len(time_record_multi)/1000:.2f}K", end="")
            last_progress_time = t
        if len(t_window) > 8:
            t_window = t_window[1:]
    if progress_interval_ns > 0:
        print()
    benchmark_end = time.time_ns()
    if disable_gc and gc_was_enabled:
        gc.enable()
    for instance in instances:
        remove_instance(instance)
    await asyncio.sleep(0.2)

    print(f"{len(time_record_multi)/((benchmark_end-st)/1e9)}")

    analyzer = Analyzer()
    analyzer.add_record(f"Multi-{multi_task_count}", time_record_multi)
    analyzer.speed_info()
    phase_times: Dict[str, Any] = {
        "request_start_ns": request_start_ns,
        "prefill_end_ns": prefill_end_ns,
        "first_token_ns": first_token_ns,
        "end_ns": benchmark_end,
    }
    return time_record_multi, detail_time_traces, output_tokens, phase_times
    # with open(output_file, "w") as f:
    #     json.dump(time_record_multi, f)


def _user_model_config(
    config: InferBenchmarkConfig,
) -> Union[LoraConfig, PtuningConfig, None]:
    if config.adapter == "base":
        return None
    if config.adapter == "ptuning":
        return PtuningConfig(
            build_prompt_str=config.adapter_prompt_template,
            dropout_p=0.1,
            prefix_len=config.prefix_len,
        )
    if config.adapter == "lora":
        return LoraConfig(
            r_q=-1,
            r_k=0,
            r_v=-1,
            build_prompt_str=config.adapter_prompt_template,
            dropout_p=0.0,
            len_limit=config.lora_len_limit,
        )
    raise ValueError(config.adapter)


async def run_benchmark(
    request_node: RequestNode,
    config: InferBenchmarkConfig,
    output_dir: Path,
) -> None:
    model_info = models.models_storage[config.model_name]
    tokenizer = ExtendTokenizer.load_tokenier(
        path=model_info.tokenizer_path,
        cfg=model_info.config.tokenizer_special_tokens,
    )
    extra_info = await req_extra_info(request_node.base_node)
    if config.model_name not in extra_info.models:
        raise ValueError(
            f"model {config.model_name!r} is absent from network "
            f"{extra_info.network_name!r}"
        )
    base_models = extra_info.models[config.model_name].models
    inspected = await request_node.inspect_fastest_n_path(
        type="infer",
        base_models=base_models,
        allow_reuse=config.allow_provider_reuse,
        max_n=config.num_paths,
        providers=None,
    )
    if not inspected:
        raise RuntimeError("no complete inference path is available")
    if len(inspected.fastest_paths) != config.num_paths:
        raise RuntimeError(
            f"requested {config.num_paths} inference paths, "
            f"but only found {len(inspected.fastest_paths)}"
        )
    keypoints = inspected.keypoints
    print("keypoints: ", keypoints)

    user_model, build_prompt = gen_user_model(
        user_model_name=config.adapter_name,
        base_model_config=model_info.config,
        user_model_config=_user_model_config(config),
        user_model_data_file=config.adapter_state,
    )
    instance_scope = uuid4().hex[:8]

    def create_instance(name: str, prompt_ids: List[int], max_len: int):
        infer_ctrl = request_node.create_infer_control(
            seq_id=f"{name}{instance_scope}",
            base_models=base_models,
            seq_core=SequenceCore(
                stop_config=StopConfig(max_tokens=max_len, ignore_eos=True),
                prompt_token_ids=prompt_ids.copy(),
            ),
            sampling_params=SamplingParams(
                temperature=0.0,
                logprobs=config.logprobs,
            ),
            user_model_data=user_model,
            keypoint_providers=keypoints,
            retrieve=config.retrieve,
        )
        return infer_ctrl

    time_record_multi, detail_time_traces, output_tokens, phase_times = await run_benchmark_multi(
        tokenizer=tokenizer,
        create_instance=create_instance,
        remove_instance=lambda instance: request_node.remove_infer_control(
            instance.ctrl_id),
        sample_string=build_prompt(config.prompt),
        max_len=config.max_tokens,
        multi_task_count=config.parallel_seqs,
        with_detail_trace=config.detail_trace,
        warmup_tokens=config.warmup_tokens,
        progress_interval=config.progress_interval,
        disable_gc=config.disable_gc,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "result.json").open("w") as f:
        json.dump(time_record_multi, f)
    if len(detail_time_traces[0]) > 0:
        with (output_dir / "traces.json").open("w") as f:
            json.dump(detail_time_traces, f)
    with (output_dir / "tokens.json").open("w") as f:
        json.dump(output_tokens, f)
    phase_times["network_name"] = extra_info.network_name
    phase_times["network_spec_hash"] = extra_info.network_spec_hash
    phase_times["keypoints"] = keypoints
    phase_times["deployment"] = (
        "persistent-agent"
        if request_node.is_remote
        else "combined-request-agent"
    )
    parameters = asdict(config)
    parameters["logprobs"] = 0 if config.logprobs is None else config.logprobs
    phase_times["parameters"] = parameters
    with (output_dir / "phases.json").open("w") as f:
        json.dump(phase_times, f)
    graph_stats = {
        dealer_name: await request_node.base_node.rpc_no_args(
            node=dealer_name,
            api=API_DECODE_GRAPH_STATS,
            T=dict,
        )
        for dealer_name in sorted({keypoint[0] for keypoint in keypoints})
    }
    with (output_dir / "decode_graph.json").open("w") as f:
        json.dump(graph_stats, f, indent=2)
    print("decode graph:", graph_stats)
    print("to quick review result, use the following cmd:")
    print(f"python -m scripts.online.benchmark.analysis_infer {output_dir}")


async def _decode_prefill(
    instances: List[InferControlLike],
    prompt_ids: List[int],
) -> List[int]:
    outputs = await asyncio.gather(
        *(
            instance.infer_step(
                inp=prompt_ids.copy(),
                is_prefill=True,
                detail_trace=False,
            )
            for instance in instances
        )
    )
    return [int(output[1]) for output in outputs]


async def _decode_round(
    instances: List[InferControlLike],
    next_tokens: List[int],
) -> Tuple[List[int], List[bool]]:
    outputs = await asyncio.gather(
        *(
            instance.infer_step(
                inp=token,
                is_prefill=False,
                detail_trace=False,
            )
            for instance, token in zip(
                instances,
                next_tokens,
                strict=True,
            )
        )
    )
    return (
        [int(output[1]) for output in outputs],
        [bool(output[0]) for output in outputs],
    )


async def _decode_rounds(
    instances: List[InferControlLike],
    next_tokens: List[int],
    rounds: int,
) -> Tuple[List[int], List[List[int]]]:
    token_ids = [[] for _ in instances]
    for round_index in range(rounds):
        next_tokens, stopped = await _decode_round(instances, next_tokens)
        for sequence_index, token in enumerate(next_tokens):
            token_ids[sequence_index].append(token)
        if any(stopped) and round_index != rounds - 1:
            raise RuntimeError(
                "online request stopped before decode phase ended")
    return next_tokens, token_ids


async def collect_decode_benchmark(
    request_node: RequestNode,
    config: DecodeBenchmarkConfig,
) -> Dict[str, Any]:
    model_info = models.models_storage[config.model_name]
    tokenizer = ExtendTokenizer.load_tokenier(
        path=model_info.tokenizer_path,
        cfg=model_info.config.tokenizer_special_tokens,
    )
    prompt_ids = tokenizer.encode(config.prompt)
    request_tokens = (
        1 + config.warmup_decode_tokens + config.decode_tokens
    )
    if len(prompt_ids) + request_tokens > config.max_model_len:
        raise ValueError(
            "prompt plus generated tokens exceeds max model length")

    extra_info = await req_extra_info(request_node.base_node)
    if config.model_name not in extra_info.models:
        raise ValueError(
            f"model {config.model_name!r} is absent from network "
            f"{extra_info.network_name!r}"
        )
    base_models = extra_info.models[config.model_name].models
    inspected = await request_node.inspect_fastest_n_path(
        type="infer",
        base_models=base_models,
        allow_reuse=False,
        max_n=1,
        providers=None,
    )
    if not inspected:
        raise RuntimeError("no online inference path is available")
    keypoints = inspected.keypoints
    dealer_names = sorted({keypoint[0] for keypoint in keypoints})
    if (
        config.expected_dealers is not None
        and len(dealer_names) != config.expected_dealers
    ):
        raise RuntimeError(
            "selected online path has an unexpected dealer count: "
            f"expected={config.expected_dealers}, actual={dealer_names}"
        )

    instance_counter = 0

    def create_instances(num_seqs: int) -> List[InferControlLike]:
        nonlocal instance_counter
        instances = []
        for _ in range(num_seqs):
            name = f"decode-benchmark-{instance_counter}"
            instance_counter += 1
            instances.append(
                request_node.create_infer_control(
                    seq_id=name,
                    base_models=base_models,
                    seq_core=SequenceCore(
                        stop_config=StopConfig(
                            max_tokens=request_tokens,
                            ignore_eos=True,
                        ),
                        prompt_token_ids=prompt_ids.copy(),
                    ),
                    sampling_params=SamplingParams(
                        temperature=0.0,
                        logprobs=None,
                    ),
                    user_model_data=None,
                    keypoint_providers=keypoints,
                    retrieve=False,
                )
            )
        return instances

    async def run_request(
        num_seqs: int,
        measured: bool,
    ) -> Dict[str, Any]:
        instances = create_instances(num_seqs)
        try:
            next_tokens = await _decode_prefill(instances, prompt_ids)
            next_tokens, _ = await _decode_rounds(
                instances,
                next_tokens,
                config.warmup_decode_tokens,
            )
            if not measured:
                await _decode_rounds(
                    instances,
                    next_tokens,
                    config.decode_tokens,
                )
                return {}

            gc_was_enabled = gc.isenabled()
            if gc_was_enabled:
                gc.collect()
                gc.disable()
            started = time.perf_counter()
            try:
                _, output_tokens = await _decode_rounds(
                    instances,
                    next_tokens,
                    config.decode_tokens,
                )
            finally:
                seconds = time.perf_counter() - started
                if gc_was_enabled:
                    gc.enable()
            measured_tokens = num_seqs * config.decode_tokens
            return {
                "tokens": measured_tokens,
                "logical_rounds": config.decode_tokens,
                "seconds": seconds,
                "tok_s": measured_tokens / seconds,
                "avg_ms_per_logical_round": (
                    seconds * 1000 / config.decode_tokens
                ),
                "output_token_ids_sha256": [
                    hashlib.sha256(
                        json.dumps(
                            tokens,
                            separators=(",", ":"),
                        ).encode("ascii")
                    ).hexdigest()
                    for tokens in output_tokens
                ],
            }
        finally:
            for instance in instances:
                request_node.remove_infer_control(instance.ctrl_id)
            await asyncio.sleep(0.2)

    cases = []
    for num_seqs in config.num_seqs:
        for _ in range(config.warmup_requests):
            await run_request(num_seqs, measured=False)
        runs = []
        for run_index in range(config.runs):
            result = await run_request(num_seqs, measured=True)
            result["run_index"] = run_index
            runs.append(result)
        rates = [run["tok_s"] for run in runs]
        round_times = [
            run["avg_ms_per_logical_round"] for run in runs
        ]
        cases.append(
            {
                "num_seqs": num_seqs,
                "prompt_tokens_per_seq": len(prompt_ids),
                "runs": runs,
                "summary": {
                    "decode_tok_s_median": statistics.median(rates),
                    "decode_tok_s_min": min(rates),
                    "decode_tok_s_max": max(rates),
                    "decode_round_ms_median": statistics.median(
                        round_times
                    ),
                },
            }
        )

    graph_stats = {}
    for dealer_name in dealer_names:
        if not await request_node.base_node.try_connect(dealer_name):
            raise ConnectionError(
                f"cannot connect Request node to Dealer {dealer_name!r}"
            )
        graph_stats[dealer_name] = await request_node.base_node.rpc_no_args(
            node=dealer_name,
            api=API_DECODE_GRAPH_STATS,
            T=dict,
        )
    graph_states = {
        bool(component["enabled"])
        for stats in graph_stats.values()
        for component in (stats, stats["head"], stats["tail"])
    }
    execution = (
        "graph"
        if graph_states == {True}
        else "eager"
        if graph_states == {False}
        else "mixed"
    )
    report = {
        "schema_version": 1,
        "engine": "drofell-online",
        "model": config.model_name,
        "execution": execution,
        "deployment": (
            "request-agent"
            if request_node.is_remote
            else "combined-request-agent"
        ),
        "measurement": {
            "mode": "decode",
            "runs": config.runs,
            "warmup_requests": config.warmup_requests,
            "warmup_decode_tokens_per_seq": config.warmup_decode_tokens,
            "decode_tokens_per_seq": config.decode_tokens,
            "timing_scope": "post_prefill_request_observed_decode",
        },
        "topology": {
            "network_name": extra_info.network_name,
            "network_spec_hash": extra_info.network_spec_hash,
            "expected_dealers": config.expected_dealers,
            "selected_dealers": dealer_names,
            "keypoints": keypoints,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cases": cases,
        "decode_graph": graph_stats,
    }
    return report


def write_decode_report(
    report: Dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


async def run_decode_benchmark(
    request_node: RequestNode,
    config: DecodeBenchmarkConfig,
    output_dir: Path,
) -> None:
    report = await collect_decode_benchmark(request_node, config)
    report["topology"]["agent"] = (
        request_node.remote_agent_node if request_node.is_remote else request_node.base_node.node_name
    )
    report["topology"]["request"] = request_node.base_node.node_name
    write_decode_report(report, output_dir)


async def reset_drop_probability(
    request_node: RequestNode,
    probability: float,
) -> None:
    providers = await req_providers(request_node.base_node, None)
    dealers = sorted(
        {dealer for _, entries in providers for dealer, _ in entries}
    )
    capable = []
    for dealer in dealers:
        tags = await request_node.base_node.req_tags(dealer)
        if "infer_drop_reset" not in tags:
            continue
        capable.append(dealer)
        await request_node.base_node.rpc_with_args(
            node=dealer,
            api="infer_drop_reset",
            args=probability,
            T=float,
        )
    if not capable:
        raise RuntimeError(
            "network has no failure-enabled dealer; regenerate it with "
            "--failure-injection"
        )


async def run_failure_benchmark(
    request_node: RequestNode,
    config: FailureBenchmarkConfig,
    output_root: Path,
) -> None:
    async def postpone(
        kind: Literal[
            "infer", "train", "backbone.infer", "backbone.train"
        ],
    ) -> None:
        await asyncio.sleep(config.inspection_postpone)

    flags.InspectPostpone = postpone
    try:
        for repeat in range(config.repeats):
            for probability_value in config.probabilities:
                await reset_drop_probability(request_node, probability_value)
                label = format(probability_value, ".8g").replace(".", "p")
                destination = (
                    output_root
                    / f"drop-{label}"
                    / f"repeat-{repeat}"
                )
                print(
                    f"failure benchmark: p={probability_value}, "
                    f"repeat={repeat}, output={destination}"
                )
                await run_benchmark(
                    request_node,
                    config.request,
                    destination,
                )
    finally:
        await reset_drop_probability(request_node, 0.0)


def benchmark_node(ctx: Context) -> None:
    async def coro(
        ctx: Context,
        measurement: Measurement,
        config: BenchmarkConfig,
        output_dir: Path,
        agent_name: Optional[str],
        path_search_redundancy: int,
    ) -> None:
        ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        request_node = RequestNode(
            base_node,
            agent_node=agent_name,
            path_search_redundancy=path_search_redundancy,
        )
        ctx.report_ready()

        async def benchmark_and_stop() -> None:
            if measurement == "request":
                assert isinstance(config, InferBenchmarkConfig)
                await run_benchmark(request_node, config, output_dir)
            elif measurement == "decode":
                assert isinstance(config, DecodeBenchmarkConfig)
                await run_decode_benchmark(request_node, config, output_dir)
            else:
                assert isinstance(config, FailureBenchmarkConfig)
                await run_failure_benchmark(
                    request_node,
                    config,
                    output_dir,
                )
            await asyncio.sleep(0.3)
            raise SystemExit(0)

        asyncio.create_task(benchmark_and_stop())

        while True:
            packet = await ctx.recv()
            base_node.on_packet(packet)
    asyncio.get_event_loop().create_task(
        coro(ctx, *ctx.args.args, **ctx.args.kwargs)
    )


def run_satellite(
    role: str,
    orbit_addr: str,
    advertise_host: str,
    measurement: Measurement,
    config: BenchmarkConfig,
    output_dir: Path,
    async_backend: str,
    busy_wait: bool,
    agent_name: Optional[str],
    path_search_redundancy: int,
) -> None:
    launch_pad = LaunchPad(
        partial(initialize_process, async_backend, busy_wait)
    )

    async def coro():
        satellite = Satellite(
            launch_pad=launch_pad,
            functions_dict={},
            host_id=role,
            network_hostname=advertise_host,
            stub_remote_log=True,
        )
        satellite.start(orbit_addr)
        await satellite.ready()
        satellite.launch(
            benchmark_node,
            node_name=role,
            measurement=measurement,
            config=config,
            output_dir=output_dir,
            agent_name=agent_name,
            path_search_redundancy=path_search_redundancy,
        )
        await asyncio.Future()

    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def probability(value: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("probability must be in [0, 1]")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an online inference benchmark")
    parser.add_argument(
        "--measurement",
        choices=("request", "decode", "failure"),
        default="request",
    )
    parser.add_argument("--orbit", required=True, metavar="HOST:PORT")
    parser.add_argument(
        "--agent",
        help=(
            "Persistent benchmark Agent node. Inference control runs in this "
            "node; omit to combine Request and Agent in the temporary node. "
            "Adapter inference is allowed, but PEFT conditions may make CUDA "
            "graph decode fall back to eager execution."
        ),
    )
    parser.add_argument("--host-id", default=f"benchmark-infer-{os.getpid()}")
    parser.add_argument("--advertise-host", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="once upon a time")
    parser.add_argument("--max-tokens", type=positive_int)
    parser.add_argument("--warmup-tokens", type=positive_int)
    parser.add_argument("--seqs", type=positive_int)
    parser.add_argument("--num-seqs", nargs="+", type=positive_int)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--warmup-decode-tokens", type=int, default=8)
    parser.add_argument("--decode-tokens", type=positive_int, default=200)
    parser.add_argument("--max-model-len", type=positive_int, default=256)
    parser.add_argument("--expected-dealers", type=positive_int)
    parser.add_argument(
        "--drop-probabilities",
        nargs="+",
        type=probability,
    )
    parser.add_argument("--repeats", type=positive_int)
    parser.add_argument("--inspection-postpone", type=float, default=0.3)
    parser.add_argument("--num-paths", type=positive_int, default=1)
    parser.add_argument(
        "--path-search-redundancy",
        type=positive_int,
        default=1,
        help=(
            "Agent provider-search redundancy; each model segment considers "
            "at most this value plus one distinct providers."
        ),
    )
    parser.add_argument("--allow-provider-reuse", action="store_true")
    parser.add_argument("--retrieve", action="store_true")
    parser.add_argument("--detail-trace", action="store_true")
    parser.add_argument(
        "--logprobs",
        type=int,
        default=1,
        help="Number of log probabilities; 0 disables collection.",
    )
    parser.add_argument("--progress-interval", type=float, default=0.0)
    parser.add_argument(
        "--disable-gc",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--adapter", choices=("base", "ptuning", "lora"), default="base"
    )
    parser.add_argument("--adapter-name", default="")
    parser.add_argument("--adapter-state", default="")
    parser.add_argument(
        "--adapter-prompt-template",
        default="[instruction]{}\n[generation]",
    )
    parser.add_argument("--prefix-len", type=positive_int, default=128)
    parser.add_argument("--lora-len-limit", type=positive_int, default=128)
    parser.add_argument(
        "--async-backend",
        choices=("py", "chimera", "golang"),
        default="golang",
    )
    parser.add_argument(
        "--busy-wait",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.logprobs < 0:
        parser.error("--logprobs must be non-negative")
    if args.progress_interval < 0:
        parser.error("--progress-interval must be non-negative")
    if args.inspection_postpone < 0:
        parser.error("--inspection-postpone must be non-negative")
    if (
        args.drop_probabilities is not None
        and len(set(args.drop_probabilities)) != len(args.drop_probabilities)
    ):
        parser.error("--drop-probabilities must be unique")
    if args.warmup_requests < 0 or args.warmup_decode_tokens < 0:
        parser.error("decode warmup counts must be non-negative")
    if args.measurement in {"request", "failure"}:
        if args.max_tokens is None or args.seqs is None:
            parser.error(
                f"{args.measurement} mode requires --max-tokens and --seqs"
            )
        if args.num_seqs is not None:
            parser.error(
                f"{args.measurement} mode uses --seqs, not --num-seqs"
            )
        if args.warmup_tokens is None:
            args.warmup_tokens = 32
        if args.adapter == "base":
            if args.adapter_name or args.adapter_state:
                parser.error(
                    "base adapter cannot use --adapter-name/--adapter-state"
                )
        elif not args.adapter_name or not args.adapter_state:
            parser.error(
                "ptuning/lora requires --adapter-name and --adapter-state"
            )
        if args.measurement == "failure":
            if args.drop_probabilities is None or args.repeats is None:
                parser.error(
                    "failure mode requires --drop-probabilities and --repeats"
                )
            if args.allow_provider_reuse:
                parser.error("failure mode does not support provider reuse")
            if args.progress_interval != 0.0:
                parser.error("failure mode requires --progress-interval 0")
            if not args.disable_gc:
                parser.error(
                    "failure mode requires garbage collection disabled")
            if args.adapter != "base":
                parser.error("failure mode only supports the base model")
        elif args.drop_probabilities is not None or args.repeats is not None:
            parser.error(
                "request mode does not use failure-matrix arguments"
            )
    else:
        if args.num_seqs is None:
            parser.error("decode mode requires --num-seqs")
        if len(set(args.num_seqs)) != len(args.num_seqs):
            parser.error("--num-seqs values must be unique")
        if args.max_tokens is not None or args.seqs is not None:
            parser.error("decode mode uses --num-seqs and --decode-tokens")
        if args.warmup_tokens is not None:
            parser.error(
                "decode mode uses --warmup-decode-tokens, not --warmup-tokens"
            )
        if args.adapter != "base" or args.adapter_name or args.adapter_state:
            parser.error("decode mode does not support adapters")
        if args.retrieve:
            parser.error("decode mode requires retrieval to remain disabled")
        if args.detail_trace:
            parser.error("decode mode does not collect detail traces")
        if args.drop_probabilities is not None or args.repeats is not None:
            parser.error("decode mode does not use failure-matrix arguments")
    return args


def main() -> None:
    args = parse_args()
    measurement = cast(Measurement, args.measurement)
    if measurement == "request":
        config: BenchmarkConfig = InferBenchmarkConfig(
            model_name=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            warmup_tokens=args.warmup_tokens,
            parallel_seqs=args.seqs,
            num_paths=args.num_paths,
            allow_provider_reuse=args.allow_provider_reuse,
            retrieve=args.retrieve,
            detail_trace=args.detail_trace,
            logprobs=None if args.logprobs == 0 else args.logprobs,
            progress_interval=args.progress_interval,
            disable_gc=args.disable_gc,
            adapter=args.adapter,
            adapter_name=args.adapter_name,
            adapter_state=args.adapter_state,
            adapter_prompt_template=args.adapter_prompt_template,
            prefix_len=args.prefix_len,
            lora_len_limit=args.lora_len_limit,
        )
    elif measurement == "decode":
        config = DecodeBenchmarkConfig(
            model_name=args.model,
            prompt=args.prompt,
            num_seqs=tuple(args.num_seqs),
            runs=args.runs,
            warmup_requests=args.warmup_requests,
            warmup_decode_tokens=args.warmup_decode_tokens,
            decode_tokens=args.decode_tokens,
            max_model_len=args.max_model_len,
            expected_dealers=args.expected_dealers,
        )
    else:
        config = FailureBenchmarkConfig(
            request=InferBenchmarkConfig(
                model_name=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                warmup_tokens=args.warmup_tokens,
                parallel_seqs=args.seqs,
                num_paths=args.num_paths,
                allow_provider_reuse=False,
                retrieve=args.retrieve,
                detail_trace=args.detail_trace,
                logprobs=None if args.logprobs == 0 else args.logprobs,
                progress_interval=0.0,
                disable_gc=True,
                adapter="base",
                adapter_name="",
                adapter_state="",
                adapter_prompt_template="[instruction]{}\n[generation]",
                prefix_len=128,
                lora_len_limit=128,
            ),
            probabilities=tuple(args.drop_probabilities),
            repeats=args.repeats,
            inspection_postpone=args.inspection_postpone,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "args.json").open("w") as file:
        json.dump(
            {
                "measurement": measurement,
                "orbit": args.orbit,
                "agent": args.agent,
                "host_id": args.host_id,
                "advertise_host": args.advertise_host,
                "async_backend": args.async_backend,
                "busy_wait": args.busy_wait,
                "path_search_redundancy": args.path_search_redundancy,
                "benchmark": asdict(config),
            },
            file,
            indent=2,
        )
    run_satellite(
        role=args.host_id,
        orbit_addr=args.orbit,
        advertise_host=args.advertise_host,
        measurement=measurement,
        config=config,
        output_dir=args.output,
        async_backend=args.async_backend,
        busy_wait=args.busy_wait,
        agent_name=args.agent,
        path_search_redundancy=args.path_search_redundancy,
    )


if __name__ == "__main__":
    main()
