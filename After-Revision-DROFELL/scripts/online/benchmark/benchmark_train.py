import argparse
from dataclasses import asdict, dataclass
from functools import partial
import json
import os
from pathlib import Path
import random
import time
from asyncio import Task
from typing import Any, Callable, Dict, List, Sequence, Tuple

import cache
import data
import extend_asyncio as asyncio
from dataset.advertise_gen import make_dataset_adv_gen
from dataset.common import ProcessedRecord
from dealers.common.brain import OptimizerConfig, UserModelDataInitConfig
from node.aux_node.protocols import req_extra_info, req_providers
from node.base import BaseNode
from node.compute_node.train_meta import TrainHeadInput
from node.path_utils import paths_to_keypoints
from node.request import RequestNode, TrainControlLike
from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig
from orbit import Satellite
from orbit.defines import Context
from orbit.launch_pad import LaunchPad
from orbit.link.extender import ContextNetExtender
from scripts.online import models
from scripts.online.benchmark.launch_network import initialize_process
from scripts.base.train_state_maintainer import StepSize, TrainStatusMaintainer
from utils.minibatch_feeder import AsyncMinibatchFeeder


@dataclass(frozen=True)
class TrainBenchmarkConfig:
    model_name: str
    samples_per_instance: int
    parallel_instances: int
    minibatch_size: int
    grad_accum_steps: int
    sample_len: int
    bf16: bool
    detail_trace: bool
    ckpt_period: int
    adapter: str
    adapter_prompt_template: str
    adapter_dropout: float
    ptuning_prefix_len: int
    lora_r_q: int
    lora_r_k: int
    lora_r_v: int
    lora_len_limit: int
    optimizer: str
    learning_rate: float
    optimizer_eps: float
    submission_interval_ms: float = 0.0
    dataset: str = "synthetic"
    dataset_seed: int = 0
    dataset_query_tokens: int = 32
    dataset_response_tokens: int = 32
    num_paths: int = 1
    allow_provider_reuse: bool = False


async def wait_first_ready(
    tasks: Tuple[Task[None], ...],
) -> tuple[set[Task[None]], set[Task[None]]]:
    done = {task for task in tasks if task.done()}
    if done:
        return done, set(tasks) - done
    completed, pending = await asyncio.wait_first(tasks)
    return set(completed), set(pending)


async def run_benchmark_multi(
    create_instance: Callable[[str], TrainControlLike],
    remove_instance: Callable[[TrainControlLike], None],
    samples_each_instance: int = 100,
    parallel_instances: int = 2,
    batch_size: int = 2,
    minibatch_size: int = 2,
    sample_len: int = 128,
    is_bfloat_16: bool = False,
    with_detail_trace: bool = False,
    submission_interval_ms: float = 0.0,
    sample_records: Sequence[ProcessedRecord] | None = None,
    dataset_seed: int = 0,
):

    global_train_status_maintainer = TrainStatusMaintainer(
        step_size=StepSize(
            steps_in_each_update=batch_size,
            step_batch_size=minibatch_size
        ),
        total_samples=samples_each_instance*parallel_instances,
        events=None,
        prefix="total"
    )

    detail_time_traces: Dict[str, List[List[Tuple[str, int]]]] = {}

    async def instance(name: str, instance_i: int):
        total_samples = 0
        real_sample_indices: list[int] = []
        if sample_records is not None:
            if samples_each_instance > len(sample_records):
                raise ValueError(
                    f"requested {samples_each_instance} samples from a "
                    f"dataset containing {len(sample_records)} records"
                )
            real_sample_indices = list(range(len(sample_records)))
            random.Random(dataset_seed + instance_i).shuffle(real_sample_indices)
            real_sample_indices = real_sample_indices[:samples_each_instance]

        def gen_sample():
            nonlocal total_samples
            if total_samples >= samples_each_instance:
                return None

            records: List[ProcessedRecord] = []
            for _ in range(minibatch_size):
                if sample_records is None:
                    record = ProcessedRecord(
                        input_ids=[3 for _ in range(sample_len)],
                        output_ids=[3 for _ in range(sample_len)]
                    )
                else:
                    record = sample_records[real_sample_indices[total_samples]]
                records.append(record)
                total_samples += 1
                if total_samples >= samples_each_instance:
                    break
            return records

        feeder = AsyncMinibatchFeeder(
            get_sample=gen_sample,
            batch_size=batch_size,
            max_concurrent=batch_size*2
        )

        train_status_maintainer = TrainStatusMaintainer(
            step_size=StepSize(
                steps_in_each_update=batch_size,
                step_batch_size=minibatch_size
            ),
            total_samples=samples_each_instance,
            events=None,
            prefix=name
        )
        train_ctrl = create_instance(name)

        async def do_one_batch(sample: Tuple[List[str], int, str, List[ProcessedRecord]]):
            batch_ids, req_version, minibatch_id, records = sample
            _input = TrainHeadInput.from_records(records, is_bfloat16=is_bfloat_16)
            start_time = time.time()
            ret = await train_ctrl.train_step_with_detail(
                req_version=req_version,
                full_batch_ids=batch_ids,
                minibatch_id=minibatch_id,
                sample=_input,
                detail_trace=with_detail_trace
            )
            assert ret.task.loss is not None
            loss = ret.task.loss
            trace = ret.detail_trace
            if with_detail_trace and trace is not None:
                minibatch_i = int(minibatch_id.split("-")[1]) % batch_size
                detail_time_traces.setdefault(f"{instance_i}-{minibatch_i}", []).append(trace)

            total_tokens = sum(len(r.input_ids) for r in records)
            print(train_status_maintainer.update_loss(loss, total_tokens, start_time, minibatch_id))
            print(global_train_status_maintainer.update_loss(loss, total_tokens, start_time, minibatch_id))

        if submission_interval_ms > 0:
            # Open-loop submission is required by recovery experiments: task
            # creation must not be gated on completion of an earlier task.
            # This models the minibatches inside a gradient-accumulation
            # batch that the Agent may dispatch independently.
            submitted: List[Task[None]] = []
            while True:
                sample = await feeder.get_next(wait=False)
                if sample is None:
                    break
                submitted.append(asyncio.create_task(do_one_batch(sample)))
                await asyncio.sleep(submission_interval_ms / 1000.0)
            await asyncio.gather(*submitted)
            remove_instance(train_ctrl)
            return train_status_maintainer.dump_info()

        running_tasks: Tuple[Task[None], ...]
        _running_tasks: List[Task[None]] = []
        for _ in range(2):
            sample = await feeder.get_next(wait=False)
            if sample is None:
                raise RuntimeError(
                    "training feeder ended during initial submission"
                )
            t = asyncio.create_task(do_one_batch(sample))
            _running_tasks.append(t)

        running_tasks = tuple(_running_tasks)

        while True:
            sample = await feeder.get_next(wait=False)
            if sample is None:
                break
            task: Task[None] = asyncio.create_task(do_one_batch(sample))
            running_tasks = (*running_tasks, task)
            done, pending = await wait_first_ready(running_tasks)
            running_tasks = tuple(pending)
        while running_tasks:
            done, pending = await wait_first_ready(running_tasks)
            running_tasks = tuple(pending)
        remove_instance(train_ctrl)
        return train_status_maintainer.dump_info()
    results = await asyncio.gather(*[
        instance(f"train-{instance_i}", instance_i)
        for instance_i in range(parallel_instances)
    ])
    results.append(global_train_status_maintainer.dump_info())
    return results, detail_time_traces


def _adapter_config(config: TrainBenchmarkConfig):
    if config.adapter == "ptuning":
        return PtuningConfig(
            build_prompt_str=config.adapter_prompt_template,
            dropout_p=config.adapter_dropout,
            prefix_len=config.ptuning_prefix_len,
        )
    if config.adapter == "lora":
        return LoraConfig(
            r_q=config.lora_r_q,
            r_k=config.lora_r_k,
            r_v=config.lora_r_v,
            build_prompt_str=config.adapter_prompt_template,
            dropout_p=config.adapter_dropout,
            len_limit=config.lora_len_limit,
        )
    raise ValueError(config.adapter)


async def run_benchmark(
    request_node: RequestNode,
    config: TrainBenchmarkConfig,
    output_dir: Path,
) -> None:
    extra_info = await req_extra_info(request_node.base_node)
    if config.model_name not in extra_info.models:
        raise ValueError(
            f"model {config.model_name!r} is absent from network "
            f"{extra_info.network_name!r}"
        )
    base_models = extra_info.models[config.model_name].models
    inspected = await request_node.inspect_fastest_n_path(
        "train",
        base_models,
        allow_reuse=config.allow_provider_reuse,
        max_n=config.num_paths,
        providers=None,
    )
    if not inspected:
        raise RuntimeError("no complete training path is available")
    if len(inspected.fastest_paths) != config.num_paths:
        raise RuntimeError(
            f"requested {config.num_paths} training paths, "
            f"but only {len(inspected.fastest_paths)} are available"
        )
    keypoints = inspected.keypoints
    print("paths: ", inspected.fastest_paths)
    print("keypoints: ", keypoints)

    sample_records: Sequence[ProcessedRecord] | None = None
    if config.dataset == "advertise-gen":
        model_info = models.models_storage[config.model_name]
        sample_records = make_dataset_adv_gen(
            json_file=data.advertise_gen_train_path,
            build_prompt_str=config.adapter_prompt_template,
            tokenizer_path=model_info.tokenizer_path,
            eos_token_id=model_info.config.tokenizer_special_tokens.eos_token_id,
            max_query_tokens=-1,
            max_response_tokens=-1,
            cache_dir=cache.dataset_cache_path,
            force_query_tokens=config.dataset_query_tokens,
            force_response_tokens=config.dataset_response_tokens,
        ).records
        print(
            f"loaded {len(sample_records)} real AdvertiseGen records; "
            f"deterministic sample seed={config.dataset_seed}"
        )
    elif config.dataset != "synthetic":
        raise ValueError(f"unsupported training dataset: {config.dataset}")

    dealer_nodes = {
        provider
        for path in inspected.fastest_paths
        for provider in path
    }
    for dealer_node in dealer_nodes:
        tags = await request_node.base_node.req_tags(dealer_node)
        print(f"{dealer_node} {tags}")
        if tags and 'time_trace' in tags:
            await request_node.base_node.rpc_no_args(
                dealer_node, "telemetric-time-start", int
            )

    def create_instance(name: str):
        train_ctrl = request_node.create_train_control(
            user_model_name=name,
            base_models=base_models,
            user_model_config=_adapter_config(config),
            optimizer_config=OptimizerConfig(
                optimizer_name=config.optimizer,
                lr=config.learning_rate,
                eps=config.optimizer_eps,
            ),
            scheduler_config=None,
            user_model_init_config=UserModelDataInitConfig(
                init_method="reinit",
                ckpt_bytes=None,
            ),
            keypoint_providers=keypoints,
            ckpt_period=config.ckpt_period,
        )
        return train_ctrl

    time_record_multi, detail_time_traces = await run_benchmark_multi(
        create_instance=create_instance,
        remove_instance=lambda instance: request_node.remove_train_control(instance.ctrl_id),
        samples_each_instance=config.samples_per_instance,
        parallel_instances=config.parallel_instances,
        batch_size=config.grad_accum_steps,
        minibatch_size=config.minibatch_size,
        sample_len=config.sample_len,
        is_bfloat_16=config.bf16,
        with_detail_trace=config.detail_trace,
        submission_interval_ms=config.submission_interval_ms,
        sample_records=sample_records,
        dataset_seed=config.dataset_seed,
    )

    dealer_time_traces: Dict[str, Any] = {}
    for dealer_node in dealer_nodes:
        tags = await request_node.base_node.req_tags(dealer_node)
        print(f"{dealer_node} {tags}")
        if tags and 'time_trace' in tags:
            dealer_time_trace = await request_node.base_node.rpc_no_args(
                dealer_node, "telemetric-time-end", list
            )
            dealer_time_traces[dealer_node] = dealer_time_trace

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w") as f:
        json.dump(time_record_multi, f)
    if len(detail_time_traces) > 0:
        with (output_dir / "traces.json").open("w") as f:
            json.dump(detail_time_traces, f)
    if len(dealer_time_traces) > 0:
        with (output_dir / "brains.json").open("w") as f:
            json.dump(dealer_time_traces, f)
    with (output_dir / "network.json").open("w") as f:
        json.dump(
            {
                "name": extra_info.network_name,
                "spec_sha256": extra_info.network_spec_hash,
                "selected_paths": inspected.fastest_paths,
                "keypoints": keypoints,
            },
            f,
            indent=2,
        )
    print("to quick review result, use the following cmd:")
    print(f"python -m scripts.online.benchmark.analysis_train {output_dir}")


def benchmark_node(ctx: Context):
    async def coro(
        ctx: Context,
        config: TrainBenchmarkConfig,
        output_dir: Path,
        agent_node: str | None,
    ):
        ctx.report_ready()
        net_extend = ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        request = RequestNode(base_node, agent_node=agent_node)
        async def benchmark_and_stop() -> None:
            await run_benchmark(request, config, output_dir)
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
    config: TrainBenchmarkConfig,
    output_dir: Path,
    agent_node: str | None,
    async_backend: str,
    busy_wait: bool,
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
        satellite.start(orbit_addr=orbit_addr)
        await satellite.ready()
        satellite.launch(
            benchmark_node,
            node_name=role,
            config=config,
            output_dir=output_dir,
            agent_node=agent_node,
        )
        await asyncio.Future()

    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an online training benchmark")
    parser.add_argument("--orbit", required=True, metavar="HOST:PORT")
    parser.add_argument("--host-id", default=f"benchmark-train-{os.getpid()}")
    parser.add_argument("--advertise-host", default="")
    parser.add_argument(
        "--agent",
        help=(
            "Persistent Agent node. Omit to construct Agent inside the "
            "Request node."
        ),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=positive_int, required=True)
    parser.add_argument(
        "--samples-per-instance", type=positive_int, required=True
    )
    parser.add_argument("--minibatch-size", type=positive_int, required=True)
    parser.add_argument("--grad-accum-steps", type=positive_int, required=True)
    parser.add_argument("--sample-len", type=positive_int, default=128)
    parser.add_argument(
        "--dataset", choices=("synthetic", "advertise-gen"),
        default="synthetic",
    )
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--dataset-query-tokens", type=int, default=32)
    parser.add_argument("--dataset-response-tokens", type=int, default=32)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--detail-trace", action="store_true")
    parser.add_argument("--num-paths", type=positive_int, default=1)
    parser.add_argument("--allow-provider-reuse", action="store_true")
    parser.add_argument("--ckpt-period", type=positive_int, default=16)
    parser.add_argument(
        "--adapter", choices=("ptuning", "lora"), required=True
    )
    parser.add_argument(
        "--adapter-prompt-template",
        default="[instruction]{}\n[generation]",
    )
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--ptuning-prefix-len", type=positive_int, default=128)
    parser.add_argument("--lora-r-q", type=int, default=-1)
    parser.add_argument("--lora-r-k", type=int, default=0)
    parser.add_argument("--lora-r-v", type=int, default=-1)
    parser.add_argument("--lora-len-limit", type=positive_int, default=128)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--optimizer-eps", type=float, default=1e-4)
    parser.add_argument(
        "--submission-interval-ms",
        type=float,
        default=0.0,
        help=(
            "Submit minibatches open-loop at this interval; zero preserves "
            "the completion-gated benchmark scheduler."
        ),
    )
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
    if args.samples_per_instance < 2 * args.minibatch_size:
        parser.error(
            "--samples-per-instance must provide at least two minibatches"
        )
    if not 0.0 <= args.adapter_dropout <= 1.0:
        parser.error("--adapter-dropout must be in [0, 1]")
    if args.learning_rate <= 0 or args.optimizer_eps <= 0:
        parser.error("optimizer learning rate and epsilon must be positive")
    if args.submission_interval_ms < 0:
        parser.error("--submission-interval-ms must be non-negative")
    if args.dataset == "advertise-gen" and any(
        value < 0 or (value > 0 and value % 8 != 0)
        for value in (args.dataset_query_tokens, args.dataset_response_tokens)
    ):
        parser.error(
            "real-dataset force-token counts must be 0 or positive multiples of 8"
        )
    return args


def main() -> None:
    args = parse_args()
    config = TrainBenchmarkConfig(
        model_name=args.model,
        samples_per_instance=args.samples_per_instance,
        parallel_instances=args.instances,
        minibatch_size=args.minibatch_size,
        grad_accum_steps=args.grad_accum_steps,
        sample_len=args.sample_len,
        bf16=args.bf16,
        detail_trace=args.detail_trace,
        ckpt_period=args.ckpt_period,
        adapter=args.adapter,
        adapter_prompt_template=args.adapter_prompt_template,
        adapter_dropout=args.adapter_dropout,
        ptuning_prefix_len=args.ptuning_prefix_len,
        lora_r_q=args.lora_r_q,
        lora_r_k=args.lora_r_k,
        lora_r_v=args.lora_r_v,
        lora_len_limit=args.lora_len_limit,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        optimizer_eps=args.optimizer_eps,
        submission_interval_ms=args.submission_interval_ms,
        dataset=args.dataset,
        dataset_seed=args.dataset_seed,
        dataset_query_tokens=args.dataset_query_tokens,
        dataset_response_tokens=args.dataset_response_tokens,
        num_paths=args.num_paths,
        allow_provider_reuse=args.allow_provider_reuse,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "args.json").open("w") as file:
        json.dump(
            {
                "orbit": args.orbit,
                "host_id": args.host_id,
                "advertise_host": args.advertise_host,
                "agent": args.agent,
                "async_backend": args.async_backend,
                "busy_wait": args.busy_wait,
                "benchmark": asdict(config),
            },
            file,
            indent=2,
        )
    run_satellite(
        role=args.host_id,
        orbit_addr=args.orbit,
        advertise_host=args.advertise_host,
        config=config,
        output_dir=args.output,
        agent_node=args.agent,
        async_backend=args.async_backend,
        busy_wait=args.busy_wait,
    )


if __name__ == "__main__":
    main()
