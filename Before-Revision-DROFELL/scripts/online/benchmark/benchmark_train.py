import argparse
import asyncio
import importlib
import json
import os
import shutil
import sys
import time
from asyncio import Task
from typing import Any, Callable, Dict, List, Optional, Tuple

import extend_asyncio as asyncio
from dataset.common import ProcessedRecord
from node.aux_node.protocols import req_extra_info, req_providers
from node.base import BaseNode
from node.client.client import ClientNode
from node.compute_node.train_meta import TrainHeadInput
from node.train.train_ctrl import TrainControl
from orbit import *
from orbit.defines import Context
from orbit.link import ContextNetExtender
from orbit.link.extender import ContextNetExtender
from scripts.base.train_state_maintainer import StepSize, TrainStatusMaintainer
from utils.minibatch_feeder import AsyncMinibatchFeeder


async def run_benchmark_multi(
    create_instance: Callable[[str], TrainControl],
    remove_instance: Callable[[TrainControl], None],
    samples_each_instance: int = 100,
    parallel_instances: int = 2,
    batch_size: int = 2,
    minibatch_size: int = 2,
    sample_len: int = 128,
    is_bfloat_16: bool = False,
    with_detail_trace: bool = False,
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

        def gen_sample():
            nonlocal total_samples
            if total_samples >= samples_each_instance:
                return None

            records: List[ProcessedRecord] = []
            for _ in range(minibatch_size):
                record = ProcessedRecord(
                    input_ids=[3 for _ in range(sample_len)],
                    output_ids=[3 for _ in range(sample_len)]
                )
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

        running_tasks: Tuple[Task[None], ...]
        _running_tasks: List[Task[None]] = []
        for _ in range(2):
            sample = await feeder.get_next(wait=False)
            assert sample is not None
            t = asyncio.create_task(do_one_batch(sample))
            _running_tasks.append(t)

        running_tasks = tuple(_running_tasks)

        while True:
            sample = await feeder.get_next(wait=False)
            if sample is None:
                break
            task: Task[None] = asyncio.create_task(do_one_batch(sample))
            running_tasks = (*running_tasks, task)
            done, pending = await asyncio.wait_first(running_tasks)
            running_tasks = tuple(pending)
        while running_tasks:
            done, pending = await asyncio.wait_first(running_tasks)
            running_tasks = tuple(pending)
        remove_instance(train_ctrl)
        return train_status_maintainer.dump_info()
    results = await asyncio.gather(*[
        instance(f"train-{instance_i}", instance_i)
        for instance_i in range(profile.PARALLEL_INSTANCES)
    ])
    results.append(global_train_status_maintainer.dump_info())
    return results, detail_time_traces


async def run_benchmark(client_node: ClientNode, profile: Any, output_dir: str):
    extra_info = await req_extra_info(client_node.base_node)
    base_models = extra_info.models[profile.MODEL_NAME].models
    keypoints = await profile.GEN_INIT_PATH(base_models, client_node)
    print("keypoints: ", keypoints)

    dealer_nodes = set(p for _, ps in await req_providers(client_node.base_node, None) for p, _ in ps)
    for dealer_node in dealer_nodes:
        tags = await client_node.base_node.req_tags(dealer_node)
        print(f"{dealer_node} {tags}")
        if 'time_trace' in tags:
            await client_node.base_node.rpc_no_args(dealer_node, "telemetric-time-start", Optional[List[str]])

    def create_instance(name: str):
        train_ctrl = client_node.create_train_control(
            user_model_name=name,
            base_models=base_models,
            user_model_config=profile.USER_MODEL_DATA_CFG,
            optimizer_config=profile.OPTIMIZER_CFG,
            scheduler_config=profile.SCHEDULER_CFG,
            user_model_init_data_provider=profile.USER_MODEL_INIT_DATA_PROVIDER,
            keypoint_providers=keypoints,
            ckpt_period=profile.CKPT_PERIOD
        )
        return train_ctrl

    time_record_multi, detail_time_traces = await run_benchmark_multi(
        create_instance=create_instance,
        remove_instance=lambda instance: client_node.remove_train_control(instance.ctrl_id),
        samples_each_instance=profile.SAMPLES_EACH_INSTANCE,
        parallel_instances=profile.PARALLEL_INSTANCES,
        batch_size=profile.BATCH_SIZE,
        minibatch_size=profile.MINIBATCH_SIZE,
        sample_len=profile.SAMPLE_LEN,
        is_bfloat_16=profile.IS_BFLOAT16,
        with_detail_trace=profile.DETAIL_TRACE
    )

    dealer_nodes = set(p for _, ps in await req_providers(client_node.base_node, None) for p, _ in ps)
    dealer_time_traces: Dict[str, Any] = {}
    for dealer_node in dealer_nodes:
        tags = await client_node.base_node.req_tags(dealer_node)
        print(f"{dealer_node} {tags}")
        if 'time_trace' in tags:
            dealer_time_trace = await client_node.base_node.rpc_no_args(dealer_node, "telemetric-time-end", Optional[List[str]])
            dealer_time_traces[dealer_node] = dealer_time_trace

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(time_record_multi, f)
    if len(detail_time_traces) > 0:
        with open(os.path.join(output_dir, "traces.json"), "w") as f:
            json.dump(detail_time_traces, f)
    if len(dealer_time_traces) > 0:
        with open(os.path.join(output_dir, "brains.json"), "w") as f:
            json.dump(dealer_time_traces, f)
    print("to quick review result, use the following cmd:")
    print(f"python analysis_train.py {output_dir}")

    await asyncio.sleep(0.3)
    exit(0)


def benchmark_node(ctx: Context):
    async def coro(ctx: Context):
        ctx.report_ready()
        net_extend = ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        ctrl = ClientNode(base_node)
        asyncio.create_task(run_benchmark(ctrl, ctx.args.profile, ctx.args.output_dir))

        while True:
            packet = await ctx.recv()
            base_node.on_packet(packet)
    asyncio.get_event_loop().create_task(coro(ctx))


def run_satellite(role: str, orbit_addr: str, profile: Any, output_dir: str):
    def init_fn():
        def on_except(_: asyncio.Loop, e: Exception):
            raise e
        asyncio.config_impl(*profile.ASYNC_IMPLEMENT)
        loop = asyncio.get_event_loop()
        loop.on_except(on_except)
        return loop
    launch_pad = LaunchPad(init_fn)

    async def coro():
        satellite = Satellite(launch_pad=launch_pad, functions_dict={}, host_id=role, network_hostname="", stub_remote_log=True)
        satellite.start(orbit_addr=orbit_addr)
        await satellite.ready()
        satellite.launch(benchmark_node, profile=profile, output_dir=output_dir)
        await asyncio.Future()

    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="7B", required=True, help="profile file to use (under profiles/train dir)")
    parser.add_argument("--output", type=str, default="", help="detail results output file")
    parser.add_argument("--setup", type=str, default="", help="setup task to run before benchmark")
    parser.add_argument("--orbit", type=str, default="localhost:7241", help="orbit addr", required=False)
    args, unknown = parser.parse_known_args()
    profile_file = args.profile
    output_dir = args.output
    profile = importlib.import_module(f"scripts.online.benchmark.profiles.train.{profile_file}")
    if output_dir == "":
        output_dir = os.path.join(os.path.dirname(__file__), "results", "train", f"{profile_file}")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(os.path.join(os.path.dirname(__file__), "profiles", "train", f"{profile_file}.py"), os.path.join(output_dir, "profile.py"))
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(sys.argv[1:], f)

    setup_task = args.setup
    if setup_task != "":
        print(f"setup task is set, profile async implement will be ignored, extra args: {profile.SETUP_EXTRA} will effect setup")

        script = importlib.import_module(f"scripts.online.benchmark.setups.{setup_task}")
        shutil.copyfile(os.path.join(os.path.dirname(__file__), "setups", setup_task+".py"), os.path.join(output_dir, "setup.py"))
        launch_pad = LaunchPad(script.PROC_INIT)
        unknown += profile.SETUP_EXTRA

        async def amain():
            orbit = await script.SETUP(launch_pad, *unknown)
            print("Setup Done")
            await asyncio.sleep(1)
            await orbit.launch_local("benchmark", benchmark_node, profile=profile, output_dir=output_dir)
            await asyncio.Future()

        launch_pad.loop.create_task(amain())
        launch_pad.loop.run_forever()
    else:
        print(f"setup task is not set, profile async implement: {profile.ASYNC_IMPLEMENT} will effect, extra args will be ignored")
        run_satellite("benchmark", args.orbit, profile, output_dir)
