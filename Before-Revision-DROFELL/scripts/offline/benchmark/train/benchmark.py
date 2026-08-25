import argparse
from asyncio import Task
import importlib
import json
import os

import time
from typing import List, Tuple
from dataset.common import ProcessedRecord
from node.compute_node import TrainHeadInput
from scripts.base.train_state_maintainer import TrainStatusMaintainer, StepSize
from utils.minibatch_feeder import AsyncMinibatchFeeder
import extend_asyncio as asyncio

parser = argparse.ArgumentParser()
parser.add_argument("--profile", type=str, default="7B", required=True, help="profile file to use (under profiles dir)")
parser.add_argument("--output", type=str, default="", help="detail results output dir")
args, _ = parser.parse_known_args()
profile_file = args.profile
output_dir = args.output
if output_dir == "":
    output_dir = os.path.join(os.path.dirname(__file__), "results", f"{profile_file}")

profile = importlib.import_module(f"scripts.offline.benchmark.train.profiles.{profile_file}")

asyncio.config_impl(*profile.ASYNC_IMPLEMENT)


async def amain():
    print(f"result will be write to: {output_dir}")
    testbed, extra_record_dump = await profile.GEN_TESTBED()

    os.makedirs(output_dir, exist_ok=True)

    global_train_status_maintainer = TrainStatusMaintainer(
        step_size=StepSize(
            steps_in_each_update=profile.BATCH_SIZE,
            step_batch_size=profile.MINIBATCH_SIZE
        ),
        total_samples=profile.SAMPLES_EACH_INSTANCE*profile.PARALLEL_INSTANCES,
        events=None,
        prefix="total"
    )

    async def instance(prefix: str):
        total_samples = 0

        def gen_sample():
            nonlocal total_samples
            # print(f"{total_samples=} {profile.SAMPLES_EACH_INSTANCE=}")
            if total_samples >= profile.SAMPLES_EACH_INSTANCE:
                return None

            records: List[ProcessedRecord] = []
            for _ in range(profile.MINIBATCH_SIZE):
                record = ProcessedRecord(
                    input_ids=[3 for _ in range(profile.SAMPLE_LEN)],
                    output_ids=[3 for _ in range(profile.SAMPLE_LEN)]
                )
                records.append(record)
                total_samples += 1
                # print(f"{total_samples=} {profile.SAMPLES_EACH_INSTANCE=}")
                if total_samples >= profile.SAMPLES_EACH_INSTANCE:
                    break
            return records

        feeder = AsyncMinibatchFeeder(gen_sample, profile.BATCH_SIZE, profile.BATCH_SIZE*2)

        train_status_maintainer = TrainStatusMaintainer(
            step_size=StepSize(
                steps_in_each_update=profile.BATCH_SIZE,
                step_batch_size=profile.MINIBATCH_SIZE
            ),
            total_samples=profile.SAMPLES_EACH_INSTANCE,
            events=None,
            prefix=prefix
        )
        train_instance = await testbed.create_train_instance(
            groups=profile.GROUPS(prefix),
            user_model_config=profile.USER_MODEL_DATA_CFG,
            optimizer_config=profile.OPTIMIZER_CFG,
            scheduler_config=profile.SCHEDULER_CFG,
            data_init_method=profile.DATA_INIT_CFG,
            test_config=profile.TEST_CFG,
        )

        async def do_one_batch(sample: Tuple[List[str], int, str, List[ProcessedRecord]]):
            batch_ids, req_version, minibatch_id, records = sample
            _input = TrainHeadInput.from_records(records, is_bfloat16=profile.IS_BFLOAT16)
            start_time = time.time()
            loss = await train_instance.do_mini_batch(req_version, full_batch_ids=batch_ids, minibatch_id=minibatch_id, sample=_input)
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
        return train_status_maintainer.dump_info()
    results = await asyncio.gather(*[
        instance(f"train-{instance_i}")
        for instance_i in range(profile.PARALLEL_INSTANCES)
    ])
    results.append(global_train_status_maintainer.dump_info())
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f)
    extra_record_dump(output_dir)
    print("to quick review result, use the following cmd:")
    print(f"python analysis_utils.py {output_dir}")


asyncio.run(amain())
