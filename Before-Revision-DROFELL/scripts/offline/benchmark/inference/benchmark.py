import argparse
import importlib
import os
import shutil


import extend_asyncio as asyncio
from scripts.offline.benchmark.inference.runner import (make_and_put_user_model,
                                                        run_benchmark_multi)

parser = argparse.ArgumentParser()
parser.add_argument("--profile", type=str, default="7B", required=True, help="profile file to use (under profiles dir)")
parser.add_argument("--output", type=str, default="", help="detail results output file")
args, _ = parser.parse_known_args()
profile_file = args.profile
output_dir = args.output
if output_dir == "":
    output_dir = os.path.join(os.path.dirname(__file__), "results", f"{profile_file}")
os.makedirs(output_dir, exist_ok=True)
shutil.copyfile(os.path.join(os.path.dirname(__file__), "profiles", f"{profile_file}.py"), os.path.join(output_dir, "profile.py"))
profile = importlib.import_module(f"scripts.offline.benchmark.inference.profiles.{profile_file}")

asyncio.config_impl(*profile.ASYNC_IMPLEMENT)


async def amain():
    print(f"result will be write to: {output_dir}")
    testbed = await profile.GEN_TESTBED()
    if profile.USER_MODEL_DATA_CFG is not None:
        await make_and_put_user_model(
            testbed=testbed,
            cfg=profile.USER_MODEL_DATA_CFG,
            state_file=profile.USER_MODEL_DATA_FILE,
            user_model_name=profile.USER_MODEL_DATA_NAME
        )

    await run_benchmark_multi(
        testbed=testbed,
        groups=profile.GROUPS("user-infer"),
        test_config=profile.TEST_CFG,
        output_file=os.path.join(output_dir, "result.json"),
        max_len=profile.SEQ_LEN,
        multi_task_count=profile.PARALLEL_SEQS,
        user_model_name=profile.USER_MODEL_DATA_NAME,
        sample_string=profile.SAMPLE_STR
    )
    print("to quick review result, use the following cmd:")
    print(f"python analysis_utils.py {output_dir}")


asyncio.run(amain())
