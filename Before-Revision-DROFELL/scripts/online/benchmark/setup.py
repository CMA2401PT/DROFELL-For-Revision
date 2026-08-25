import argparse
import importlib
import time
import extend_asyncio as asyncio
from orbit.launch_pad import LaunchPad

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, required=False, default="7B_basic", help="setup script to use")
    args, unknown = parser.parse_known_args()
    script_file = args.script
    script = importlib.import_module(f"scripts.online.benchmark.setups.{script_file}")
    launch_pad = LaunchPad(script.PROC_INIT)

    async def amain():
        await script.SETUP(launch_pad, *unknown)
        print(f"Setup Done in {time.time()-start_time}")
        await asyncio.Future()

    launch_pad.loop.create_task(amain())
    launch_pad.loop.run_forever()

# asyncio.config_impl(*profile.ASYNC_IMPLEMENT)


# async def amain():
#     print(f"result will be write to: {output_dir}")
#     testbed = await profile.GEN_TESTBED()
#     if profile.USER_MODEL_DATA_CFG is not None:
#         await make_and_put_user_model(
#             testbed=testbed,
#             cfg=profile.USER_MODEL_DATA_CFG,
#             state_file=profile.USER_MODEL_DATA_FILE,
#             user_model_name=profile.USER_MODEL_DATA_NAME
#         )

#     await run_benchmark_multi(
#         testbed=testbed,
#         groups=profile.GROUPS("user-infer"),
#         test_config=profile.TEST_CFG,
#         output_file=os.path.join(output_dir, "result.json"),
#         max_len=profile.SEQ_LEN,
#         multi_task_count=profile.PARALLEL_SEQS,
#         user_model_name=profile.USER_MODEL_DATA_NAME,
#         sample_string=profile.SAMPLE_STR
#     )
#     print("to quick review result, use the following cmd:")
#     print(f"python analysis_utils.py {output_dir}")


# asyncio.run(amain())
