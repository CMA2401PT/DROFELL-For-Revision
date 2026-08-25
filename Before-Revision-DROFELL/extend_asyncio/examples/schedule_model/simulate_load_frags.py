import asyncio
import argparse
import os
import random
import time
from typing import Callable, Dict, List, Tuple, cast
from async_aware_scheduler import AsyncAwareSubmitter
from normal_scheduler import BlockingSubmitter, get_scheduler
from matplotlib import pyplot as plt
import numpy as np
from orbit import *
import extend_asyncio as asyncio
from orbit.link import OrbitCPNCommander, LinkDefine, LinkPerfDescribe, PacketTypeNewNode, ContextNetExtender
from load import work

T_NORMAL = 0.02
T_PRIORITY_TIME = 0.01
T_NORMAL_WORK_PERCENTAGE = 0.2
T_NORMAL_SPLITS_CONT = 10
T_NORMAL_STEP = T_NORMAL/T_NORMAL_SPLITS_CONT
T_NORMAL_STEP_WORK = T_NORMAL_STEP*T_NORMAL_WORK_PERCENTAGE
T_NORMAL_STEP_IDLE = T_NORMAL_STEP*(1-T_NORMAL_WORK_PERCENTAGE)

ASYNC_AWARE = True
T_SUBMIT = T_NORMAL_STEP_WORK+T_NORMAL_STEP_IDLE


# del T_NORMAL, T_NORMAL_STEP

EXP_SECONDS = 2
COOL_DOWN_SECONDS = 0.3


RANDOM_TIMES: list[float] = []
random.seed(42)
while sum(RANDOM_TIMES) < EXP_SECONDS:
    RANDOM_TIMES.append(random.random() % (T_NORMAL))


def runner_node(ctx: Context):
    commander_ready = False
    end_time_seq = 0
    end_time_random = 0
    submitter: BlockingSubmitter

    async def low_priority_pusher():
        nonlocal submitter
        while True:
            st = time.time_ns()
            for _ in range(T_NORMAL_SPLITS_CONT):
                if not ASYNC_AWARE:
                    await submitter.submit(work, T_NORMAL_STEP_WORK+T_NORMAL_STEP_IDLE, "low-priority", task_name="low")
                else:
                    await submitter.submit(work, T_NORMAL_STEP_WORK, "low-priority", task_name="low")
            et = time.time_ns()
            if commander_ready:
                ctx.send(("commander", "low", (st, et)))
            if end_time_seq > 0 and et > end_time_seq:
                break
        ctx.send(("commander", "end_seq", None))
        while True:
            st = time.time_ns()
            for _ in range(T_NORMAL_SPLITS_CONT):
                if not ASYNC_AWARE:
                    await submitter.submit(work, T_NORMAL_STEP_WORK+T_NORMAL_STEP_IDLE, "low-priority", task_name="low")
                else:
                    await submitter.submit(work, T_NORMAL_STEP_WORK, "low-priority", task_name="low")
            et = time.time_ns()
            if commander_ready:
                ctx.send(packet=("commander", "low", (st, et)))
            if et > end_time_random:
                break
        ctx.send(("commander", "end_random", None))

    async def coro(ctx: Context):
        nonlocal commander_ready
        nonlocal end_time_seq
        nonlocal end_time_random
        nonlocal submitter
        container_implement = ctx.args.implement
        submitter = get_scheduler(container_implement)
        if ASYNC_AWARE:
            submitter = AsyncAwareSubmitter(submitter, T_NORMAL_STEP_IDLE)

        net_extend = ContextNetExtender(ctx)
        ctx.report_ready()
        await net_extend.wait_node_net_ready()

        asyncio.create_task(low_priority_pusher())

        def high_priority_task():
            st = time.time_ns()
            work(T_PRIORITY_TIME, "high-priority")
            return st
        while True:
            src_name, api, data = await ctx.recv()
            if src_name == OrbitName:
                if api == PacketTypeNewNode and data == "commander":
                    ctx.logger.info(f"acknowledge commander ready")
                    commander_ready = True
                    end_time_seq = time.time_ns()+EXP_SECONDS*(10**9)
                    end_time_random = time.time_ns()+(EXP_SECONDS*(10**9)*2+COOL_DOWN_SECONDS)
                continue

            async def _sub_task(recv_time: int):
                priority_st = await submitter.submit(high_priority_task, task_name="high")
                priority_et = time.time_ns()

                async def _send_task():
                    ctx.send(("commander", "high", (recv_time, priority_st, priority_et)))
                # simulate: send coro is created by task coro
                asyncio.create_task(_send_task())
            # simulate: task coro is created by recv coro
            recv_time = time.time_ns()
            asyncio.create_task(_sub_task(recv_time))
    asyncio.get_event_loop().create_task(coro(ctx))


def commander_node(ctx: Context):
    async def coro(ctx: Context):
        net_extend = ContextNetExtender(ctx)
        ctx.report_ready()
        await net_extend.wait_node_net_ready()
        finish_queue = asyncio.Queue()
        priority_results: List[Tuple[float, float, float, float, float]] = []
        normal_results: List[Tuple[float, float]] = []
        seq_end = False
        random_end = False

        async def high_priority_pusher():
            nonlocal priority_results, normal_results
            while not seq_end:
                st = time.time_ns()
                ctx.send(("runner", "", st))
                recv_time, priority_st, priority_et = await finish_queue.get()
                rt: int = time.time_ns()
                priority_results.append((st, recv_time, priority_st, priority_et, rt))

            await asyncio.sleep(COOL_DOWN_SECONDS)
            ctx.send((OrbitName, "start_random", (priority_results, normal_results)))
            priority_results = []
            normal_results = []

            while not random_end:
                sleep_time = RANDOM_TIMES.pop()
                await asyncio.sleep(sleep_time)
                st = time.time_ns()
                ctx.send(("runner", "", st))
                recv_time, priority_st, priority_et = await finish_queue.get()
                rt: int = time.time_ns()
                priority_results.append((st, recv_time, priority_st, priority_et, rt))

        asyncio.create_task(high_priority_pusher())

        while True:
            src_name, api, data = await ctx.recv()
            if src_name == OrbitName:
                pass
            assert src_name == "runner"
            if api == "high":
                finish_queue.put_nowait(data)
            elif api == "low":
                normal_results.append(data)
            elif api == "end_seq":
                ctx.send((OrbitName, "done_seq", (priority_results, normal_results)))
                seq_end = True
            else:
                assert api == "end_random"
                ctx.send((OrbitName, "done_random", (priority_results, normal_results)))
                random_end = True
    asyncio.get_event_loop().create_task(coro(ctx))


functions: Dict[str, Callable[..., None]] = {
    "runner": runner_node,
    "commander": commander_node,
}


def net_model(src: Location, dst: Location) -> LinkDefine:
    return ("tcp", LinkPerfDescribe(0, 2**10))


def plot_normal_results(normal_results: List[Tuple[float, float]], tag: str):
    plt.figure(figsize=(10, 5), dpi=330)

    plt.plot([s/(10**6) for s, _ in normal_results], [(e-s)/(10**6) for s, e in normal_results], label="Run Time")
    plt.plot([s/(10**6) for s, _ in normal_results[1:]], [(s1-s0)/(10**6) for ((s0, _), (s1, _)) in zip(normal_results[:-1], normal_results[1:])], label="Period")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), f"{tag}_one_by_one.png"))


def plot_priority_results(priority_results: List[Tuple[float, float, float, float, float]], tag: str):
    plt.figure(figsize=(10, 5), dpi=330)

    plt.plot([s/(10**6) for s, a, r, f, e in priority_results], [(f-r)/(10**6) for s, a, r, f, e in priority_results], label="Actual Run Time")
    plt.plot([s/(10**6) for s, a, r, f, e in priority_results], [(e-s)/(10**6) for s, a, r, f, e in priority_results], label="Total Time")
    plt.plot([s/(10**6) for s, a, r, f, e in priority_results], [(r-s)/(10**6) for s, a, r, f, e in priority_results], label="Pending Time")
    plt.plot([s/(10**6) for s, a, r, f, e in priority_results], [(e-f)/(10**6) for s, a, r, f, e in priority_results], label="Recv Time")
    plt.plot([s/(10**6) for s, a, r, f, e in priority_results[1:]], [(s1-s0)/(10**6) for ((s0, a0, r0, f0, e0), (s1, a0, r1, f1, e1)) in zip(priority_results[:-1], priority_results[1:])], label="Period")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), f"{tag}.png"))


def init_fn():
    def on_except(_: asyncio.Loop, e: Exception):
        raise e
        os._exit(-1)
    asyncio.config_impl("golang", True)
    loop = asyncio.get_event_loop()
    loop.on_except(on_except)
    return loop


def run_orbit(implement: str):
    launch_pad = LaunchPad(init_fn)
    logger = SimpleLogger()

    async def coro():
        orbit = Orbit(launch_pad, functions, logger=logger,  log_level="info")
        net_builder = OrbitCPNCommander(orbit, net_model)
        await orbit.init(port=7241, satellites=tuple())
        await net_builder.launch_and_join(loc=Location("", 1, 0, 2, "runner"), fn_name="runner", implement=implement)
        await net_builder.launch_and_join(loc=Location("", 0, 0, 2, "commander"), fn_name="commander")
        s = time.time_ns()
        _, _, (priority_results_seq, normal_results_seq) = await orbit.recv()
        priority_results_seq = cast(List[Tuple[float, float, float, float, float]], [tuple(e-s for e in p) for p in priority_results_seq])
        normal_results_seq: List[Tuple[float, float]] = cast(List[Tuple[float, float]], [tuple(e-s for e in p) for p in normal_results_seq])

        _, _, _ = await orbit.recv()
        s = time.time_ns()
        _, _, (priority_results_random, normal_results_random) = await orbit.recv()
        priority_results_random = cast(List[Tuple[float, float, float, float, float]], [tuple(e-s for e in p) for p in priority_results_random])
        normal_results_random: List[Tuple[float, float]] = cast(List[Tuple[float, float]], [tuple(e-s for e in p) for p in normal_results_random])

        priority_results, normal_results = priority_results_seq, normal_results_seq
        plot_priority_results(priority_results, implement+"_onebyone")
        # plot_normal_results(normal_results, IMPLEMENT)
        priority_results_total_time = np.mean([(e-s)/(10**6) for (s, a, r, f, e) in priority_results])
        normal_results_execution_time = np.mean([(e-s)/(10**6) for (s, e) in normal_results])
        print("sequental:")
        print(f"\tpriority tasks finished: {len(priority_results)}")
        print(f"\t\tavg total time: {priority_results_total_time:.2f}ms")
        print(f"\t\tavg send time: {np.mean([(a-s)/(10**6) for (s,a, r, f, e) in priority_results]):.2f}ms")
        print(f"\t\tavg pending time: {np.mean([(r-a)/(10**6) for (s,a, r, f, e) in priority_results]):.2f}ms")
        print(f"\t\tavg execution time: {np.mean([(f-r)/(10**6) for (s,a, r, f, e) in priority_results]):.2f}ms")
        print(f"\t\tavg retrieving time: {np.mean([(e-f)/(10**6) for (s,a, r, f, e) in priority_results]):.2f}ms")
        print(f"\tnormal tasks finished: {len(normal_results)}")
        print(f"\t\tavg total time: {normal_results_execution_time:.2f}ms")

        priority_results, normal_results = priority_results_random, normal_results_random
        plot_priority_results(priority_results, implement+"_random")
        # plot_normal_results(normal_results, "random")
        priority_results_total_time = np.mean([(e-s)/(10**6) for (s, a, r, f, e) in priority_results])
        normal_results_execution_time = np.mean([(e-s)/(10**6) for (s, e) in normal_results])
        print("random:")
        print(f"\tpriority tasks finished: {len(priority_results)}")
        print(f"\t\tavg total time: {priority_results_total_time:.2f}ms")
        print(f"\t\tavg send time: {np.mean([(a-s)/(10**6) for (s,a, r, f, e) in priority_results]):.2f}ms")
        print(f"\t\tavg pending time: {np.mean([(r-a)/(10**6) for (s,a, r, f, e) in priority_results]):.2f}ms")
        print(f"\t\tavg execution time: {np.mean([(f-r)/(10**6) for (s,a, r, f, e) in priority_results]):.2f}ms")
        print(f"\t\tavg retrieving time: {np.mean([(e-f)/(10**6) for (s,a, r, f, e) in priority_results]):.2f}ms")
        print(f"\tnormal tasks finished: {len(normal_results)}")
        print(f"\t\tavg total time: {normal_results_execution_time:.2f}ms")

    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


def run_satellite(role: str):
    launch_pad = LaunchPad(init_fn)

    async def coro():
        satellite = Satellite(launch_pad=launch_pad, functions_dict=functions, host_id=role, network_hostname="localhost")
        await satellite.run("localhost:7241")

    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", "-r", type=str, default=OrbitName)
    parser.add_argument("--implement", "-i", type=str, default="seq-priority")
    args, _ = parser.parse_known_args()
    role = args.role
    if role == OrbitName:
        run_orbit(args.implement)
    else:
        run_satellite(role)
