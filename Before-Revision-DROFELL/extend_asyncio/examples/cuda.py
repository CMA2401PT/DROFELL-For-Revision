import os
from typing import List, cast
import torch
import time
import pandas as pd
import extend_asyncio as asyncio
import cuda_async_load

asyncio.config_impl("golang", True)

device = torch.device("cuda")
tensor_size, batch = (2048, 10)  # (512,10)
tensor_a = torch.rand((tensor_size, tensor_size), device=device)
tensor_b = torch.rand((tensor_size, tensor_size), device=device)


async def test_c_blocking_closure_overhead(repeat_times: int):
    stream = torch.cuda.Stream()
    loop = asyncio.get_event_loop()
    results = []
    for i in range(repeat_times):
        t0 = time.time_ns()
        sync_event = cuda_async_load.CudaEventSyncer(stream.cuda_stream)
        t1 = time.time_ns()
        fut = loop.delegate_c_blocking_closure(sync_event)
        t2 = time.time_ns()
        t3 = await fut
        t4 = time.time_ns()
        results.append({
            "t1-t0": (t1 - t0) / 1e6,
            "t2-t1": (t2 - t1) / 1e6,
            "t3-t2": (t3 - t2) / 1e6,
            "t4-t3": (t4 - t3) / 1e6,
            "t4-t0": (t4 - t0) / 1e6,
        })
    df = pd.DataFrame(results[10:])
    print("\n=== Summary-Empty-Load-Blocking-All (ms) ===")
    print(df.describe().to_markdown(floatfmt=".4f"))

# asyncio.run(test_c_blocking_closure_overhead(100))


async def test_c_poll_closure_overhead(repeat_times: int):
    stream = torch.cuda.Stream()
    loop = asyncio.get_event_loop()
    results = []
    for i in range(repeat_times):
        t0 = time.time_ns()
        sync_event = cuda_async_load.CudaEventSyncer(stream.cuda_stream)
        t1 = time.time_ns()
        fut = loop.delegate_c_poll_closure(sync_event)
        t2 = time.time_ns()
        t3 = await fut
        t4 = time.time_ns()
        results.append({
            "t1-t0": (t1 - t0) / 1e6,
            "t2-t1": (t2 - t1) / 1e6,
            "t3-t2": (t3 - t2) / 1e6,
            "t4-t3": (t4 - t3) / 1e6,
            "t4-t0": (t4 - t0) / 1e6,
        })
    df = pd.DataFrame(results[10:])
    print("\n=== Summary-Empty-Load-Poll-All (ms) ===")
    print(df.describe().to_markdown(floatfmt=".4f"))

# asyncio.run(test_c_poll_closure_overhead(100))
# os._exit(-1)


def time_consuming_cuda_operate():
    for i in range(batch):
        o1 = torch.matmul(tensor_a, tensor_b)


def quick_test(repeat_times: int):
    torch.cuda.synchronize()
    for _ in range(repeat_times):
        time_consuming_cuda_operate()
    torch.cuda.synchronize()

    st = time.perf_counter_ns()
    for _ in range(repeat_times):
        time_consuming_cuda_operate()
    torch.cuda.synchronize()
    et = time.perf_counter_ns()
    per_run_device_time = (et-st)/repeat_times
    print(f"per run device time: {per_run_device_time/(10**6):.4f}ms")
    return per_run_device_time


device_time = quick_test(300)


async def test_c_blocking_closure_overhead_with_load(repeat_times: int):
    stream = cast(torch.cuda.Stream, torch.cuda.Stream())
    with torch.cuda.stream(stream):
        time_consuming_cuda_operate()
    loop = asyncio.get_event_loop()
    results = []
    for i in range(repeat_times):
        st = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            time_consuming_cuda_operate()
        sync_event = cuda_async_load.CudaEventSyncer(stream.cuda_stream)
        await loop.delegate_c_blocking_closure(sync_event)
        event_sync_over_head = (time.perf_counter_ns()-st)-device_time
        results.append({"event_sync_over_head": event_sync_over_head / 1e6, })

    df = pd.DataFrame(results[10:len(results)//2])
    print("\n=== Summary-With-Load-Blocking-Half (ms) ===")
    print(df.describe().to_markdown(floatfmt=".4f"))

    df = pd.DataFrame(results[10:])
    print("\n=== Summary-With-Load-Blocking-All (ms) ===")
    print(df.describe().to_markdown(floatfmt=".4f"))

# asyncio.run(test_c_blocking_closure_overhead_with_load(100))


async def test_cuda_synchronize_overhead_with_load(repeat_times: int):
    stream = cast(torch.cuda.Stream, torch.cuda.Stream())
    with torch.cuda.stream(stream):
        time_consuming_cuda_operate()
    loop = asyncio.get_event_loop()
    results = []
    for i in range(repeat_times):
        st = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            time_consuming_cuda_operate()
        torch.cuda.synchronize()
        event_sync_over_head = (time.perf_counter_ns()-st)-device_time
        results.append({"event_sync_over_head": event_sync_over_head / 1e6, })

    df = pd.DataFrame(results[10:len(results)//2])
    print("\n=== Summary-With-Load-Synchronize-Half (ms) ===")
    print(df.describe().to_markdown(floatfmt=".4f"))

    df = pd.DataFrame(results[10:])
    print("\n=== Summary-With-Load-Synchronize-All (ms) ===")
    print(df.describe().to_markdown(floatfmt=".4f"))

asyncio.run(test_cuda_synchronize_overhead_with_load(100))
os._exit(0)


async def test_c_poll_closure_overhead_with_load(repeat_times: int):
    stream = cast(torch.cuda.Stream, torch.cuda.Stream())
    with torch.cuda.stream(stream):
        time_consuming_cuda_operate()
    loop = asyncio.get_event_loop()
    results = []
    for i in range(repeat_times):
        st = time.perf_counter_ns()
        with torch.cuda.stream(stream):
            time_consuming_cuda_operate()
        sync_event = cuda_async_load.CudaEventSyncer(stream.cuda_stream)
        await loop.delegate_c_poll_closure(sync_event)
        event_sync_over_head = (time.perf_counter_ns()-st)-device_time
        results.append({"event_sync_over_head": event_sync_over_head / 1e6, })

    df = pd.DataFrame(results[10:len(results)//2])
    print("\n=== Summary-With-Load-Poll-Half (ms) ===")
    print(df.describe().to_markdown(floatfmt=".4f"))

    df = pd.DataFrame(results[10:])
    print("\n=== Summary-With-Load-Poll-All (ms) ===")
    print(df.describe().to_markdown(floatfmt=".4f"))

asyncio.run(test_c_poll_closure_overhead_with_load(100))

os._exit(-1)

repeat_times = 10


def pre_test():
    torch.cuda.synchronize()
    for _ in range(repeat_times):
        time_consuming_cuda_operate()
    torch.cuda.synchronize()

    st = time.perf_counter_ns()
    for _ in range(repeat_times):
        time_consuming_cuda_operate()
    torch.cuda.synchronize()
    et = time.perf_counter_ns()
    per_run_device_time = (et-st)/repeat_times
    print(f"per run device time: {per_run_device_time/(10**6):.4f}ms")
    print("="*80)

    torch.cuda.synchronize()
    launch_time: List[float] = []
    extra_time: List[float] = []
    st = time.perf_counter_ns()
    for _ in range(repeat_times):
        time_consuming_cuda_operate()
        et = time.perf_counter_ns()
        launch_time.append(et-st)
        st = et
        torch.cuda.synchronize()
        et = time.perf_counter_ns()
        extra_time.append(et-st)
        st = et
    per_run_cpu_time = sum(launch_time)/repeat_times
    per_run_extra_device_time = sum(extra_time)/repeat_times
    print(f"per run cpu time: {per_run_cpu_time/(10**6):.4f}ms")
    print(f"per run extra device time: {per_run_extra_device_time/(10**6):.4f}ms")
    print("="*80)


pre_test()


# async def cuda_coro():
#     loop = asyncio.get_event_loop()

#     print("simple 1 stream run")
#     torch.cuda.synchronize()
#     loop.enable_profile(True)
#     st = time.perf_counter_ns()
#     for _ in range(repeat_times):
#         time_consuming_cuda_operate()
#         await asyncio.sleep(0)
#     torch.cuda.synchronize()
#     await asyncio.sleep(0)
#     et = time.perf_counter_ns()
#     per_run_device_time = (et-st)/repeat_times
#     print(f"per run device time (no async wait) : {per_run_device_time/(10**6):.4f}ms")
#     print(f"blocking time (no async wait) :{loop.blocking_time_ms:.4f}ms")
#     loop.clear_profile()
#     print("="*80)

#     print("simple 1 stream with async wait")
#     torch.cuda.synchronize()
#     loop.enable_profile(True)
#     st = time.perf_counter_ns()
#     async_bocking_wait_time: List[int] = []
#     for _ in range(repeat_times):
#         time_consuming_cuda_operate()
#         sync_event = cuda_async_load.CudaEventSyncer()
#         async_bocking_wait_time.append(await loop.delegate_c_blocking_closure(sync_event))
#     torch.cuda.synchronize()
#     et = time.perf_counter_ns()
#     per_run_device_time = (et-st)/repeat_times
#     async_bocking_wait_ns = sum(async_bocking_wait_time)/len(async_bocking_wait_time)
#     print(f"per run device time (async wait) : {per_run_device_time/(10**6):.4f}ms")
#     print(f"blocking time (async wait):{loop.blocking_time_ms:.4f}ms")
#     print(f"async blocking time (async wait):{async_bocking_wait_ns/(10**6):.4f}ms")
#     loop.clear_profile()
#     print("="*80)

#     print("1 stream with dequeue async wait")
#     queue_depth = 3
#     torch.cuda.synchronize()
#     evs: deque[Awaitable[int]] = deque()
#     for _ in range(queue_depth):
#         sync_event = cuda_async_load.CudaEventSyncer()
#         evs.append(loop.delegate_c_blocking_closure(sync_event))
#     loop.enable_profile(True)
#     st = time.perf_counter_ns()
#     async_bocking_wait_time: List[int] = []
#     for _ in range(repeat_times):
#         time_consuming_cuda_operate()
#         sync_event = cuda_async_load.CudaEventSyncer()
#         evs.append(loop.delegate_c_blocking_closure(sync_event))
#         async_bocking_wait_time.append(await evs.popleft())
#     torch.cuda.synchronize()
#     et = time.perf_counter_ns()
#     per_run_device_time = (et-st)/repeat_times
#     print(f"per run device time (dequeue async wait) : {per_run_device_time/(10**6):.4f}ms")
#     print(f"blocking time (dequeue async wait):{loop.blocking_time_ms:.4f}ms")
#     loop.clear_profile()
#     print("="*80)

#     print("2 stream with hide")
#     torch.cuda.synchronize()
#     async_bocking_wait_time_1: List[int] = []
#     async_bocking_wait_time_2: List[int] = []
#     stream_1 = torch.cuda.Stream(device)
#     stream_2 = torch.cuda.Stream(device)
#     loop.clear_profile()
#     loop.enable_profile(True)

#     st = time.perf_counter_ns()

#     async def sub_coro_1():
#         for _ in range(repeat_times):
#             with torch.cuda.stream(stream_1):
#                 time_consuming_cuda_operate()
#                 sync_event = cuda_async_load.CudaEventSyncer()
#                 async_bocking_wait_time_1.append(await loop.delegate_c_blocking_closure(sync_event))

#     async def sub_coro_2():
#         for _ in range(repeat_times):
#             with torch.cuda.stream(stream_2):
#                 time_consuming_cuda_operate()
#                 sync_event = cuda_async_load.CudaEventSyncer()
#                 async_bocking_wait_time_2.append(await loop.delegate_c_blocking_closure(sync_event))
#     t1 = asyncio.create_task(sub_coro_1())
#     t2 = asyncio.create_task(sub_coro_2())
#     await t1
#     await t2
#     et = time.perf_counter_ns()
#     per_run_device_time = ((et-st)/(repeat_times))/2
#     async_bocking_wait_ns_1 = sum(async_bocking_wait_time_1)/len(async_bocking_wait_time_1)
#     async_bocking_wait_ns_2 = sum(async_bocking_wait_time_2)/len(async_bocking_wait_time_2)
#     print(f"per run device time (with hide) : {per_run_device_time/(10**6):.4f}ms")
#     print(f"blocking time (with hide):{loop.blocking_time_ms:.4f}ms")
#     print(f"async blocking time, stream 1 (with hide):{async_bocking_wait_ns_1/(10**6):.4f}ms")
#     print(f"async blocking time, stream 2 (with hide):{async_bocking_wait_ns_2/(10**6):.4f}ms")
#     loop.clear_profile()
#     print("="*80)

#     print("n stream with hide")

#     def create_stream():
#         individual_stream = torch.cuda.Stream(device)

#         async def _sub_coro():
#             async_bocking_wait_time: List[int] = []
#             for _ in range(repeat_times):
#                 with torch.cuda.stream(individual_stream):
#                     time_consuming_cuda_operate()
#                     sync_event = cuda_async_load.CudaEventSyncer()
#                     async_bocking_wait_time.append(await loop.delegate_c_blocking_closure(sync_event))
#             return sum(async_bocking_wait_time)/len(async_bocking_wait_time)
#         return _sub_coro()

#     streams_count = 5
#     loop.clear_profile()
#     tasks = [asyncio.create_task(create_stream()) for _ in range(streams_count)]
#     loop.enable_profile(True)
#     st = time.perf_counter_ns()
#     blocking_times = [(await t) for t in tasks]
#     et = time.perf_counter_ns()
#     torch.cuda.synchronize()
#     per_run_device_time = ((et-st)/(repeat_times))/streams_count
#     print(f"per run device time (with hide) : {per_run_device_time/(10**6):.4f}ms")
#     print(f"blocking time (with hide):{loop.blocking_time_ms:.4f}ms")
#     for (stream_i, async_bocking_wait_ns) in enumerate(blocking_times):
#         print(f"async blocking time, stream {stream_i} (with hide):{async_bocking_wait_ns/(10**6):.4f}ms")
#     print("="*80)
#     # loop.clear_profile()
#     # torch.cuda.synchronize()
#     # stream_1 = torch.cuda.Stream(device)
#     # stream_2 = torch.cuda.Stream(device)
#     # last_ev1 = loop.torch_event_sync()
#     # last_ev2 = loop.torch_event_sync()

#     # loop.enable_profile(True)
#     # st = time.perf_counter_ns()
#     # async_bocking_wait_time: List[int] = []
#     # for _ in range(repeat_times):
#     #     async_bocking_wait_time.append(await last_ev1)
#     #     with torch.cuda.stream(stream_1):
#     #         time_consuming_cuda_operate()
#     #         last_ev1 = loop.torch_event_sync()
#     #     async_bocking_wait_time.append(await last_ev2)
#     #     with torch.cuda.stream(stream_2):
#     #         time_consuming_cuda_operate()
#     #         last_ev2 = loop.torch_event_sync()

#     # torch.cuda.synchronize()
#     # await asyncio.sleep(0)
#     # et = time.perf_counter_ns()
#     # per_run_device_time = (et-st)/repeat_times
#     # async_bocking_wait_ns = sum(async_bocking_wait_time)/len(async_bocking_wait_time)
#     # print(f"per run device time (async wait) : {per_run_device_time/(10**6):.4f}ms")
#     # print(f"blocking time (async wait):{loop.blocking_time_ms:.4f}ms")
#     # print(f"async blocking time (async wait):{async_bocking_wait_ns/(10**6):.4f}ms")
#     # # await loop.torch_event_sync()
#     # print("="*80)

# asyncio.run(cuda_coro())
