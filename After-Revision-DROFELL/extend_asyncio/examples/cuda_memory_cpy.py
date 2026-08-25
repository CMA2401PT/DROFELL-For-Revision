from typing import Any, Callable, Coroutine, List
import torch
import time
import cuda_async_load
import extend_asyncio as asyncio

asyncio.config_impl("golang", True)

tensor_size = (1024, 1024)  # (512,10)
repeat_times = 100

cpu = torch.device("cpu")
device = torch.device("cuda")


def make_tensor(device: bool):
    if device:
        return torch.empty(tensor_size, device=device)
    else:
        return torch.empty(tensor_size, device=cpu, pin_memory=True)


def time_consuming_cuda_operate(t1: torch.Tensor, t2: torch.Tensor):
    for _ in range(10):
        t1.matmul(t2)


def async_tensor_copy(dst: torch.Tensor, src: torch.Tensor, stream: torch.cuda.Stream):
    size = src.numel()*src.element_size()
    return cuda_async_load.CudaMemoryCopier(size, dst.data_ptr(), dst.is_cuda, src.data_ptr(), src.is_cuda, stream.cuda_stream)


def to_check():
    cuda_tensor = make_tensor(True).fill_(42)
    cpu_tensor = make_tensor(False)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):
        _ = torch.empty((2**28,), device=device).fill_(123)  # 消耗时间
        cpu_tensor.copy_(cuda_tensor, non_blocking=True)
    print("Trying to access cuda_tensor[-1] before stream sync:")
    print(cpu_tensor.view((-1,))[-1])  # 可能不是 42.0
    stream.synchronize()
    print("Trying to access cuda_tensor[-1] after stream sync:")
    print(cpu_tensor.view((-1,))[-1])


# to_check()


async def correctness_check_coro():
    loop = asyncio.get_event_loop()
    cuda_tensor = make_tensor(True).fill_(43)
    cpu_tensor = make_tensor(False)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):
        # launch a kernel to target stream to postpone async_tensor_copy
        _ = torch.empty((2**28,), device=device).fill_(123)
    finish = loop.delegate_c_blocking_closure(
        async_tensor_copy(cpu_tensor, cuda_tensor, stream))
    assert cpu_tensor.view((-1,))[-1] != 43
    # print(cpu_tensor.view((-1,))[-1])
    await finish
    assert cpu_tensor.view((-1,))[-1] == 43
    # print(cpu_tensor.view((-1,))[-1])
    print("correctness ok")


# asyncio.run(correctness_check_coro())
# os._exit(-1)


def pre_test():
    torch.cuda.synchronize()
    t1 = make_tensor(True)
    t2 = make_tensor(True)
    for _ in range(repeat_times):
        time_consuming_cuda_operate(t1, t2)
    torch.cuda.synchronize()

    st = time.perf_counter_ns()
    for _ in range(repeat_times):
        time_consuming_cuda_operate(t1, t2)
    torch.cuda.synchronize()
    et = time.perf_counter_ns()
    per_run_device_time = (et-st)/repeat_times
    print(f"operate device time: {per_run_device_time/(10**6):.4f}ms")

    torch.cuda.synchronize()

    st = time.perf_counter_ns()
    for _ in range(repeat_times):
        t1 = t1.to(device=cpu)
        t1 = t1.to(device=device)
    torch.cuda.synchronize()
    et = time.perf_counter_ns()
    per_run_device_time = (et-st)/repeat_times
    print(f"cpy time: {per_run_device_time/(10**6):.4f}ms")
    print("="*80)


pre_test()


async def simple_1_stream_run():
    loop = asyncio.get_event_loop()
    print("simple 1 stream run")
    torch.cuda.synchronize()
    loop.enable_profile(False, False)
    t1 = make_tensor(True)
    t2 = make_tensor(True)
    st = time.perf_counter_ns()
    for _ in range(repeat_times):
        time_consuming_cuda_operate(t1, t2)
        await asyncio.sleep(0)
    torch.cuda.synchronize()
    await asyncio.sleep(0)
    et = time.perf_counter_ns()
    per_run_device_time = (et-st)/repeat_times
    print(
        f"per run device time (no async wait) : {per_run_device_time/(10**6):.4f}ms")
    print(
        f"blocking time (no async wait) :{loop.non_blocking_calltime_ms:.4f}ms")
    loop.clear_profile()
    print("="*80)


async def simple_1_stream_with_async_wait():
    loop = asyncio.get_event_loop()
    print("simple 1 stream with async wait")
    stream = torch.cuda.Stream(device)
    torch.cuda.synchronize()
    loop.enable_profile(False, False)
    with torch.cuda.stream(stream):
        t1 = make_tensor(True)
        t2 = make_tensor(True)
        st = time.perf_counter_ns()
        async_bocking_wait_time: List[int] = []
        for _ in range(repeat_times):
            time_consuming_cuda_operate(t1, t2)
            sync_time = time.time_ns()
            sync_event = cuda_async_load.CudaEventSyncer(stream.cuda_stream)
            async_bocking_wait_time.append((await loop.delegate_c_blocking_closure(sync_event))-sync_time)
        torch.cuda.synchronize()
    et = time.perf_counter_ns()
    per_run_device_time = (et-st)/repeat_times
    async_bocking_wait_ns = sum(
        async_bocking_wait_time)/len(async_bocking_wait_time)
    print(
        f"per run device time (async wait) : {per_run_device_time/(10**6):.4f}ms")
    print(f"blocking time (async wait):{loop.non_blocking_calltime_ms:.4f}ms")
    print(
        f"async blocking time (async wait):{async_bocking_wait_ns/(10**6):.4f}ms")
    loop.clear_profile()
    print("="*80)


async def test_impl(impl: Callable[[], Coroutine[Any, Any, Any]], coros_count: int):
    loop = asyncio.get_event_loop()
    loop.clear_profile()
    tasks = [asyncio.create_task(impl()) for _ in range(coros_count)]
    loop.enable_profile(False, False)
    st = time.perf_counter_ns()
    [(await t) for t in tasks]
    et = time.perf_counter_ns()
    torch.cuda.synchronize()
    per_run_device_time = ((et-st)/(repeat_times))/coros_count
    print(
        f"per run device time (x{coros_count}) : {per_run_device_time/(10**6):.4f}ms")
    print(
        f"blocking time (x{coros_count}):{loop.non_blocking_calltime_ms:.4f}ms")


def make_common_impl():
    async def coro():
        t = make_tensor(False)
        aux = make_tensor(True)
        for _ in range(repeat_times):
            t = t.to(device=device, non_blocking=True)
            time_consuming_cuda_operate(t, aux)
            t.to(device=cpu)
            await asyncio.sleep(0)
    return coro()


def make_stream_impl():
    individual_stream = torch.cuda.Stream(device)
    loop = asyncio.get_event_loop()

    async def coro():
        t = make_tensor(True)
        aux = make_tensor(True)
        for _ in range(repeat_times):
            with torch.cuda.stream(individual_stream):
                t = t.to(device=device, non_blocking=True)
                time_consuming_cuda_operate(t, aux)
                t.to(device=cpu)
                await asyncio.sleep(0)
    return coro()


def make_stream_async():
    individual_stream = torch.cuda.Stream(device)
    loop = asyncio.get_event_loop()

    async def coro():
        t = make_tensor(True)
        aux = make_tensor(True)
        torch.cuda.set_stream(individual_stream)
        for _ in range(repeat_times):
            t = t.to(device=device, non_blocking=True)
            time_consuming_cuda_operate(t, aux)
            cpu_t = make_tensor(False)
            await loop.delegate_c_blocking_closure(async_tensor_copy(cpu_t, t, individual_stream))
            t = cpu_t
    return coro()


async def impls_test():
    coros_count = 2
    print("common impl")
    await test_impl(make_common_impl, coros_count)
    print("="*80)

    print("stream impl")
    await test_impl(make_stream_impl, coros_count)
    print("="*80)

    print("stream async impl")
    await test_impl(make_stream_async, coros_count)
    print("="*80)


async def cuda_coro():
    await simple_1_stream_run()
    await simple_1_stream_with_async_wait()
    await impls_test()

asyncio.run(cuda_coro())
