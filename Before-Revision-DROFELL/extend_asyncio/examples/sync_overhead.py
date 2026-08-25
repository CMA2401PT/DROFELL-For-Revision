import time
from typing import List, cast
import numpy as np
import torch
import extend_asyncio as asyncio
from operators.helpers.cuda_async import CudaEventSyncer


class SyncOverheadMeasure:
    def __init__(self, stream: torch.cuda.Stream) -> None:
        self.stream = stream
        self.device = self.stream.device
        self.tensor_a = torch.rand((2048, 2048), device=self.device)
        self.tensor_b = torch.rand((2048, 2048), device=self.device)
        self.operate_repeat_time = 5
        self.cpu_time, self.device_time = self.compute_actual_device_time()
        # print(f"cpu time: {self.cpu_time/1e6:.4f}ms")
        # print(f"extra device time: {self.extra_device_time/1e6:.4f}ms")

    def time_consuming_cuda_operate(self):
        with torch.no_grad():
            with torch.cuda.stream(self.stream):
                o1 = torch.matmul(self.tensor_a, self.tensor_b)

    def compute_actual_device_time(self):
        self.stream.synchronize()
        st = time.perf_counter_ns()
        while time.perf_counter_ns()-st < 1e9:
            self.time_consuming_cuda_operate()
        self.stream.synchronize()
        with torch.cuda.stream(self.stream):
            st = time.perf_counter_ns()
            operate_count = 0
            while time.perf_counter_ns()-st < 1e9:
                self.time_consuming_cuda_operate()
                operate_count += 1
            self.stream.synchronize()
            et = time.perf_counter_ns()
        t_device = int((et-st)/operate_count)
        launch_times: List[float] = []
        with torch.cuda.stream(self.stream):
            measure_start = time.perf_counter_ns()
            while time.perf_counter_ns()-measure_start < 1e9:
                st = time.perf_counter_ns()
                self.time_consuming_cuda_operate()
                launch_times.append(time.perf_counter_ns()-st)
                self.stream.synchronize()
        t_cpu = int(np.mean(launch_times))
        # with torch.cuda.stream(self.stream):
        #     st = time.perf_counter_ns()
        #     operate_count = 0
        #     while time.perf_counter_ns()-st < 1e9:
        #         self.time_consuming_cuda_operate()
        #         self.stream.synchronize()
        #         operate_count += 1
        #     et = time.perf_counter_ns()
        # t_total = (et-st)/operate_count
        # print(f"t_device: {t_device/1e6:.4f}ms t_cpu: {t_cpu/1e6:.4f}ms t_pure_cpu_sync: {(t_total-t_device)/1e6:.4f}ms ")
        return t_cpu, t_device

    async def test_event_overhead(self, repeat_times: int, load: bool, poll: bool):
        loop = asyncio.get_event_loop()
        if not load:
            with torch.cuda.stream(self.stream):
                results: List[float] = []
                for i in range(repeat_times):
                    st = time.perf_counter_ns()
                    sync_event = CudaEventSyncer(self.stream.cuda_stream)
                    if poll:
                        await loop.delegate_c_poll_closure(sync_event)
                    else:
                        await loop.delegate_c_blocking_closure(sync_event)
                    results.append(time.perf_counter_ns()-st)
                    self.stream.synchronize()
            return np.mean(results[10:])
        else:
            with torch.cuda.stream(self.stream):
                st = time.perf_counter_ns()
                operate_count = 0
                while time.perf_counter_ns()-st < 1e9:
                    self.time_consuming_cuda_operate()
                    sync_event = CudaEventSyncer(self.stream.cuda_stream)
                    if poll:
                        await loop.delegate_c_poll_closure(sync_event)
                    else:
                        await loop.delegate_c_blocking_closure(sync_event)
                    operate_count += 1
                et = time.perf_counter_ns()
            t_total = int((et-st)/operate_count)
            return t_total-self.device_time


if __name__ == "__main__":
    stream = cast(torch.cuda.Stream, torch.cuda.Stream())
    asyncio.config_impl("golang", True)

    async def amain():
        measure = SyncOverheadMeasure(stream)
        measure.compute_actual_device_time()
        overhead = (await measure.test_event_overhead(100, False, False))
        print(f"event_sync blocking overhead: {overhead/1e6:.4f}ms")

        overhead = (await measure.test_event_overhead(100, True, False))
        print(f"event_sync blocking overhead (with load): {overhead/1e6:.4f}ms")

        overhead = (await measure.test_event_overhead(100, False, True))
        print(f"event_sync poll overhead: {overhead/1e6:.4f}ms")

        overhead = (await measure.test_event_overhead(100, True, True))
        print(f"event_sync poll overhead (with load): {overhead/1e6:.4f}ms")

        # event_sync_blocking_overhead = (await measure.test_c_poll_closure_overhead(100))
        # print(f"event_sync poll overhead: {event_sync_blocking_overhead:.4f}ms")
        # event_sync_blocking_overhead = (measure.test_stream_sync_overhead(100))
        # print(f"stream_sync overhead: {event_sync_blocking_overhead:.4f}ms")
    asyncio.run(amain())
