import time
from typing import Any


class PerfProb:
    def __init__(self) -> None:
        self.accumulate_time = 0
        self.count = 0
        self.start_time = 0

    def clear(self):
        self.accumulate_time = 0
        self.count = 0
        self.start_time = 0
        print()

    def __enter__(self):
        self.start_time = time.time_ns()

    def __exit__(self, type: Any, value: Any, tb: Any):
        self.accumulate_time += time.time_ns()-self.start_time
        self.count += 1
        print(f"per scope time: {(self.accumulate_time/self.count)/1e6:.4f}ms", end="\r")


perf_prob = PerfProb()
