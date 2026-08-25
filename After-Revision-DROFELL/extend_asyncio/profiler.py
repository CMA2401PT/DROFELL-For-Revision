import cProfile
import os
import time
from typing import Optional


class MixtureProfiler:
    def __init__(self) -> None:
        self.cProf = cProfile.Profile()
        self.run_total_time: int = 0
        self.run_total_count: int = 0
        self._enable = False
        self.last_start = 0

    def enable(self, enable: bool):
        self._enable = enable

    def __enter__(self):
        self.last_start = time.time_ns()
        if self._enable:
            self.cProf.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):  # type:ignore
        if self.last_start == 0:
            # clear is called before exit
            return
        self.run_total_time += time.time_ns()-self.last_start
        self.run_total_count += 1
        if self._enable:
            self.cProf.disable()

    @property
    def blocking_time_ms(self) -> float:
        if self.run_total_count == 0:
            return 0
        return (self.run_total_time/self.run_total_count)/1e6

    def clear_profile(self):
        self.run_total_time: int = 0
        self.run_total_count: int = 0
        self._enable = False
        self.last_start = 0
        self.cProf = cProfile.Profile()

    def dump_profile(self, path: Optional[str] = None):
        import seaborn as sns
        import matplotlib.pyplot as plt
        if path is None:
            path = "profile_output"
        os.makedirs(path, exist_ok=True)
        print(f"avg runtime ms: {self.blocking_time_ms:.4f}")
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(8, 6))
        # sns.histplot(run_time_ms, kde=True, stat="probability", bins=30, color="skyblue")
        # 添加 rug plot（每个数据点的短竖线）
        # sns.rugplot(self.run_times, color='black', height=0.05)
        # 添加标题和标签
        plt.title("Blocking Time")
        plt.xlabel("Time (ms)")
        plt.ylabel("Probability")
        plt.savefig(os.path.join(path, "blocking_time.png"), dpi=300)
        self.cProf.dump_stats(os.path.join(path, "cProf.profile"))
