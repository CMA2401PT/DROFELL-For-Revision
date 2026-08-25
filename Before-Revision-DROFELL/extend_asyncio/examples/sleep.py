import time
from typing import List
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import extend_asyncio as asyncio
import os
asyncio.config_impl("golang", True)


async def simple_sleep_coro():
    print("Start")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await asyncio.postpone()
    print("Slept 0 sec")
    await asyncio.sleep(1)
    print("Slept 1 sec")
    await asyncio.sleep(2)
    print("Slept 2 more sec")
asyncio.run(simple_sleep_coro())


async def measure_sleep_coro():

    def plot_sleep_results(df: pd.DataFrame, filename: str):
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")

        sns.pointplot(
            data=df,
            x="sleep_time_ms",
            y="diff_percentage",
            errorbar="sd",
            capsize=0.1,
            join=False,
            markers="o",
            color="b"
        )

        plt.xlabel('Sleep Time (ms)')
        plt.ylabel('Delay Difference (%)')
        plt.title('asyncio.sleep Precision Test (Percentage)')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.savefig(filename)
        print(f"\nPlot saved as {filename}")

    async def measure_once(sleep_time: float, test_time: float):
        diffs_percentage: List[float] = []
        et = time.time()+test_time
        while time.time() < et:
            start = time.perf_counter()
            await asyncio.sleep(sleep_time)
            end = time.perf_counter()
            actual_delay = end - start
            diff_percentage = ((actual_delay - sleep_time) / sleep_time) * 100
            diffs_percentage.append(diff_percentage)
        return diffs_percentage

    loop = asyncio.get_event_loop()
    loop.enable_profile(False, False)

    async def coro():
        sleep_times = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]  # seconds
        test_time = 3
        records = []
        for sleep_time in sleep_times:
            print(f"Testing sleep time: {sleep_time}s")
            diffs_percentage = await measure_once(sleep_time, test_time)
            for dp in diffs_percentage:
                records.append({'sleep_time_ms': sleep_time * 1000, 'diff_percentage': dp})
        print(f"blocking call time: {loop.blocking_calltime_ms:.4f}ms")
        print(f"non blocking call time: {loop.non_blocking_calltime_ms:.4f}ms")
        df_results = pd.DataFrame(records)
        plot_sleep_results(df_results, os.path.join(os.path.dirname(__file__), "sleep.png"))
    await coro()

asyncio.run(measure_sleep_coro())
