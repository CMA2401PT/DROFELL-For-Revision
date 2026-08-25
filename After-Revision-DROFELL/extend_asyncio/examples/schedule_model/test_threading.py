import asyncio
import concurrent.futures
import os
import time
import extend_asyncio as asyncio
asyncio.config_impl("golang", False)


def blocking_task(name: str, duration: float):
    print(f"[{name}] Start blocking task on thread {os.getpid()}")
    time.sleep(duration)
    print(f"[{name}] Finished blocking task")
    return f"{name} done"


async def run_blocking_tasks(pool_size: int):
    # 处理 -1 表示无限的情况
    if pool_size == -1:
        executor = concurrent.futures.ThreadPoolExecutor()
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=pool_size)

    loop = asyncio.get_event_loop()

    # 用 asyncio 提交多个任务到线程池
    tasks = [
        loop.run_in_executor(executor, blocking_task, f"task-{i}", 0.2)
        for i in range(5)
    ]

    results = await asyncio.gather(*tasks)
    for r in results:
        print(f"Result: {r}")

    executor.shutdown(wait=True)

if __name__ == "__main__":
    # 传入线程池大小（-1 表示无限制）
    asyncio.run(run_blocking_tasks(pool_size=3))
