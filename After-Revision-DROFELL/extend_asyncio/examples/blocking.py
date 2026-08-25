import time
import extend_asyncio as asyncio


asyncio.config_impl("py", False)

disable_blocking = False
blocking_run_count = 0
last_blocking_run_count = 0


def blocking_task():
    global blocking_run_count
    blocking_run_count += 1
    time.sleep(0.5)


async def non_blocking():
    global blocking_run_count, last_blocking_run_count
    assert disable_blocking or last_blocking_run_count == blocking_run_count
    et = time.time()+1
    print("run nonblocking")
    while time.time() < et:
        s = blocking_run_count
        await asyncio.sleep(0)
        assert disable_blocking or s == blocking_run_count
    print("nonblocking sleep")
    s = blocking_run_count
    await asyncio.sleep(0.1)
    print("nonblocking awake")
    assert disable_blocking or (s+1 == blocking_run_count) or (blocking_run_count == s)
    last_blocking_run_count = blocking_run_count

    while time.time() < et:
        s = blocking_run_count
        await asyncio.sleep(0)
        assert disable_blocking or s == blocking_run_count


# async def blocking_task_launcher():
#     global blocking_run_count, last_blocking_run_count
#     loop = asyncio.get_event_loop()
#     for i in range(10):
#         print(f"run blocking: {i}")
#         await loop.run_blocking_soon(blocking_task)
#         print(f"blocking: {i} done")


async def nonblocking_task_launcher():
    global blocking_run_count, last_blocking_run_count
    for i in range(10):
        print("calling non blocking")
        last_blocking_run_count = blocking_run_count
        await non_blocking()
        print("calling non called")
        assert disable_blocking or last_blocking_run_count == blocking_run_count


async def coro():
    asyncio.get_event_loop().set_blocking_scheduler(asyncio.BasicBlockingScheduler())
    asyncio.create_task(nonblocking_task_launcher())
    # asyncio.create_task(blocking_task_launcher())
    await nonblocking_task_launcher()

asyncio.run(coro())
