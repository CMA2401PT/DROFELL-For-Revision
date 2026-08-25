import os
import time
from typing import Any, Dict
import extend_asyncio as asyncio

asyncio.config_impl("golang", True)


async def exception():
    async def coro():
        raise ValueError("test")

    def on_except(loop: asyncio.Loop, exce: Any):
        print(f"handle exception: {exce}")
    asyncio.get_event_loop().on_except(on_except)
    await asyncio.create_task(coro())
    await asyncio.Future()
# asyncio.run(exception())
# os._exit(-1)


async def event_test():
    async def sender(ev: asyncio.Event):
        print("sender start")
        await asyncio.sleep(1)
        print("begin event set")
        ev.set()
        print("event set")

    async def reciver(ev: asyncio.Event):
        print("reciver waiting")
        await ev.wait()
        print("reciver recived event")

    ev = asyncio.Event()
    asyncio.get_event_loop().create_task(reciver(ev))
    asyncio.get_event_loop().create_task(reciver(ev))
    asyncio.get_event_loop().create_task(sender(ev))
    await asyncio.Future()

# asyncio.run(event_test())
# os._exit(0)


async def gather_test():
    async def sub(i: int):
        print(f"sub {i} start")
        await asyncio.sleep(1)
        print(f"sub {i} done")

    tasks = [
        asyncio.create_task(sub(i))
        for i in range(2, 5)
    ]

    await asyncio.gather(sub(0), sub(1), *tasks)
    print("all done")
asyncio.run(gather_test())
os._exit(0)


async def gather_event_wait_for_test():
    events: Dict[int, asyncio.Event] = {}

    async def sub(event_i: int):
        print("sub start")
        await asyncio.sleep(3)
        print("sub done")
        events[event_i].set()

    events[0] = asyncio.Event()
    asyncio.create_task(sub(0))
    gathered = asyncio.gather(*(e.wait() for e in events.values()))
    try:
        await asyncio.wait_for(gathered, timeout=1)
        print("wait complete")
    except Exception as e:
        print("exception:", e)
    await asyncio.Future()
asyncio.run(gather_event_wait_for_test())
os._exit(-1)


async def wait_for_test():
    async def sub():
        print("sub start")
        await asyncio.sleep(3)
        print("sub done")
        return "done"
    ret = await asyncio.wait_for(sub(), 1)
    print(ret)


# async def gather_event_wait_for_test():
#     events: Dict[int, asyncio.Event] = {}

#     async def sub(event_i: int):
#         print("sub start")
#         await asyncio.sleep(3)
#         print("sub done")
#         events[event_i].set()

#     events[0] = asyncio.Event()
#     asyncio.create_task(sub(0))
#     gathered = asyncio.gather(*(e.wait() for e in events.values()))
#     await asyncio.wait_for(gathered, timeout=5)
#     print("wait complete")
# asyncio.run(gather_event_wait_for_test())
# os._exit(-1)


async def wait_first_test():
    async def sub_1():
        await asyncio.sleep(1)
        print("sub 1 complete")

    async def sub_3():
        await asyncio.sleep(3)
        print("sub 3 complete")

    _1 = asyncio.create_task(sub_1())
    _3 = asyncio.create_task(sub_3())
    done, pending = await asyncio.wait_first((_1, _3))
    assert _1 in done
    assert _3 in pending
    print("check ok, but sub will still running")
    await asyncio.Future()
# asyncio.run(wait_first_test())


async def generators_test():
    async def sub(s: int, i: int):
        await asyncio.sleep(1)
        return (s, i)

    running_tasks = (
        asyncio.create_task(sub(0, 0)),
        asyncio.create_task(sub(1, 1))
    )

    while len(running_tasks):
        done, pending = await asyncio.wait_first(running_tasks)
        next_tasks = list(pending)
        for d in done:
            s, i = d.result()
            print(s, i)
            if i > 3:
                continue
            next_tasks.append(asyncio.create_task(sub(s, i+1)))
        running_tasks = tuple(next_tasks)

    await asyncio.Future()
asyncio.run(generators_test())


async def call_later_test():
    st = time.time()

    await asyncio.sleep_until(time.time()+3)
    print(f"wake at: {time.time()-st}")
    st = time.time()

    def _sub_1(tag: str):
        print(f"call after: {time.time()-st}: {tag}")
    asyncio.get_event_loop().call_later(0, _sub_1, ("tag1",))
    await asyncio.Future()
asyncio.run(call_ater_test())
# async def sub_5():
#     await asyncio.sleep(5)
#     print("sub 5 complete")

# asyncio.run(sub_5())
