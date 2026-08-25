# type:ignore
import time
import extend_asyncio as asyncio


asyncio.config_impl("py", False)


async def subblocking(tag: str = ""):
    print(f"blocking {tag} staring")
    for i in range(6):
        print(f"blocking {tag} run: {i}")
        assert asyncio.get_event_loop().is_running_blocking
        await asyncio.sleep(0.1)
        assert asyncio.get_event_loop().is_running_blocking
        await asyncio.sleep(0)
    assert asyncio.get_event_loop().is_running_blocking
    print(f"blocking {tag} finish")


async def blocking(tag: str = ""):
    print(f"blocking {tag} staring")
    asyncio.create_task(coro=subblocking(tag))
    for i in range(6):
        print(f"blocking {tag} run: {i}")
        assert asyncio.get_event_loop().is_running_blocking
        await asyncio.sleep(0.1)
        assert asyncio.get_event_loop().is_running_blocking
        await asyncio.sleep(0)
    assert asyncio.get_event_loop().is_running_blocking
    print(f"blocking {tag} finish")


async def sub_non_blocking(tag: str = ""):
    print(f"non blocking {tag} staring")
    for i in range(3):
        print(f"non blocking {tag} run: {i}")
        assert not asyncio.get_event_loop().is_running_blocking
        await asyncio.sleep(timeout=0.2)
        et = time.time()+0.2
        print(f"non blocking {tag} HOLDING: {i}")
        loop = asyncio.get_event_loop()
        while time.time() < et:
            assert not loop.is_running_blocking
            await asyncio.sleep(0)
        print(f"non blocking {tag} UNHOLD: {i}")
    assert not asyncio.get_event_loop().is_running_blocking
    print(f"non blocking {tag} finish")


async def non_blocking(tag: str = ""):
    print(f"non blocking {tag} staring")
    asyncio.create_task(sub_non_blocking())
    for i in range(3):
        print(f"non blocking {tag} run: {i}")
        assert not asyncio.get_event_loop().is_running_blocking
        await asyncio.sleep(timeout=0.2)
        assert not asyncio.get_event_loop().is_running_blocking
        et = time.time()+0.2
        print(f"non blocking {tag} HOLDING: {i}")
        loop = asyncio.get_event_loop()
        while time.time() < et:
            assert not loop.is_running_blocking
            await asyncio.sleep(0)
        print(f"non blocking {tag} UNHOLD: {i}")
    assert not asyncio.get_event_loop().is_running_blocking
    print(f"non blocking {tag} finish")


async def blocking_caller():
    assert not asyncio.get_event_loop().is_running_blocking
    async with asyncio.get_event_loop().run_as_blocking():
        await blocking()
        asyncio.create_task(blocking())
    asyncio.create_task(non_blocking())
    assert not asyncio.get_event_loop().is_running_blocking

blocking_complete = False


async def simple_non_blocking():
    et = time.time()+2
    while time.time() < et:
        assert not asyncio.get_event_loop().is_running_blocking
        await asyncio.sleep(0)
    assert not blocking_complete


async def simple_blocking():
    global blocking_complete
    async with asyncio.get_event_loop().run_as_blocking():
        et = time.time()+1
        while time.time() < et:
            assert asyncio.get_event_loop().is_running_blocking
            await asyncio.sleep(0)
        blocking_complete = True


async def coro():
    asyncio.create_task(simple_non_blocking())
    await simple_blocking()
    asyncio.create_task(non_blocking())
    asyncio.create_task(blocking_caller())
    await non_blocking("")
    await asyncio.Future()


asyncio.run(coro())
