import multiprocessing
import os
import threading
import traceback
import extend_asyncio as asyncio
from orbit import *
import gc
import builtins


def init_fn():
    multiprocessing.set_start_method('spawn', force=True)
    original_delattr = builtins.delattr

    def _force_gc_after_delattr(obj: object, name: str):
        original_delattr(obj, name)
        gc.collect()

    builtins.delattr = _force_gc_after_delattr

    def on_except(_: asyncio.Loop, e: Exception):
        print(e)
        traceback.print_exc()
        os._exit(-1)
    asyncio.config_impl("py", True)
    loop = asyncio.get_event_loop()
    loop.on_except(on_except)
    return loop


def timer(ctx: Context):
    async def coro():
        name = ctx.args.name
        ctx.logger.info(f"timer {name} initing...")
        await asyncio.sleep(1)
        ctx.logger.info(f"timer {name} inited")
        ctx.report_ready()
        pid = os.getpid()
        tid = threading.get_ident()
        ctx.logger.info(f"timer={name} START@ Process={pid},Thread={tid} Period={1} Ping={await ctx.query_value('ping')}")
        i = 0
        while True:
            ctx.logger.info(f"timer={name} RUN@ Process={pid},Thread={tid} Period={1} Ping={await ctx.query_value('ping')} Count={i}")
            await asyncio.sleep(1)
            i += 1
    asyncio.get_event_loop().create_task(coro())


def reciver(ctx: Context):
    async def coro():
        name = ctx.args.name
        ctx.logger.info(f"reciver {name} initing...")
        await asyncio.sleep(1)
        ctx.logger.info(f"reciver {name} inited")
        ctx.report_ready()
        while True:
            src, cmd, data = await ctx.recv()
            ctx.logger.warn(f"recv: {src} {cmd} {data}")
    asyncio.get_event_loop().create_task(coro())


if __name__ == "__main__":
    launch_pad = LaunchPad(init_fn)
    logger = SimpleLogger()
    cmder = Orbit(
        launch_pad,
        {"timer": timer, "reciver": reciver},
        logger
    )

    async def amain():
        await cmder.init(2401, None, None)
        print("init completed")
        await cmder.launch(loc=Location("", 0, 0, "timer-0-0-2"), fn_name="timer", name="T-Orbit-0-0-2")
        await cmder.launch(loc=Location("", 0, 0, "timer-0-0-3"), fn_name="timer", name="T-Orbit-0-0-3")
        await cmder.launch(loc=Location("", 0, 0, "reciver-0-0-4"), fn_name="reciver", name="T-Reciver-0-0-4")
        await cmder.launch(loc=Location("", 0, 1, "timer-0-1-2"), fn_name="timer", name="T-Orbit-0-1-2")
        await cmder.launch(loc=Location("", 0, 1, "timer-0-1-3"), fn_name="timer", name="T-Orbit-0-1-3")
        await cmder.launch(loc=Location("", 1, 0, "timer-1-0-2"), fn_name="timer", name="T-Orbit-1-0-2")
        await cmder.launch(loc=Location("", 1, 0, "timer-1-0-3"), fn_name="timer", name="T-Orbit-1-0-3")
        await cmder.launch(loc=Location("", 1, 1, "timer-1-1-2"), fn_name="timer", name="T-Orbit-1-1-2")
        await cmder.launch(loc=Location("", 1, 1, "timer-1-1-3"), fn_name="timer", name="T-Orbit-1-1-3")

        for i in range(3):
            cmder.send("reciver-0-0-4", f"test-{i}", f"hello-Reciver-0-0-4-{i}")
            # cmder.send("S1-reciver-1-1-4", f"test-{i}", f"hello-S1-1-1-4-{i}")
            await asyncio.sleep(1)
        await asyncio.sleep(3)
        cmder.drop(proc=1)
    launch_pad.loop.create_task(amain())
    launch_pad.loop.run_forever()
