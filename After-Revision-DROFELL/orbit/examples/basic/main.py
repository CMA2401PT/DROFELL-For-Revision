import asyncio
import os
import argparse
import threading
import extend_asyncio as asyncio
from orbit import *
from typing import Callable, Dict


def timer(ctx: Context):
    async def coro(ctx: Context):
        name = ctx.args.name
        ctx.logger.info(f"timer {name} initing...")
        await asyncio.sleep(1)
        ctx.logger.info(f"timer {name} inited")
        ctx.report_ready()
        pid = os.getpid()
        tid = threading.get_ident()
        ctx.logger.info(f"timer={name} START@ Process={pid},Thread={tid} Period={1}")
        i = 0
        while True:
            ctx.logger.info(f"timer={name} RUN@ Process={pid},Thread={tid} Period={1} Count={i}")
            await asyncio.sleep(1)
            i += 1
    asyncio.get_event_loop().create_task(coro(ctx))


def reciver(ctx: Context):
    async def coro(ctx: Context):
        name = ctx.args.name
        ctx.logger.info(f"reciver {name} initing...")
        await asyncio.sleep(1)
        ctx.logger.info(f"reciver {name} inited")
        ctx.report_ready()
        while True:
            src, cmd, data = await ctx.recv()
            ctx.logger.warn(f"recv: {src} {cmd} {data}")
    asyncio.get_event_loop().create_task(coro(ctx))


def drop(ctx: Context):
    async def coro(ctx: Context):
        name = ctx.args.name
        ctx.logger.info(f"dropper {name} initing...")
        await asyncio.sleep(1)
        ctx.logger.info(f"dropper {name} inited")
        ctx.report_ready()
        for i in range(3):
            pid = os.getpid()
            tid = threading.get_ident()
            ctx.logger.info(f"dropper={name} RUN@ Process={pid},Thread={tid} Period={1} Ping={await ctx.query_value('ping')}")
            # await asyncio.sleep(1)
        os._exit(-1)
    asyncio.get_event_loop().create_task(coro(ctx))


def set_query_test(ctx: Context):
    async def coro(ctx: Context):
        name = ctx.args.name
        ctx.logger.info(f"set_query_test {name} initing...")
        await asyncio.sleep(1)
        ctx.logger.info(f"set_query_test {name} inited")
        ctx.report_ready()
        i = 1
        while True:
            ctx.set_value(f"test-{i}", (i, i**2))
            ctx.logger.warn(f"get test-{i} "+str(await ctx.query_value(f"test-{i}")))
            await asyncio.sleep(1)
            i += 1
    asyncio.get_event_loop().create_task(coro(ctx))


functions: Dict[str, Callable[..., None]] = {
    "timer": timer,
    "drop": drop,
    "reciver": reciver,
    "set_query": set_query_test
}


def init_fn():
    def on_except(_: asyncio.Loop, e: Exception):
        raise e
        os._exit(-1)
    asyncio.config_impl("golang", True)
    loop = asyncio.get_event_loop()
    loop.on_except(on_except)
    return loop


def run_orbit():
    launch_pad = LaunchPad(init_fn)

    async def coro():
        logger = FileLogger(logfile="basic.log", terminal=True, append=False)
        orbit = Orbit(launch_pad, functions, logger=logger,  log_level="info")
        await orbit.init(port=7241, satellites=("S1",),)
        await orbit.launch(loc=Location("", 0, 0,  "timer-0-0-2"), fn_name="timer", name="T-Orbit-0-0-2")
        await orbit.launch(loc=Location("", 0, 0,  "timer-0-0-3"), fn_name="timer", name="T-Orbit-0-0-3")
        await orbit.launch(loc=Location("", 0, 0,  "reciver-0-0-4"), fn_name="reciver", name="T-Reciver-0-0-4")
        await orbit.launch(loc=Location("", 0, 1,  "timer-0-1-2"), fn_name="timer", name="T-Orbit-0-1-2")
        await orbit.launch(loc=Location("", 0, 1,  "timer-0-1-3"), fn_name="timer", name="T-Orbit-0-1-3")
        await orbit.launch(loc=Location("", 1, 0,  "timer-1-0-2"), fn_name="timer", name="T-Orbit-1-0-2")
        await orbit.launch(loc=Location("", 1, 0,  "timer-1-0-3"), fn_name="timer", name="T-Orbit-1-0-3")
        await orbit.launch(loc=Location("", 1, 1,  "timer-1-1-2"), fn_name="timer", name="T-Orbit-1-1-2")
        await orbit.launch(loc=Location("", 1, 1,  "timer-1-1-3"), fn_name="timer", name="T-Orbit-1-1-3")
        await orbit.launch(loc=Location("S1", 0, 0,  "S1-timer-0-0-2"), fn_name="timer", name="T-S1-0-0-2")
        await orbit.launch(loc=Location("S1", 0, 0,  "S1-timer-0-0-3"), fn_name="timer", name="T-S1-0-0-3")
        await orbit.launch(loc=Location("S1", 0, 1,  "S1-timer-0-1-2"), fn_name="timer", name="T-S1-0-1-2")
        await orbit.launch(loc=Location("S1", 0, 1,  "S1-timer-0-1-3"), fn_name="timer", name="T-S1-0-1-3")
        await orbit.launch(loc=Location("S1", 0, 1,  "S1-set-test-0-1-4"), fn_name="set_query", name="T-S1-set-test-0-1-4")
        await orbit.launch(loc=Location("S1", 1, 0,  "S1-timer-1-0-2"), fn_name="timer", name="T-S1-1-0-2")
        await orbit.launch(loc=Location("S1", 1, 0,  "S1-timer-1-0-3"), fn_name="timer", name="T-S1-1-0-3")
        await orbit.launch(loc=Location("S1", 1, 1,  "S1-timer-1-1-2"), fn_name="timer", name="T-S1-1-1-2")
        await orbit.launch(loc=Location("S1", 1, 1,  "S1-timer-1-1-3"), fn_name="timer", name="T-S1-1-1-3")
        await orbit.launch(loc=Location("S1", 1, 1,  "S1-reciver-1-1-4"), fn_name="reciver", name="T-S1-Reciver-1-1-4")

        for i in range(3):
            orbit.send("reciver-0-0-4", f"test-{i}", f"hello-Reciver-0-0-4-{i}")
            orbit.send("S1-reciver-1-1-4", f"test-{i}", f"hello-S1-1-1-4-{i}")
            await asyncio.sleep(1)

        await asyncio.sleep(3)
        orbit.drop("S1", 1)
        await asyncio.sleep(3)
        orbit.drop("S1")
        await asyncio.sleep(3)
        orbit.drop(proc=1)
    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


def run_satellite(role: str):
    launch_pad = LaunchPad(init_fn)

    async def coro():
        satellite = Satellite(launch_pad=launch_pad, functions_dict=functions, host_id=role, network_hostname="localhost")
        await satellite.run("localhost:7241")

    launch_pad.loop.create_task(coro())
    launch_pad.loop.run_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", "-r", type=str, default=OrbitName)
    args, _ = parser.parse_known_args()
    role = args.role
    if role == OrbitName:
        run_orbit()
    else:
        run_satellite(role)
