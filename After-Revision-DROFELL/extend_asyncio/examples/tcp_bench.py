"""Loopback TCP round-trip latency micro-benchmark for async_rt.

Isolates the "idle process wake-up" effect that motivates `busy_wait` (no CUDA
involved). See docs/design/extend-asyncio-cuda-async.md §6 "busy_wait 为什么存在".

Usage (one process per combo — config_impl is once-per-process):
    PYTHONPATH=<repo> python async_rt/bench/tcp_rtt.py <impl> <busy> <sleep>
      impl  : golang | py
      busy  : 1 | 0   (busy_wait: spin the selector vs block)
      sleep : 1 | 0   (whether `await asyncio.sleep(0.01)` precedes each round)

Reproduce the full matrix on a target machine:
    for a in "golang 1 1" "golang 1 0" "golang 0 1" "golang 0 0" "py 0 1" "py 0 0"; do
        python async_rt/bench/tcp_rtt.py $a
    done

Finding (this dev box): busy_wait is the dominant knob (~5-6x). `sleep` only
inflates RTT when busy_wait is OFF — the 10ms idle window lets the process/CPU
deschedule, so the next recv pays a cold wake-up cost. busy_wait pins the
process running, so sleep stops mattering. sleep is the trigger, not the cost.
"""
import sys
import time
import os
import extend_asyncio as asyncio

from ..common import FrameConn

impl = sys.argv[1] if len(sys.argv) > 1 else "golang"
busy = (sys.argv[2] if len(sys.argv) > 2 else "1") == "1"
do_sleep = (sys.argv[3] if len(sys.argv) > 3 else "1") == "1"

asyncio.config_impl(impl, busy)


async def main():
    loop = asyncio.get_event_loop()
    loop.enable_profile(False, False)

    async def on_conn(conn: FrameConn) -> None:
        while True:
            frames = await conn.recv()
            if frames is None:
                break
            conn.send([f for f in frames], None)

    async def client_coro():
        await asyncio.sleep(1)
        handler = await loop.tcp_connect("localhost:8888")
        rtts = []
        frames = [b"2" * (1 << 3)]
        for _ in range(100):
            if do_sleep:
                await asyncio.sleep(0.01)
            start = time.perf_counter()
            handler.send(frames, None)
            await handler.recv()
            rtts.append((time.perf_counter() - start) * 1000)
        print(f"impl={impl:6s} busy={int(busy)} sleep={int(do_sleep)}  "
              f"min={min(rtts):.3f} max={max(rtts):.3f} "
              f"avg={sum(rtts)/len(rtts):.3f} ms", flush=True)
        os._exit(0)

    await asyncio.create_task(loop.tcp_server(":8888", on_conn))
    await client_coro()
    await asyncio.Future()

asyncio.run(main())
