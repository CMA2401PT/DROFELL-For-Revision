"""Probe 1: can we drive DROFELL's own event loop from a virtual clock?"""
import asyncio as std_asyncio
import time
import extend_asyncio as ea
from extend_asyncio.std_asyncio import StdEventLoop


class VirtualClockLoop(StdEventLoop):
    def __init__(self, busy_wait: bool = False) -> None:
        self._vnow = 0.0
        super().__init__(busy_wait=busy_wait)

    def time(self) -> float:
        return self._vnow

    # advance virtual time to the next scheduled callback when nothing is ready
    def _select_common(self):
        if self.ready or self._blocking_scheduler or self._stopping:
            return
        if self.scheduled:
            when = self.scheduled[0]._when
            if when > self._vnow:
                self._vnow = when
        return


def make_loop():
    return VirtualClockLoop(busy_wait=False)


std_asyncio.set_event_loop_policy(ea.EventLoopPolicy(make_loop))
loop = ea.get_event_loop()

order = []

async def worker(name, delays):
    for d in delays:
        await std_asyncio.sleep(d)
        order.append((round(loop.time(), 6), name))

async def main():
    await std_asyncio.gather(
        worker("a", [1.0, 10.0, 100.0]),
        worker("b", [0.5, 0.5, 3600.0]),
        worker("c", [50.0]),
    )
    return "done"

wall0 = time.monotonic()
res = loop.run_until_complete(main())
wall = time.monotonic() - wall0
print("result:", res)
print("event order (virtual_time, who):")
for e in order:
    print("  ", e)
print(f"virtual time elapsed: {loop.time():.3f}s")
print(f"REAL wall clock elapsed: {wall:.4f}s")
