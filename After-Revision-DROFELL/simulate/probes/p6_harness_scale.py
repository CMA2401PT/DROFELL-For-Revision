"""Probe 6: how many logical nodes can one virtual-clock loop carry?"""
import asyncio as std_asyncio, time, resource
import extend_asyncio as ea
from extend_asyncio.std_asyncio import StdEventLoop

class VClock(StdEventLoop):
    def __init__(self, busy_wait=False):
        self._vnow = 0.0
        super().__init__(busy_wait=busy_wait)
    def time(self): return self._vnow
    def _select_common(self):
        if self.ready or self._blocking_scheduler or self._stopping: return
        if self.scheduled:
            w = self.scheduled[0]._when
            if w > self._vnow: self._vnow = w

std_asyncio.set_event_loop_policy(ea.EventLoopPolicy(lambda: VClock()))
loop = ea.get_event_loop()

N_NODES = 256
MSGS = 40
inbox = {i: std_asyncio.Queue() for i in range(N_NODES)}
delivered = 0

def send(dst, payload, delay):
    def deliver():
        inbox[dst].put_nowait(payload)
    loop.call_later(delay, deliver)   # link latency in virtual time

async def node(i):
    global delivered
    for _ in range(MSGS):
        p = await inbox[i].get()
        delivered += 1
        nxt = (i + 1) % N_NODES
        send(nxt, p, 0.05)            # 50 ms link
        await std_asyncio.sleep(0.007)  # 7 ms service time

async def main():
    tasks = [std_asyncio.ensure_future(node(i)) for i in range(N_NODES)]
    for i in range(N_NODES):
        send(i, f"tok-{i}", 0.0)
    await std_asyncio.gather(*tasks)

w0 = time.monotonic()
loop.run_until_complete(main())
w = time.monotonic() - w0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"nodes={N_NODES} msgs delivered={delivered}")
print(f"virtual time simulated: {loop.time():.2f}s")
print(f"REAL wall clock: {w:.3f}s   speedup={loop.time()/max(w,1e-9):.0f}x")
print(f"peak RSS: {rss:.0f} MiB")
