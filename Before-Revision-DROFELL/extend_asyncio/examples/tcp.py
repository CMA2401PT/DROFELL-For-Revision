import time
from typing import Dict, List
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import os
import extend_asyncio as asyncio

asyncio.config_impl("golang", True)
FrameConn = asyncio.FrameConn
Frames = asyncio.Frames


async def measure_rtt_coro():
    no_gc = []

    def plot_rtt_results(all_rtts: Dict[str, List[float]]):
        plt.figure(figsize=(10, 5), dpi=330)
        for mode, rtts in all_rtts.items():
            plt.plot(rtts, label=f'{mode.upper()} (avg={sum(rtts)/len(rtts):.3f} ms)')
        plt.xlabel("Round")
        plt.ylabel("RTT (ms)")
        plt.title("RTT Comparison between UNIX and TCP sockets")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "rtt.png"))
        print("Saved rtt.png")

    async def client_coro():
        await asyncio.sleep(0.5)
        handler = await loop.tcp_connect("localhost:8888")
        rtts = []
        frames: Frames = [b"2" * (1 << 3)]
        for i in range(100):
            await asyncio.sleep(0.01)
            start = time.perf_counter()
            handler.send(frames, None)
            frames_bak = await handler.recv()
            # frames = frames_bak
            # print(i, frames_bak)
            end = time.perf_counter()
            rtts.append((end - start) * 1000)  # ms
        print(loop.blocking_calltime_ms)
        print(f"RTTs: min={min(rtts):.3f} ms, max={max(rtts):.3f} ms, avg={sum(rtts)/len(rtts):.3f} ms")
        plot_rtt_results({"10ms": rtts})

    async def on_conn(conn: FrameConn) -> None:
        while True:
            frames = await conn.recv()
            if frames is None:
                break
            conn.send([f for f in frames], None)

    loop = asyncio.get_event_loop()
    loop.enable_profile(False, False)
    await asyncio.create_task(loop.tcp_server(":8888", on_conn))
    await client_coro()
    await asyncio.Future()

asyncio.run(measure_rtt_coro())

# rameConn = asyncio.FrameConn
# Frames = asyncio.Frames


# def client_process():

#     async def coro():
#         loop = asyncio.get_event_loop()
#         await asyncio.sleep(1)
#         frame_conn = await loop.tcp_connect("localhost:7241")

#         while True:
#             frames = await frame_conn.recv()
#             if frames is None:
#                 break
#             frame_conn.send(frames)
#         print("client coro quite")
#     asyncio.run(coro())
#     print("client exit")


# def server_process():
#     print("server start")

#     async def conn(conn: FrameConn):
#         for i in range(3):
#             conn.send([f"message-{i}".encode()])
#             print(await conn.recv())
#             await asyncio.sleep(1)

#     async def coro():
#         loop = asyncio.get_event_loop()
#         await loop.tcp_server("localhost:7241", conn)
#         print("server close!")
#     asyncio.run(coro())
#     print("server exit")


# if __name__ == "__main__":
#     multiprocessing.set_start_method("spawn", force=True)

#     child_conn, parent_conn = multiprocessing.Pipe()

#     pr = multiprocessing.Process(target=client_process, args=tuple())
#     pw = multiprocessing.Process(target=server_process, args=tuple())

#     # del child_conn
#     pr.start()
#     # writer_process(parent_conn)
#     # del parent_conn
#     pw.start()
#     time.sleep(1)
#     parent_conn.close()
#     child_conn.close()
#     pw.join()
#     pr.join()
