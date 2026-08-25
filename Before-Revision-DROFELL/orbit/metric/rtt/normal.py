# filename: measure_ttl.py
import asyncio
import os
from os import path
import sys
import time
import matplotlib.pyplot as plt
from typing import Literal

sys.path.append(path.abspath(path.join(path.dirname(__file__), "..", "..", "..")))
print(sys.path)
try:
    from async_extend.loop.loop_extend import ensure_loop
except Exception as e:
    raise e

ensure_loop()

N = 100  # ping-pong 次数
SOCKET_FILE = "/tmp/test_unix_socket"
TCP_PORT = 8889

# RTT 存储容器
all_rtts = {}


async def handle_echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    while True:
        data = await reader.readexactly(4)
        writer.write(data)
        # await writer.drain()


async def unix_server():
    if os.path.exists(SOCKET_FILE):
        os.remove(SOCKET_FILE)
    server = await asyncio.start_unix_server(handle_echo, path=SOCKET_FILE)
    async with server:
        await server.serve_forever()


async def tcp_server():
    server = await asyncio.start_server(handle_echo, "127.0.0.1", TCP_PORT)
    async with server:
        await server.serve_forever()


async def measure_client(mode: Literal["unix", "tcp"]):
    if mode == "unix":
        reader, writer = await asyncio.open_unix_connection(SOCKET_FILE)
    else:
        reader, writer = await asyncio.open_connection("127.0.0.1", TCP_PORT)
        # writer.get_extra_info('socket').setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    rtts = []
    for _ in range(N):
        msg = b"ping"
        st = time.time()
        while time.time()-st < 0.01:
            pass
        start = time.perf_counter()
        writer.write(msg)
        await reader.readexactly(len(msg))
        end = time.perf_counter()
        rtts.append((end - start) * 1000)  # ms

    writer.close()
    await writer.wait_closed()

    all_rtts[mode] = rtts
    print(f"{mode.upper()} RTTs: min={min(rtts):.3f} ms, max={max(rtts):.3f} ms, avg={sum(rtts)/len(rtts):.3f} ms")


async def spawn_server_client(mode: Literal["unix", "tcp"]):
    server = await asyncio.create_subprocess_exec(sys.executable, __file__, f"{mode}_server")
    await asyncio.sleep(0.5)  # 等待服务器准备好
    await measure_client(mode)
    server.terminate()
    await server.wait()


def plot_results():
    plt.figure(figsize=(10, 5), dpi=330)
    for mode, rtts in all_rtts.items():
        plt.plot(rtts, label=f'{mode.upper()} (avg={sum(rtts)/len(rtts):.3f} ms)')
    plt.xlabel("Round")
    plt.ylabel("RTT (ms)")
    plt.title("RTT Comparison between UNIX and TCP sockets")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "normal.png"))
    print("Saved normal.png")


async def main():
    print("\nMeasuring TCP socket...")
    await spawn_server_client("tcp")
    await asyncio.sleep(0.5)
    print("Measuring UNIX socket...")
    await spawn_server_client("unix")

    plot_results()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        asyncio.run(main())
    elif sys.argv[1] == "unix_server":
        asyncio.run(unix_server())
    elif sys.argv[1] == "tcp_server":
        asyncio.run(tcp_server())
