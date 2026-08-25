# type:ignore
import asyncio
import collections
import heapq
import os
import struct
import time
import traceback
from types import CoroutineType
from typing import Dict, List, Optional, cast, Generator
from uuid import uuid4
from .common import *
from .profiler import MixtureProfiler
Handle = asyncio.events.Handle


class ConnHandler(FrameConn):
    def __init__(self, writer: "LengthPrefixedFrameWriter") -> None:
        self.writer = writer
        self._closed = False
        self.close_cb = lambda: None
        self.recv_queue: asyncio.Queue[Optional[Frames]] = asyncio.Queue()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.close_cb()
        except Exception as e:
            pass

    def __del__(self):
        self.close()

    def send(self, frames: Frames, on_release_callback: Optional[Callable[[], None]]):
        fs = [bytes(f) for f in frames]
        self.writer.write_frames(fs, on_release_callback)

    def put(self, frames: Optional[Frames]):
        self.recv_queue.put_nowait(frames)

    async def recv(self) -> Optional[Frames]:
        return await self.recv_queue.get()


class QueueWithIntercept:
    def __init__(self, base_queue: collections.deque[Handle], intercept: Callable[[Handle], bool]) -> None:
        self.base_queue = base_queue
        self.intercept = intercept

    def __len__(self):
        return len(self.base_queue)

    def popleft(self):
        return self.base_queue.popleft()

    def clear(self):
        self.base_queue.clear()

    def append(self, x: Handle):
        if not self.intercept(x):
            self.base_queue.append(x)


class _BlockingTaskContainer(BlockingTaskContainer):
    def __init__(self, loop: "StdEventLoop") -> None:
        self.loop = loop

    async def __aenter__(self):
        self.loop.mark_blocking_flag = True
        await asyncio.sleep(0)

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        self.loop.mark_blocking_flag = False
        self.loop.runing_blocking = False
        self.loop._scheduled = self.loop._nonblocking_scheduled
        print("mark as non blocking")
        return False


class StdEventLoop(asyncio.SelectorEventLoop, Loop):  # type:ignore
    def __init__(self, busy_wait: bool) -> None:
        super().__init__()
        self._scheduled: List[Handle] = self._scheduled  # type:ignore
        self._blocking_scheduled: List[Handle] = []
        self._nonblocking_scheduled: List[Handle] = self._scheduled
        self._ready: collections.deque[Handle] = QueueWithIntercept(self._ready, self._intercept)  # type:ignore
        self._blocking: collections.deque[Handle] = collections.deque()
        self.mark_blocking_flag = False
        self.runing_blocking = False
        self._clock_resolution = time.get_clock_info('monotonic').resolution
        self.profiler = MixtureProfiler()
        self.profiler.enable(enable=False)
        self.busy_wait = busy_wait
        self.no_gc: Dict[str, Any] = {}
        self.running = False

        def exit_on_except(loop: Loop, e: Any):
            if hasattr(e, "__traceback__"):
                print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            print(e)
            os._exit(-1)

        self.on_except(exit_on_except)

    def _intercept(self, handle: Handle) -> bool:
        # print(f"mark blocking: {self._mark_blocking_flag}, running blocking: {self.is_running_blocking}")
        if (not self.mark_blocking_flag) and (not self.runing_blocking):
            return False
        self._blocking.append(handle)
        # print(len(self._blocking))
        self.mark_blocking_flag = False
        return True

    @property
    def is_running_blocking(self) -> bool:
        return self.runing_blocking

    def run_as_blocking(self):
        return _BlockingTaskContainer(self)

    def _switch_blocking(self):
        self.runing_blocking = True
        assert not self._ready
        self._scheduled = self._blocking_scheduled

    def _unswitch_blocking(self):
        if self.runing_blocking:
            assert not self._ready
        self.runing_blocking = False
        self._scheduled = self._nonblocking_scheduled

    def _run_handles(self):
        end_time = self.time() + self._clock_resolution
        while self._nonblocking_scheduled:
            handle = self._nonblocking_scheduled[0]
            if handle._when >= end_time:  # type:ignore
                break
            handle = heapq.heappop(self._nonblocking_scheduled)
            handle._scheduled = False  # type:ignore
            self._ready.append(handle)

        while self._blocking_scheduled:
            handle = self._blocking_scheduled[0]
            if handle._when >= end_time:  # type:ignore
                break
            handle = heapq.heappop(self._blocking_scheduled)
            handle._scheduled = False  # type:ignore
            self._blocking.append(handle)
        ntodo = len(self._ready)
        if ntodo != 0:
            for _ in range(ntodo):
                handle = self._ready.popleft()
                if handle._cancelled:
                    continue
                else:
                    with self.profiler:
                        handle._run()
        elif self._blocking:
            blocking_handle = self._blocking.popleft()
            if not blocking_handle._cancelled:
                with self.profiler:
                    self._switch_blocking()
                    blocking_handle._run()
                    self._unswitch_blocking()
        handle = None

    def on_except(self, callback: Callable[[Loop, Any], None]) -> None:
        def exception_handler(loop: asyncio.AbstractEventLoop, context: Dict[str, Any]):
            callback(self, context['exception'] if ('exception' in context) else Exception(context))
        self.set_exception_handler(exception_handler)

    def enable_profile(self, enable: bool) -> None:
        self.profiler.enable(enable)

    @property
    def blocking_time_ms(self) -> float:
        return self.profiler.blocking_time_ms

    def clear_profile(self) -> None:
        self.profiler.clear_profile()

    def dump_profile(self, path: Optional[str] = None):
        self.profiler.dump_profile(path)

    def _remove_cancelled(self):
        # Remove delayed calls that were cancelled from head of queue.
        while self._nonblocking_scheduled and self._nonblocking_scheduled[0]._cancelled:
            self._timer_cancelled_count -= 1
            handle = heapq.heappop(self._nonblocking_scheduled)
            handle._scheduled = False  # type:ignore

        while self._blocking_scheduled and self._blocking_scheduled[0]._cancelled:
            self._timer_cancelled_count -= 1
            handle = heapq.heappop(self._blocking_scheduled)
            handle._scheduled = False  # type:ignore

    def _compute_wait_time(self) -> Optional[float]:
        # get wait timeout for io
        if self._ready or self._blocking or cast(bool, self._stopping):  # type:ignore
            return 0
        elif self._nonblocking_scheduled or self._blocking_scheduled:
            # Compute the desired timeout.
            when = 0
            if self._nonblocking_scheduled:
                when = self._nonblocking_scheduled[0]._when  # type:ignore
            if self._blocking_scheduled:
                _blocking_when = self._blocking_scheduled[0]._when  # type:ignore
                if when == 0 or (when > _blocking_when):
                    when = _blocking_when
            return max(0, when - self.time())
        else:
            return None

    def _select_common(self):
        timeout = self._compute_wait_time()
        event_list = self._selector.select(timeout)  # type:ignore
        self._process_events(event_list)  # type:ignore

    def _select_busy_wait(self):
        timeout = self._compute_wait_time()
        when = None if (timeout is None) else time.time()+timeout

        if when is None or timeout != 0:
            while (when is None) or self.time() < when:
                event_list = self._selector.select(0)  # type:ignore
                if event_list:
                    self._process_events(event_list)  # type:ignore
                    break

    def _run_once(self):
        self._remove_cancelled()

        if not self.busy_wait:
            self._select_common()
        else:
            self._select_busy_wait()

        self._run_handles()

    async def tcp_connect(self, addr: str) -> FrameConn:
        host, port = addr.split(":")
        stream_reader, stream_writer = await asyncio.open_connection(host, port)
        frame_writer = LengthPrefixedFrameWriter(stream_writer)
        frame_reader = LengthPrefixedFrameReader(stream_reader)

        h = ConnHandler(frame_writer)
        h.close_cb = stream_writer.close
        uuid_str = uuid4().hex

        async def recv_loop():
            while True:
                try:
                    fs = cast(Frames, await frame_reader.read_frames())
                    h.put(fs)
                except Exception as e:
                    del self.no_gc[uuid_str]
                    h.put(None)
                    h.close()
                    return

        self.no_gc[uuid_str] = asyncio.create_task(recv_loop())

        return h

    async def tcp_server(self, addr: str, on_conn: Callable[[FrameConn], Coroutine[Any, Any, None]]):
        host, port = addr.split(":")

        async def _on_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            fwriter = LengthPrefixedFrameWriter(writer)
            freader = LengthPrefixedFrameReader(reader)
            close_fut = asyncio.Future()

            h = ConnHandler(fwriter)
            h.close_cb = lambda: close_fut.set_result(None)

            async def _recv_loop():
                while True:
                    try:
                        frames = cast(Frames, await freader.read_frames())
                        h.put(frames)
                    except Exception as e:
                        h.put(None)
                        return
            asyncio.create_task(_recv_loop())
            asyncio.create_task(on_conn(h))
            await close_fut

        return await asyncio.start_server(_on_conn, host, port)

    def pipe_reader(self, read_conn: MultiProcessConn) -> FrameConnReader:
        return FramePipeReader(read_conn, self)

    def pipe_writer(self, write_conn: MultiProcessConn) -> FrameConnWriter:
        return FramePipeWriter(write_conn)

    def delegate_c_blocking_closure(self, load: BlockingLoad) -> Any:
        fut = asyncio.Future()
        load.run_blocking_delegate()
        fut.set_result(load.retrieve_result())
        return fut

    # async def delegate_py_blocking_closure(self, load: BlockingLoad, with_gil: bool) -> Any:
    #     load.run_blocking_delegate()
    #     return load.retrieve_result()

    # def torch_event_sync(self) -> Awaitable[None]: raise NotImplementedError

# def new_event_loop() -> StdEventLoop:
#     return StdEventLoop()


# _thread_local = threading.local()


# def set_event_loop(loop: StdEventLoop):
#     eloop = getattr(_thread_local, 'loop', None)
#     if eloop is not None:
#         raise ValueError("loop already set")
#     _thread_local.loop = loop


# def get_running_loop() -> StdEventLoop:
#     loop = getattr(_thread_local, 'loop', None)
#     if loop is None:
#         raise ValueError("no current loop")
#     return loop


# def get_event_loop() -> StdEventLoop:
#     loop = getattr(_thread_local, 'loop', None)
#     if loop is None:
#         loop = new_event_loop()
#         set_event_loop(loop)
#     return loop


# gather = asyncio.gather
# Queue = asyncio.Queue
# Future = asyncio.Future
# sleep = asyncio.sleep


# async def sleep_until(wake_time: float):
#     return await asyncio.sleep(wake_time-time.time())


class LengthPrefixedFrameWriter:
    def __init__(self, stream_writer: asyncio.StreamWriter) -> None:
        self._writer = stream_writer
        self._pack = struct.Struct('>I')  # 4-byte big-endian length

    def write_frames(self, frames: List[bytes], on_release_callback: Optional[Callable[[], None]]) -> None:
        total_len = sum([len(f)+4 for f in frames])+4
        header = bytearray()
        header += self._pack.pack(total_len)
        header += self._pack.pack(len(frames))
        for data in frames:
            header += self._pack.pack(len(data))
        self._writer.write(header)
        for frame in frames:
            self._writer.write(bytes(frame))
        if on_release_callback:
            on_release_callback()


class LengthPrefixedFrameReader:
    def __init__(self, stream_reader: asyncio.StreamReader) -> None:
        self._reader = stream_reader
        self._unpack = struct.Struct('>I')  # 4-byte big-endian length
        self.mu = asyncio.Lock()

    async def read_frames(self) -> List[bytes]:
        await self.mu.acquire()
        # 读 total_len
        total_len_bytes = await self._reader.readexactly(4)
        total_size = self._unpack.unpack(total_len_bytes)[0]

        # 一次性读完剩余内容
        buf = await self._reader.readexactly(total_size)
        view = memoryview(buf)

        # 读取帧数量
        frame_count = self._unpack.unpack(view[:4])[0]
        offset = 4

        # 读取每帧长度
        lengths: List[int] = []
        for _ in range(frame_count):
            length = self._unpack.unpack(view[offset:offset+4])[0]
            lengths.append(length)
            offset += 4

        # 提取每帧数据（仍然是 memoryview 的切片）
        frames: List[bytes] = []
        for length in lengths:
            frame = view[offset:offset+length]
            frames.append(bytes(frame))  # 可改为 frame.tobytes() 或直接返回 memoryview(frame)
            offset += length
        self.mu.release()
        return frames


# class FramePipeReader(FrameConnReader):
#     def __init__(self, conn: MultiProcessConn):
#         self.conn = conn

#     async def recv(self) -> Optional[Frames]:
#         return self.conn.recv()


# class FramePipeWriter(FrameConnWriter):
#     def __init__(self, conn: MultiProcessConn) -> None:
#         self._pack = struct.Struct('>I')
#         self.conn = conn
#         self.closed = False
#         self.fd = self.conn.fileno()

#     def close(self):
#         if self.closed:
#             return
#         self.closed = True
#         self.conn.close()

#     def send(self, frames: Frames) -> None:
#         self.conn.send(frames)

class FramePipeReader(FrameConnReader):
    def __init__(self, conn: MultiProcessConn, loop: asyncio.AbstractEventLoop):
        self.fd = conn.fileno()
        self.loop = loop
        self._closed = False
        self.loop.add_reader(self.fd, self._on_data)
        self._unpack = struct.Struct('>I')  # 4-byte big-endian length
        self._buffer = b""
        self._queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

    def _on_data(self):
        data = os.read(self.fd, 128)
        if len(data) == 0:
            self._closed = True
            self.loop.remove_reader(self.fd)
            self._queue.put_nowait(None)
        self._queue.put_nowait(data)

    async def _readexactly(self, count: int) -> Optional[bytes]:
        while len(self._buffer) < count:
            b = await self._queue.get()
            if b is None:
                return None
            self._buffer += b
        ret = self._buffer[:count]
        self._buffer = self._buffer[count:]
        return ret

    async def recv(self) -> Optional[Frames]:
        # 读 total_len
        total_len_bytes = await self._readexactly(4)
        if total_len_bytes is None:
            return None
        total_size = self._unpack.unpack(total_len_bytes)[0]
        # 一次性读完剩余内容
        buf = await self._readexactly(total_size)
        if buf is None:
            return None
        view = memoryview(buf)

        # 读取帧数量
        frame_count = self._unpack.unpack(view[:4])[0]
        offset = 4

        # 读取每帧长度
        lengths: List[int] = []
        for _ in range(frame_count):
            length = self._unpack.unpack(view[offset:offset+4])[0]
            lengths.append(length)
            offset += 4

        # 提取每帧数据（仍然是 memoryview 的切片）
        frames: List[bytes] = []
        for length in lengths:
            frame = view[offset:offset+length]
            frames.append(bytes(frame))  # 可改为 frame.tobytes() 或直接返回 memoryview(frame)
            offset += length

        return cast(Frames, frames)


class FramePipeWriter(FrameConnWriter):
    def __init__(self, conn: MultiProcessConn) -> None:
        self._pack = struct.Struct('>I')
        self.conn = conn
        self.closed = False
        self.fd = self.conn.fileno()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.conn.close()

    def send(self, frames: Frames, on_release_callback: Optional[Callable[[], None]]) -> None:
        total_len = sum([len(f)+4 for f in frames])+4
        header = bytearray()
        header += self._pack.pack(total_len)
        header += self._pack.pack(len(frames))
        for data in frames:
            header += self._pack.pack(len(data))
        os.write(self.fd, header)
        for frame in frames:
            os.write(self.fd, bytes(frame))
        if on_release_callback:
            on_release_callback()
