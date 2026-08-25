import asyncio
import collections
import heapq
import os
import struct
import time
import traceback
from typing import Dict, List, Optional, cast
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


class StdEventLoop(asyncio.SelectorEventLoop, Loop):  # type:ignore
    def __init__(self, busy_wait: bool) -> None:
        super().__init__()
        self.scheduled: List[Handle] = self._scheduled  # type:ignore
        self.ready: collections.deque[Handle] = self._ready  # type:ignore
        self._clock_resolution = time.get_clock_info('monotonic').resolution
        self.blocking_profiler = MixtureProfiler()
        self.blocking_profiler.enable(enable=False)
        self.nonblocking_profiler = MixtureProfiler()
        self.nonblocking_profiler.enable(enable=False)
        self.busy_wait = busy_wait
        self.no_gc: Dict[str, Any] = {}
        self._blocking_scheduler: Optional[BlockingScheduler] = None
        self.poll_delegate: List[Callable[[], bool]] = []

        def exit_on_except(loop: Loop, e: Any):
            if hasattr(e, "__traceback__"):
                print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            print(e)
            os._exit(-1)

        self.on_except(exit_on_except)

    def set_blocking_scheduler(self, scheduler: BlockingScheduler) -> None:
        assert self._blocking_scheduler is None
        self._blocking_scheduler = scheduler

    @property
    def blocking_scheduler(self) -> Optional[BlockingScheduler]:
        return self._blocking_scheduler

    # async def run_blocking_soon(self, call: Callable[..., RetType], *call_args: Any, **schedule_kwargs: Any) -> RetType:
    #     fut: asyncio.Future[RetType] = asyncio.Future()

    #     def enclosure():
    #         ret = call(*call_args)
    #         fut.set_result(ret)
    #     if self.blocking_scheduler is None:
    #         self.call_soon(enclosure)
    #     else:
    #         self.blocking_scheduler.add_blocking_call(enclosure, **schedule_kwargs)
    #     return await fut

    # def call_blocking_soon(self, call: Callable[..., RetType], *call_args: Any, **schedule_kwargs: Any) -> None:
    #     if self.blocking_scheduler is None:
    #         self.call_soon(call, call_args)
    #         return

    #     def enclosure():
    #         call(*call_args)
    #     self.blocking_scheduler.add_blocking_call(enclosure, **schedule_kwargs)

    def on_except(self, callback: Callable[[Loop, Any], None]) -> None:
        def exception_handler(loop: asyncio.AbstractEventLoop, context: Dict[str, Any]):
            callback(self, context['exception'] if ('exception' in context) else Exception(context))
        self.set_exception_handler(exception_handler)

    def enable_profile(self, non_blocking: bool, blocking: bool) -> None:
        self.blocking_profiler.enable(blocking)
        self.nonblocking_profiler.enable(non_blocking)

    @property
    def blocking_calltime_ms(self) -> float:
        return self.blocking_profiler.blocking_time_ms

    @property
    def non_blocking_calltime_ms(self) -> float:
        return self.nonblocking_profiler.blocking_time_ms

    def clear_profile(self) -> None:
        self.blocking_profiler.clear_profile()
        self.nonblocking_profiler.clear_profile()

    def dump_profile(self, path: Optional[str] = None):
        if path is None:
            path = "dump_profile"
        os.makedirs(path, exist_ok=True)
        print(f"avg blocking call time: {self.blocking_profiler.blocking_time_ms:.4f}ms")
        self.blocking_profiler.dump_profile(os.path.join(path, "blocking"))
        print(f"avg non blocking call time: {self.nonblocking_profiler.blocking_time_ms:.4f}ms")
        self.nonblocking_profiler.dump_profile(os.path.join(path, "nonblocking"))

    def _remove_cancelled(self):
        # Remove delayed calls that were cancelled from head of queue.
        while self.scheduled and self.scheduled[0]._cancelled:
            self._timer_cancelled_count -= 1
            handle = heapq.heappop(self.scheduled)
            handle._scheduled = False  # type:ignore

    def _run_handles(self):
        end_time = self.time() + self._clock_resolution
        while self.scheduled:
            handle = self.scheduled[0]
            if handle._when >= end_time:  # type:ignore
                break
            handle = heapq.heappop(self.scheduled)
            handle._scheduled = False  # type:ignore
            self.ready.append(handle)

        ntodo = len(self.ready)
        if ntodo != 0:
            for _ in range(ntodo):
                handle = self.ready.popleft()
                if handle._cancelled:
                    continue
                else:
                    with self.nonblocking_profiler:
                        handle._run()
        elif self._blocking_scheduler:
            with self.blocking_profiler:
                self._blocking_scheduler.run_once()
        handle = None

    def _select_common(self):
        # get wait timeout for io
        timeout = None
        if self.ready or self._blocking_scheduler or cast(bool, self._stopping):  # type:ignore
            timeout = 0
        elif self.scheduled:
            # Compute the desired timeout.
            when = self.scheduled[0]._when  # type:ignore
            timeout = max(0, when - self.time())

        event_list = self._selector.select(timeout)  # type:ignore
        self._process_events(event_list)  # type:ignore

    def _busy_wait_poll(self):
        i = 0
        while i < len(self.poll_delegate):
            if self.poll_delegate[i]():
                self.poll_delegate.pop(i)
                return True
            else:
                i += 1
        return False

    def _select_busy_wait(self):
        # get wait timeout for io
        timeout = None
        when = None
        if self.ready or self._blocking_scheduler or cast(bool, self._stopping):  # type:ignore
            timeout = 0  # wait 0s
            when = self.time()
        elif self.scheduled:
            # Compute the desired timeout.
            when = self.scheduled[0]._when  # type:ignore
            timeout = max(0, when - self.time())
            # wait timeout
        else:
            # wait forever
            timeout = None
            when = None

        event_list = self._selector.select(0)  # type:ignore
        self._process_events(event_list)  # type:ignore

        if timeout is None or timeout != 0:
            while (when is None) or self.time() < when:
                event_list = self._selector.select(0)  # type:ignore
                if event_list:
                    self._process_events(event_list)  # type:ignore
                    break
                elif self._busy_wait_poll():
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

    def delegate_c_poll_closure(self, load: PollBlockingLoad) -> Any:
        fut = asyncio.Future()
        if not self.busy_wait:
            while not load.query_finished():
                pass
            fut.set_result(load.retrieve_result())
        else:
            def enclosure() -> bool:
                ok = load.query_finished()
                if ok:
                    fut.set_result(load.retrieve_result())
                    return True
                return False
            self.poll_delegate.append(enclosure)
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
