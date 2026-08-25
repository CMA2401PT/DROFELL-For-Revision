import multiprocessing
from multiprocessing.synchronize import Event as MPEvent
from multiprocessing.connection import Connection as MPConn
import os
import time
import traceback
import extend_asyncio as asyncio
import pickle
import threading
from typing import Any, List, Optional, Tuple, Dict, Callable


# class FramePipeReader:
#     def __init__(self, fd: int, loop: Optional[asyncio.AbstractEventLoop] = None):
#         self.fd = fd
#         self.loop = loop or asyncio.get_event_loop()
#         self.buffer = bytearray()
#         self.frame_queue: asyncio.Queue[bytes] = asyncio.Queue()
#         self._closed = False
#         self.loop.add_reader(self.fd, self._on_data)

#     def _on_data(self):
#         data = os.read(self.fd, 8192)
#         self.buffer.extend(data)
#         self._extract_frames()

#     def _extract_frames(self):
#         view = memoryview(self.buffer)
#         offset = 0
#         total_len = len(view)
#         while total_len - offset >= 4:
#             frame_len = struct.unpack_from(">I", view, offset)[0]
#             if total_len - offset < 4 + frame_len:
#                 break
#             frame = bytes(view[offset + 4: offset + 4 + frame_len])
#             self.frame_queue.put_nowait(frame)
#             offset += 4 + frame_len
#         if offset:
#             self.buffer = bytearray(view[offset:].tobytes())

#     async def read_frame(self) -> bytes:
#         return await self.frame_queue.get()


class CorosPad:
    def __init__(self, loop: asyncio.Loop) -> None:
        self.loop = loop
        self.tid = threading.get_ident()
        self.no_gc: List[Any] = []

    def launch(self, func: Callable[..., None], args: Tuple[Any]):
        # assert self.loop.is_running()
        if threading.get_ident() != self.tid:
            self.loop.call_soon_threadsafe(func, *args)
        else:
            self.loop.call_soon(func, *args)


def test_coros_pad():
    asyncio.config_impl("golang", False)
    loop = asyncio.get_event_loop()
    coros_pad = CorosPad(loop)

    def test_load(tag: str):

        async def coro():
            for i in range(10):
                await asyncio.sleep(1)
                print(f"{tag}: {threading.get_ident()} {i}")
        asyncio.get_event_loop().create_task(coro())

    async def amain():
        coros_pad.launch(test_load, ("same thread",))

    def thread_launcher():
        time.sleep(1)
        coros_pad.launch(test_load, ("different thread",))

    threading.Thread(target=thread_launcher).start()
    loop.create_task(amain())
    loop.run_forever()


# test_coros_pad()


class ThreadPad:
    def __init__(self, root_loop: asyncio.Loop) -> None:
        # print(f"get loop @ {threading.get_ident()} of {os.getpid()}")
        self.thread_pads: Dict[int, CorosPad] = {
            0: CorosPad(root_loop)
        }
        self.threads: List[Any] = []
        self.loops: List[Any] = []
        self.mu = threading.Lock()

    def new_thread(self, thread_i: int, ev: threading.Event):
        import concurrent.futures.thread  # type:ignore get rid of python's very wired issue cause by threading and module import
        # print(f"new loop @ {threading.get_ident()} of {os.getpid()}")
        loop = asyncio.new_event_loop()
        self.loops.append(loop)
        pad = CorosPad(loop)
        self.thread_pads[thread_i] = pad
        loop.call_soon(ev.set)
        # print("thread init complete")
        loop.run_forever()

    def _ensure_thread(self, target_thread: int):
        ev = threading.Event()

        thread = threading.Thread(target=self.new_thread, args=(target_thread, ev), daemon=True)
        self.threads.append(thread)
        thread.start()
        ev.wait()

    def launch(self, target_thread: int, func: Callable[..., None], args: Tuple[Any]):
        self.mu.acquire_lock(True)
        if target_thread not in self.thread_pads.keys():
            self._ensure_thread(target_thread)
            # print("new thread launched")
        target_pod = self.thread_pads[target_thread]
        target_pod.launch(func, args)
        self.mu.release_lock()


def test_thread_pad():
    asyncio.config_impl("golang", True)
    loop = asyncio.get_event_loop()
    thread_pad = ThreadPad(loop)

    async def excep():
        raise RuntimeError("test exception")

    def test_load(tag: str):
        async def coro():
            for i in range(10):
                await asyncio.sleep(1)
                print(f"{tag}: {threading.get_ident()} {i}")
        asyncio.get_event_loop().create_task(coro())

    def test_load_except(tag: str):
        async def coro():
            for i in range(10):
                await asyncio.sleep(1)
                print(f"{tag}: {threading.get_ident()} {i}")
                try:
                    await excep()
                except Exception as e:
                    pass
        asyncio.get_event_loop().create_task(coro())

    async def amain():
        print("amian")
        thread_pad.launch(target_thread=0, func=test_load, args=("thread-0,coro-1",))
        thread_pad.launch(target_thread=0, func=test_load, args=("thread-0,coro-2",))
        thread_pad.launch(target_thread=1, func=test_load_except, args=("thread-1,coro-1",))
        thread_pad.launch(target_thread=1, func=test_load, args=("thread-1,coro-2",))
    loop.create_task(amain())
    loop.run_forever()


# test_thread_pad()

class LaunchPad:
    def __init__(self, process_init_fn: Optional[Callable[[], asyncio.Loop]] = None) -> None:
        if process_init_fn is None:
            process_init_fn = self.default_init_fn
        multiprocessing.set_start_method('spawn', force=True)
        self.process_pipes: Dict[int, Callable[[bytes], None]] = {}
        self.procs: Dict[int, multiprocessing.Process] = {}
        self.loop = process_init_fn()
        self.thread_pad = ThreadPad(self.loop)
        self.process_init_fn = process_init_fn
        self._counter = 0

    @staticmethod
    def default_init_fn():
        def on_except(_: asyncio.Loop, e: Exception):
            raise e
            os._exit(-1)
        asyncio.config_impl("golang", True)
        loop = asyncio.get_event_loop()
        loop.on_except(on_except)
        return loop

    @property
    def count(self):
        return self._counter

    @staticmethod
    def exception_handler(loop: asyncio.Loop, e: Exception):
        if hasattr(e, "__traceback__"):
            print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        print(e)
        os._exit(-1)

    @staticmethod
    def _process_box(process_init_fn:  Callable[[], asyncio.Loop], ready: MPEvent, reader_conn: MPConn):
        loop = process_init_fn()
        threads_pad = ThreadPad(loop)

        def on_dead():
            print("parent process dead, proc will terminate")
            os._exit(-1)

        async def process_frames():
            reader = loop.pipe_reader(reader_conn)
            ready.set()
            while True:
                frames = await reader.recv()
                if frames == None:
                    on_dead()
                else:
                    cmd, data = pickle.loads(frames[0])
                    if cmd == "launch":
                        thread_i, func, arg_pack = data
                        threads_pad.launch(thread_i, func, arg_pack)
                    elif cmd == "drop":
                        os._exit(0)

        # async def dead_checker():
        #     while True:
        #         await asyncio.sleep(1)
        #         ppid = os.getppid()
        #         if ppid == 1:
        #             on_dead()
        loop.create_task(process_frames())
        loop.run_forever()

    def _ensure_proc(self, proc_i: int):
        assert (proc_i in self.process_pipes) == (proc_i in self.procs)
        if proc_i in self.process_pipes:
            return
        ev = multiprocessing.Event()
        reader_conn, writer_conn = multiprocessing.Pipe(duplex=False)
        proc = multiprocessing.Process(target=self._process_box, args=(self.process_init_fn, ev, reader_conn), daemon=True)
        self.procs[proc_i] = proc
        proc.start()
        # os.close(read_fd)

        writer = self.loop.pipe_writer(writer_conn)

        def write_data(data: bytes):
            writer.send([data], None)
        self.process_pipes[proc_i] = write_data
        ev.wait()
        # reader_conn.close()

    def launch(self, proc: int, thread: int,  func: Callable[..., None], args: Tuple[Any, ...]):
        self._counter += 1
        if proc == 0:
            self.thread_pad.launch(thread, func, args)
        else:
            self._ensure_proc(proc_i=proc)
            pd = pickle.dumps(("launch", (thread, func, args)))
            self.process_pipes[proc](pd)

    def drop(self, proc: int):
        assert proc in self.process_pipes, f"{proc}, {self.process_pipes.keys()}"
        assert proc in self.procs, f"{proc}, {self.procs.keys()}"
        self.process_pipes[proc](pickle.dumps(("drop", None)))
        del self.process_pipes[proc]
        del self.procs[proc]


def timer(name: str):
    async def coro():
        i = 0
        while True:
            i += 1
            pid = os.getpid()
            tid = threading.get_ident()
            print(f"timer={name} @ Process={pid},Thread={tid}: {i}")
            await asyncio.sleep(1)
            if name == "1-1-0" and i == 3:
                print("exit!")
                os._exit(-1)
    asyncio.get_event_loop().create_task(coro())


def init_fn() -> asyncio.Loop:
    asyncio.config_impl("py", False)
    loop = asyncio.get_event_loop()
    loop.on_except(LaunchPad.exception_handler)
    return loop


def test_launch_pad():

    launcher = LaunchPad(init_fn)
    launcher.launch(0, 0, timer, ("0-0-0",))
    launcher.launch(0, 0, timer, ("0-0-1",))
    launcher.launch(0, 1, timer, ("0-1-0",))
    launcher.launch(0, 1, timer, ("0-1-1",))
    launcher.launch(1, 0, timer, ("1-0-0",))
    launcher.launch(1, 0, timer, ("1-0-1",))
    launcher.launch(1, 1, timer, ("1-1-0",))
    launcher.launch(1, 1, timer, ("1-1-1",))
    launcher.loop.run_forever()


# test_launch_pad()

if __name__ == "__main__":

    test_launch_pad()
