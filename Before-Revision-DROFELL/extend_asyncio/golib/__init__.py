from abc import ABC, abstractmethod
from multiprocessing.connection import Connection as MultiProcessConn
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast, overload
from . import goselector  # type: ignore

# TODO: remove this, for test only
goselector.config_async_zmc_send(True)
# goselector.config_debug_print_pin_buffer_borrow_give_back(True)
MAXIMUM_KEEP_ALIVES = 1000


class Future(ABC):
    @abstractmethod
    def set_result(self, value: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def __await__(self) -> Any:
        raise NotImplementedError


class Queue(ABC):
    @abstractmethod
    def get(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def put_nowait(self, item: Any) -> Any:
        raise NotImplementedError


EventData = TypeVar("EventData")

RECIVER_ID_INTERRUPT = 1
RECIVER_ID_FREE_PYOBJ = 2
NORMAL_RECIVER_BEGINS = 100


class _Frames(ABC):
    @abstractmethod
    def __str__(self) -> str: raise NotImplementedError
    @abstractmethod
    def __len__(self) -> int: raise NotImplementedError

    @overload
    def __getitem__(self, index: int) -> Union[memoryview, bytes]: ...

    @overload
    def __getitem__(self, index: slice) -> List[Union[memoryview, bytes]]: ...

    @abstractmethod
    def __getitem__(self, index: Union[int, slice]) -> Union[Union[memoryview, bytes], List[Union[memoryview, bytes]]]: raise NotImplementedError


Frames = Union[_Frames, List[bytes], List[memoryview]]


class ConnHandler:
    def __init__(self, handler: Any, keep_alive: Callable[[Callable[[], None]], int]) -> None:
        self.handler = handler
        self.keep_alive = keep_alive
        self.recv_queue: Queue = Queue()  # type:ignore

    def send(self, frames: Frames, on_release_callback: Optional[Callable[[], None]]) -> None:
        if not isinstance(frames, list):
            frames = [f for f in frames]

        def _release():
            nonlocal frames
            del frames
            if on_release_callback is not None:
                on_release_callback()
        keep_alive_id = self.keep_alive(_release)
        self.handler.send(frames, keep_alive_id)

    def _put(self, frames: Any) -> bool:
        if len(frames) == 0:
            self.close()
            self.recv_queue.put_nowait(None)
            return False
        self.recv_queue.put_nowait(frames)
        return True

    async def recv(self) -> Optional[Any]:
        return await self.recv_queue.get()

    def close(self) -> None:
        self.handler.close()


class ServerHandler:
    def __init__(self, error_fut: Future) -> None:
        self.error_fut = error_fut

    def wait_server_error(self):
        return self.error_fut


class Selector:
    def __init__(self, busy_wait: bool) -> None:
        self.base = goselector.Selector(busy_wait)
        self.reciver_id = NORMAL_RECIVER_BEGINS
        self.recivers: Dict[int, Tuple[Callable[[Any], Optional[bool]], Callable[[], Any]]] = {}
        self.keep_alives: Dict[int, Callable[[], None]] = {}
        self.keep_alive_id: int = 0

    def keep_alive(self, on_free: Callable[[], None]):
        self.keep_alive_id += 1
        i = self.keep_alive_id
        self.keep_alives[i] = on_free
        if len(self.keep_alives) > MAXIMUM_KEEP_ALIVES:
            raise RuntimeError("too much keep alives!")
        return i

    def _next_reciver(self, reciver: Callable[[EventData], Optional[bool]], taker: Callable[[], EventData]) -> int:
        self.reciver_id += 1
        self.recivers[self.reciver_id] = (reciver, taker)
        return self.reciver_id

    def interrupt(self):
        self.base.interrupt()

    def select_by_until(self, until: float):
        reciver_id = self.base.select(until)
        if reciver_id == 0:
            return None
        while reciver_id != 0:
            if reciver_id < NORMAL_RECIVER_BEGINS:
                if reciver_id == RECIVER_ID_INTERRUPT:
                    self.base.take_none()
                elif reciver_id == RECIVER_ID_FREE_PYOBJ:
                    obj_keep_alive_id = self.base.take_int()
                    # print("release py side ka obj:", obj_keep_alive_id)
                    self.keep_alives[obj_keep_alive_id]()
                    del self.keep_alives[obj_keep_alive_id]
                else:
                    raise ValueError(reciver_id)
                reciver_id = self.base.ready_reciver_id()
                continue
            r, g = self.recivers[reciver_id]
            # put result but not send it to coro
            # but if is a callback, it will run directly
            st = time.perf_counter_ns()
            keep = r(g())
            et = time.perf_counter_ns()
            if et-st > 5*1e4:
                print(f"WARNING: blocking! expect no longer than 0.05ms, actual: {(et-st)/1e6}ms @ {r}")
            if not keep:
                del self.recivers[reciver_id]
            reciver_id = self.base.ready_reciver_id()

    async def debug_event_after(self, seconds: float, ev_type: str) -> Any:
        taker = self.base.take_none
        if ev_type == "string":
            taker = self.base.take_str
        elif ev_type == "bytes":
            taker = self.base.take_bytes
        elif ev_type == "frames":
            taker = self.base.take_frames
        fut = Future()  # type:ignore
        i = self._next_reciver(fut.set_result, taker)
        self.base.debug_event_after(seconds, i, ev_type)
        r = await fut
        return r

    def debug_callback_after(self, seconds: float, ev_type: str, on_result_cb: Callable[[Any], bool]):
        taker = self.base.take_none
        if ev_type == "string":
            taker = self.base.take_str
        elif ev_type == "bytes":
            taker = self.base.take_bytes
        elif ev_type == "frames":
            taker = self.base.take_frames

        i = self._next_reciver(cast(Any, on_result_cb), taker)
        self.base.debug_callback_after(seconds, i, ev_type)

    def _take_conn(self) -> Union[ConnHandler, Exception]:
        recvID = self._next_reciver(None, None)  # type:ignore
        try:
            res = self.base.take_connect(recvID)
            _handler = res.make_handler()
            handler = ConnHandler(_handler, self.keep_alive)
            self.recivers[recvID] = (handler._put, self.base.take_zmc_frames)  # type:ignore
            return handler

        except Exception as e:
            del self.recivers[recvID]
            return e

    async def tcp_connect(self, addr: str) -> ConnHandler:
        conn_res = Future()  # type:ignore
        conn_res_id = self._next_reciver(conn_res.set_result, self._take_conn)
        self.base.tcp_connect(conn_res_id, addr)
        res = await conn_res
        if isinstance(res, Exception):
            raise res
        return res

    async def tcp_server(self, loop: Any, addr: str, on_conn: Any):
        server_err = Future()  # type:ignore

        def _on_new_conn(conn: Union[ConnHandler, Exception]) -> bool:
            if isinstance(conn, Exception):
                server_err.set_result(conn)
                return False

            async def _clean_up(c: ConnHandler):
                await on_conn(c)
                c.close()
            loop.create_task(_clean_up(conn))
            return True
        new_conn_id = self._next_reciver(_on_new_conn, self._take_conn)
        self.base.tcp_server(new_conn_id, addr)
        return ServerHandler(server_err)

    def pipe_reader(self, read_conn: MultiProcessConn) -> ConnHandler:
        recvID = self._next_reciver(None, None)  # type:ignore
        _h = self.base.pipe_reader(read_conn.fileno(), recvID)
        handler = ConnHandler(_h, None)  # type:ignore
        self.recivers[recvID] = (handler._put, self.base.take_zmc_frames)  # type:ignore
        handler.__dict__["underlay"] = read_conn
        return handler

    def pipe_writer(self, write_conn: MultiProcessConn) -> ConnHandler:
        _h = self.base.pipe_writer(write_conn.fileno())
        handler = ConnHandler(_h, self.keep_alive)
        handler.__dict__["underlay"] = write_conn
        return handler

    def delegate_c_blocking_closure(self, load: Any) -> Future:
        fut = Future()  # type:ignore

        def _taker():
            self.base.take_none()
            ret = load.retrieve_result()
            return ret
        reciver_id = self._next_reciver(fut.set_result, _taker)
        self.base.delegate_c_blocking_closure(load.get_blocking_delegate(), reciver_id)
        return fut

    def delegate_c_poll_closure(self, load: Any) -> Future:
        fut = Future()  # type:ignore

        def _taker():
            self.base.take_none()
            ret = load.retrieve_result()
            return ret
        reciver_id = self._next_reciver(fut.set_result, _taker)
        self.base.delegate_c_poll_closure(load.get_poll_delegate(), reciver_id)
        return fut

    # async def delegate_py_blocking_closure(self, load: Any, with_gil: bool) -> Any:
    #     fut = Future()
    #     reciver_id = self._next_reciver(fut.set_result, self.base.take_none)
    #     self.base.delegate_py_blocking_closure(load.get_blocking_delegate(), reciver_id, with_gil)
    #     await fut
    #     return load.retrieve_result()

    # def torch_event_sync(self) -> Awaitable[int]:
    #     fut = Future()
    #     reciver_id = self._next_reciver(fut.set_result, self.base.take_int64)
    #     self.base.torch_event_sync(reciver_id)
    #     return fut
