from abc import ABC, abstractmethod
import collections
from typing import TypeVar, Any, Awaitable, Callable, Coroutine, List, Optional, Union, overload
from multiprocessing.connection import Connection as MultiProcessConn
from concurrent.futures import Executor


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


class FrameConnReader(ABC):
    @abstractmethod
    async def recv(self) -> Optional[Frames]: raise NotImplementedError


class FrameConnWriter(ABC):
    @abstractmethod
    def send(self, frames: Frames, on_release_callback: Optional[Callable[[], None]]) -> None: raise NotImplementedError
    @abstractmethod
    def close(self) -> None: raise NotImplementedError


class FrameConn(FrameConnReader, FrameConnWriter):
    ...


class BlockingLoad(ABC):
    @abstractmethod
    def get_blocking_delegate(self) -> Any: raise NotImplementedError
    @abstractmethod
    def run_blocking_delegate(self) -> None: raise NotImplementedError
    @abstractmethod
    def retrieve_result(self) -> Any: raise NotImplementedError


class PollBlockingLoad(ABC):
    @abstractmethod
    def get_poll_delegate(self) -> Any: raise NotImplementedError
    @abstractmethod
    def query_finished(self) -> bool: raise NotImplementedError
    @abstractmethod
    def retrieve_result(self) -> Any: raise NotImplementedError

# class BlockingTaskContainer:
#     @abstractmethod
#     async def __aenter__(self) -> None: raise NotImplementedError
#     @abstractmethod
#     async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool: raise NotImplementedError


class BlockingScheduler(ABC):
    # @abstractmethod
    # def add_blocking_call(self, call: Callable[[], None]) -> None: raise NotImplementedError
    @abstractmethod
    def __bool__(self) -> bool: raise NotImplementedError
    @abstractmethod
    def __len__(self) -> int: raise NotImplementedError
    @abstractmethod
    def run_once(self) -> None: raise NotImplementedError


class BasicBlockingScheduler(BlockingScheduler):
    def __init__(self) -> None:
        self.blockings: collections.deque[Callable[[], None]] = collections.deque()

    # def add_blocking_call(self, call:  Callable[[], None]) -> None:
    #     self.blockings.append(call)

    def __bool__(self) -> bool:
        return len(self.blockings) != 0

    def __len__(self) -> int:
        return len(self.blockings)

    def run_once(self) -> None:
        self.blockings.popleft()()


RetType = TypeVar("RetType")


class Loop(ABC):
    # refer to link_capture_std_asyncio.py if looking for full trace blocking capture
    @abstractmethod
    def set_blocking_scheduler(self, scheduler: BlockingScheduler) -> None: raise NotImplementedError
    @property
    @abstractmethod
    def blocking_scheduler(self) -> Optional[BlockingScheduler]: raise NotImplementedError
    # @abstractmethod
    # async def run_blocking_soon(self, call: Callable[..., RetType], *call_args: Any, **schedule_kwargs: Any) -> RetType: raise NotImplementedError
    # @abstractmethod
    # def call_blocking_soon(self, call: Callable[..., RetType], *call_args: Any, **schedule_kwargs: Any) -> None: raise NotImplementedError

    # @property
    # @abstractmethod
    # def is_running_blocking(self) -> bool: raise NotImplementedError
    # @abstractmethod
    # def run_as_blocking(self) -> BlockingTaskContainer: raise NotImplementedError

    @abstractmethod
    def is_running(self) -> bool: raise NotImplementedError
    @abstractmethod
    def create_task(self, coro: Coroutine[Any, Any, Any]) -> Awaitable[Any]: raise NotImplementedError
    @abstractmethod
    def run_forever(self) -> None: raise NotImplementedError
    @abstractmethod
    def call_soon(self, callback: Callable[..., Any], *args: Any) -> None: raise NotImplementedError
    @abstractmethod
    def call_later(self, call_after_seconds: float, callback: Callable[..., Any], *args: Any) -> None: raise NotImplementedError
    @abstractmethod
    def call_soon_threadsafe(self, callback: Callable[..., Any], *args: Any) -> None: raise NotImplementedError
    @abstractmethod
    def on_except(self, callback: Callable[["Loop", Any], None]) -> None: raise NotImplementedError
    @abstractmethod
    async def run_in_executor(self, executor: Executor, task: Callable[..., RetType], *args: Any) -> RetType: raise NotImplementedError
    @abstractmethod
    def enable_profile(self, non_blocking: bool, blocking: bool) -> None: raise NotImplementedError
    @abstractmethod
    def dump_profile(self, path: Optional[str] = None) -> None: raise NotImplementedError
    @abstractmethod
    def clear_profile(self) -> None: raise NotImplementedError
    @property
    @abstractmethod
    def non_blocking_calltime_ms(self) -> float: raise NotImplementedError
    @property
    @abstractmethod
    def blocking_calltime_ms(self) -> float: raise NotImplementedError

    @abstractmethod
    async def tcp_connect(self, addr: str) -> FrameConn: raise NotImplementedError
    @abstractmethod
    async def tcp_server(self, addr: str, on_conn: Callable[[FrameConn], Coroutine[Any, Any, None]]) -> Any: raise NotImplementedError
    @abstractmethod
    def pipe_reader(self, read_conn: MultiProcessConn) -> FrameConnReader: raise NotImplementedError
    @abstractmethod
    def pipe_writer(self, write_conn: MultiProcessConn) -> FrameConnWriter: raise NotImplementedError
    @abstractmethod
    def delegate_c_blocking_closure(self, load: BlockingLoad) -> Awaitable[Any]: raise NotImplementedError
    @abstractmethod
    def delegate_c_poll_closure(self, load: PollBlockingLoad) -> Awaitable[Any]: raise NotImplementedError
    # @abstractmethod
    # async def delegate_py_blocking_closure(self, load: BlockingLoad, with_gil: bool) -> Any: raise NotImplementedError
    # @abstractmethod
    # def torch_event_sync(self) -> Awaitable[int]: raise NotImplementedError
