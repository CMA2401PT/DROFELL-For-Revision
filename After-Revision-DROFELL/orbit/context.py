import extend_asyncio as asyncio
import pickle
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type
from uuid import uuid4

from .args import ArgsKwargs
from .defines import Location, Logger, OrbitName, Packet, Context
from .logger import RemoteLogger


class Merger:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Packet] = asyncio.Queue()
        self._writer: Dict[str, Callable[[str, Any], None]] = {}

        def _default_writer(dst: str, api: str, data: Any, not_exits_ok: bool):
            assert not_exits_ok, f"connection to {dst} is not established"
        self._default_writer: Callable[[str, str, Any, bool], None] = _default_writer
        self._handlers: Dict[str, Callable[[Packet], None]] = {}
        self._listeners: List[Callable[[Packet], None]] = []
        self._no_gc: List[Any] = []

    def put_packet(self, packet: Packet):
        if packet[1] in self._handlers:
            self._handlers[packet[1]](packet)
            return
        for l in self._listeners:
            l(packet)
        self._queue.put_nowait(packet)

    def add_writer(self, target: str, writer: Callable[[str, Any], None]):
        self._writer[target] = writer

    def set_default_writer(self, writer: Callable[[str, str, Any, bool], None]) -> None:
        self._default_writer = writer

    async def recv(self) -> Packet:
        return await self._queue.get()

    def send(self, packet: Packet, not_exist_ok: bool = False) -> None:
        target, cmd, data = packet
        writer: Optional[Callable[[str, Any], None]] = self._writer.get(target)
        if writer is None:
            self._default_writer(target, cmd, data, not_exist_ok)
            return
        writer(cmd, data)

    def append_listener(self, listener: Callable[[Packet], None]) -> None:
        self._listeners.append(listener)

    def append_handler(self, packet_type: str, handler: Callable[[Packet], None]) -> None:
        self._handlers[packet_type] = handler

    def append_no_gc(self, no_gc: Any) -> None:
        self._no_gc.append(no_gc)


class BaseServiceContext(Merger):
    def __init__(self, location: Location, logger: Logger, network_hostname: str) -> None:
        Merger.__init__(self)
        self._loc = location
        self._logger = logger
        self._network_hostname = network_hostname
        self._args = ArgsKwargs()
        self._merger = Merger()
        self.no_gc: List[Any] = []
        name = self.name
        self.add_writer(name, lambda api, data: self.put_packet((name, api, data)))

        def _default_writer(dst: str, api: str, data: Any, not_exits_ok: bool):
            assert not_exits_ok, f"connection {self.name}->{dst} is not established"
        self.set_default_writer(_default_writer)

    def send(self, packet: Packet, not_exist_ok: bool = False) -> None:
        target, cmd, data = packet
        writer: Optional[Callable[[str, Any], None]] = self._writer.get(target)
        if writer is None:
            self._default_writer(target, cmd, data, not_exist_ok)
            return
        writer(cmd, data)

    def set_args(self, arg_pack: Tuple[Tuple[Any], Dict[str, Any]]):
        self._args.set(arg_pack)

    @property
    def name(self):
        return self._loc.name

    @property
    def location(self):
        return self._loc

    @property
    def network_hostname(self):
        return self._network_hostname

    @property
    def logger(self):
        return self._logger

    @property
    def args(self):
        return self._args


class LocalServiceContext(BaseServiceContext, Context):
    def __init__(self, location: Location, logger: Logger, network_hostname: str, vars: Dict[str, Any]) -> None:
        BaseServiceContext.__init__(self, location, logger, network_hostname)
        self._ready_fut: asyncio.Future[None] = asyncio.Future()
        self._vars: Dict[str, Any] = vars

    def report_ready(self) -> None:
        self._ready_fut.set_result(None)

    async def wait_ready_local(self) -> None:
        await self._ready_fut

    @property
    def ready_fut(self):
        return self._ready_fut

    def query_value(self, key: str) -> Awaitable[Any]:
        fut: asyncio.Future[Any] = asyncio.Future()
        fut.set_result(self._vars[key])
        return fut

    def set_value(self, key: str, value: Any) -> None:
        self._vars[key] = value

    async def try_connect(self, target: str) -> bool:
        return False


class RemoteServiceContext(BaseServiceContext, Context):
    def __init__(self, location: Location, logger: Logger, network_hostname: str, fwriter: asyncio.FrameConnWriter) -> None:
        BaseServiceContext.__init__(self, location, logger, network_hostname)
        self.fwriter = fwriter
        self._ready_fut: asyncio.Future[None] = asyncio.Future()
        self.result_waiter: Dict[bytes, Callable[[Any], None]] = {}
        self.location_b = self.location.name.encode()

    def report_ready(self) -> None:
        self._ready_fut.set_result(None)
        self.fwriter.send([b"ready", self.location_b], None)

    async def wait_ready_local(self) -> None:
        await self._ready_fut

    def query_value(self, key: str) -> Awaitable[Any]:
        uuid_bytes = uuid4().bytes
        fut: asyncio.Future[Any] = asyncio.Future()
        self.result_waiter[uuid_bytes] = fut.set_result
        self.fwriter.send([b"query", uuid_bytes, key.encode()], None)
        return fut

    def set_value(self, key: str, value: Any) -> None:
        self.fwriter.send([b"set", key.encode(), pickle.dumps(value)], None)

    async def try_connect(self, target: str) -> bool:
        return False


async def make_remote_service_context(location: Location, orbit_addr: str, network_hostname: str, logger_impl: Optional[Type[RemoteLogger]] = None):
    conn = await asyncio.get_event_loop().tcp_connect(orbit_addr)
    conn.send([b"ORBITV01", location.name.encode(), pickle.dumps(location)], None)
    if logger_impl is None:
        logger_impl = RemoteLogger
    logger = logger_impl(location.name, frames_sender=conn, start_time=0)
    ctx = RemoteServiceContext(location=location, logger=logger, network_hostname=network_hostname, fwriter=conn)

    def write(typ: str, data: Any):
        conn.send([typ.encode(), pickle.dumps(data)], None)
    ctx.add_writer(OrbitName, write)

    async def read_loop():
        while True:
            frames = await conn.recv()
            if frames is None:
                raise RuntimeError("disconnect with orbit")
            api = bytes(frames[0]).decode()
            if api != "resp":
                ctx.put_packet((OrbitName, api, pickle.loads(frames[1])))
            else:
                uuid_bytes, ret = pickle.loads(frames[1])
                ctx.result_waiter[uuid_bytes](ret)
    coro = read_loop()
    ctx.no_gc.append(coro)
    task = asyncio.create_task(coro)
    ctx.no_gc.append(task)
    log_level = await ctx.query_value("log_level")
    start_time = await ctx.query_value("start_time")
    logger.start_time = start_time
    logger.set_level(log_level)
    return ctx, (coro, task)
