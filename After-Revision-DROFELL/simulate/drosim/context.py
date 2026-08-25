"""SimContext: the `orbit.defines.Context` contract, backed by SimFabric.

DROFELL's whole networking dependency is the 12-method `Context` ABC -- the
orchestration code never constructs an IP, port or route, only
`(logical node name, api name, payload)` triples. That is what makes this
simulator possible without touching `node/` at all (docs/02 §3.1).

Structure follows `orbit/context.py::Merger`, which is already a pure-Python
queue implementation; the only substantive change is that `send` goes through
SimFabric so it costs latency and bandwidth on the virtual timeline.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional

import extend_asyncio as asyncio
from orbit.args import ArgsKwargs
from orbit.defines import Context, Location, Logger, Packet

from .fabric import SimFabric


class SimLogger(Logger):
    """Logger that stamps virtual time and can be silenced."""

    def __init__(self, name: str, clock: Any, level: str = "warning") -> None:
        self.name = name
        self.clock = clock
        order = {"debug": 0, "info": 1, "warning": 2, "error": 3}
        self._threshold = order.get(level, 2)
        self.records: List[tuple[int, str, str]] = []
        self.keep_records = False

    def _emit(self, level: str, rank: int, data: str) -> None:
        if self.keep_records:
            self.records.append((self.clock.now_ns, level, data))
        if rank >= self._threshold:
            print(f"[{self.clock.now_s:9.4f}s {level[0].upper()} {self.name}] {data}")

    def debug(self, data: str) -> None:
        self._emit("debug", 0, data)

    def info(self, data: str) -> None:
        self._emit("info", 1, data)

    def warn(self, data: str) -> None:
        self._emit("warning", 2, data)

    def error(self, data: str) -> None:
        self._emit("error", 3, data)


class SimContext(Context):
    """One logical node's view of the simulated world."""

    def __init__(
        self,
        name: str,
        fabric: SimFabric,
        clock: Any,
        host: str = "host0",
        cpu_domain: Optional[str] = None,
        log_level: str = "warning",
        shared_values: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._loc = Location(host=host, name=name)
        self._name = name
        self._fabric = fabric
        self._clock = clock
        self._host = host
        self._cpu_domain = cpu_domain or f"{host}:{name}"
        self._logger = SimLogger(name, clock, log_level)
        self._args = ArgsKwargs()
        self._queue: asyncio.Queue[Packet] = asyncio.Queue()
        self._handlers: Dict[str, Callable[[Packet], None]] = {}
        self._listeners: List[Callable[[Packet], None]] = []
        self._writers: Dict[str, Callable[[str, Any], None]] = {}
        self._no_gc: List[Any] = []
        self._ready_fut: asyncio.Future[None] = asyncio.Future()
        self._values: Dict[str, Any] = (
            shared_values if shared_values is not None else {}
        )
        self.dropped = False

        fabric.register(
            name,
            self.put_packet,
            host=host,
            cpu_domain=self._cpu_domain,
        )
        # a node can always talk to itself without touching the fabric
        self._writers[name] = lambda api, data: self.put_packet((name, api, data))

    # -- identity ---------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def location(self) -> Location:
        return self._loc

    @property
    def network_hostname(self) -> str:
        return self._host

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def args(self) -> ArgsKwargs:
        return self._args

    def set_args(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        self._args.set((args, kwargs))

    # -- transport --------------------------------------------------------
    async def recv(self) -> Packet:
        return await self._queue.get()

    def send(self, packet: Packet, not_exist_ok: bool = False) -> None:
        target, api, data = packet
        if self.dropped:
            return
        writer = self._writers.get(target)
        if writer is not None:
            writer(api, data)
            return
        delivered = self._fabric.send(self._name, target, api, data)
        if not delivered and not not_exist_ok:
            # Mirror production behaviour: a send to an unknown peer is a bug
            # unless the caller explicitly tolerates it.
            if not self._fabric.known(target):
                raise AssertionError(
                    f"connection {self._name}->{target} is not established"
                )

    async def try_connect(self, target: str) -> bool:
        """Connection setup. Reachability is a fabric property, not a socket."""
        return self._fabric.known(target) and not self.dropped

    def put_packet(self, packet: Packet) -> None:
        if self.dropped:
            return
        handler = self._handlers.get(packet[1])
        if handler is not None:
            handler(packet)
            return
        for listener in self._listeners:
            listener(packet)
        self._queue.put_nowait(packet)

    def add_writer(self, target: str, writer: Callable[[str, Any], None]) -> None:
        self._writers[target] = writer

    def set_default_writer(
        self, writer: Callable[[str, str, Any, bool], None]
    ) -> None:
        self._default_writer = writer

    def append_handler(
        self, packet_type: str, handler: Callable[[Packet], None]
    ) -> None:
        self._handlers[packet_type] = handler

    def append_listener(self, listener: Callable[[Packet], None]) -> None:
        self._listeners.append(listener)

    def append_no_gc(self, no_gc: Any) -> None:
        self._no_gc.append(no_gc)

    # -- orbit interaction ------------------------------------------------
    def report_ready(self) -> None:
        if not self._ready_fut.done():
            self._ready_fut.set_result(None)

    async def wait_ready_local(self) -> None:
        await self._ready_fut

    @property
    def ready_fut(self) -> "asyncio.Future[None]":
        return self._ready_fut

    def query_value(self, key: str) -> Awaitable[Any]:
        fut: asyncio.Future[Any] = asyncio.Future()
        fut.set_result(self._values.get(key))
        return fut

    def set_value(self, key: str, value: Any) -> None:
        self._values[key] = value

    # -- failure injection ------------------------------------------------
    def drop(self) -> None:
        """Simulate this logical node vanishing: no sends, no deliveries."""
        self.dropped = True
        self._fabric.unregister(self._name)

    def __repr__(self) -> str:
        return f"<SimContext {self._name}@{self._host}>"
