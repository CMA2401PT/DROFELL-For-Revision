import collections
from dataclasses import dataclass
import os
import pickle
import threading
import time

from .defines import Context, LocHandler, SatelliteName
from .context import LocalServiceContext, make_remote_service_context
from .launch_pad import LaunchPad
from .defines import LocationByName, Logger, Location, Orbit, OrbitName, Packet
from typing import Any, Callable, Iterable, Literal, Optional, Tuple, Dict, cast, List
from .logger import LocalLogger, SimpleLogger, PrefixLogger
import extend_asyncio as asyncio


@dataclass
class LocationContext(LocHandler):
    _location: Location
    logger: Logger
    _ready_ev: asyncio.Event
    _writer: Optional[Callable[[str, bytes], None]] = None

    @property
    def location(self):
        return self._location

    async def wait_ready(self) -> None:
        if self._ready_ev.is_set():
            return
        await self._ready_ev.wait()

    def ready_fut(self):
        return self._ready_ev.wait()

    def set_ready(self):
        self._ready_ev.set()

    def set_writer(self, w: Callable[[str, Any], None]):
        if self._writer is not None:
            raise ValueError(f"cmd writer of {self._location} is already set")
        self._writer = w

    def write_cmd(self, cmd: str, data: Any):
        if self._writer is None:
            raise ValueError(f"cmd writer of {self._location} is not set")
        self._writer(cmd, data)


class SimpleOrbit(Orbit):
    def __init__(self,
                 launch_pad: LaunchPad,
                 functions_dict: Dict[str, Callable[..., None]],
                 logger: Optional[Logger] = None,
                 network_hostname: str = "localhost",
                 log_level: Literal["debug", "info", "warning", "error"] = "info"
                 ) -> None:
        self.start_time = time.time_ns()
        self.functions_dict = functions_dict
        self._root_logger: Logger = logger or SimpleLogger()
        self._network_hostname = network_hostname
        self._log_level = log_level
        self._locations: Dict[LocationByName, LocationContext] = {}
        self._vars: Dict[str, Any] = {"ping": "pong", "log_level": log_level}
        self._system_location = Location(OrbitName, 0, 0,  OrbitName)
        _, self._system_context = self.make_local_ctx(self._system_location)
        self._system_logger = self._system_context.logger
        self._port: int = 0
        self._server_handler = None
        self.launch_pad = launch_pad
        self._no_gc: List[Any] = []
        self._handlers: Dict[str, Callable[[Packet], None]] = {}
        self._listeners: List[Callable[[Packet], None]] = []
        self._msg_queue: asyncio.Queue[Packet] = asyncio.Queue()
        self._vars["start_time"] = self.start_time
        self._free_to_join: bool = True

        self._coros_counter: Dict[Tuple[str, int, int], int] = collections.defaultdict(lambda: 0)

    def __getitem__(self, loc: LocationByName) -> LocHandler:
        return self._locations[loc]

    def new_loc(self, host: str = "", process: int = 0, thread: int = 0, Name: str = "") -> Location:
        t = (host, process, thread)
        if Name == "":
            Name = f"{host}-{process}-{thread}-{self._coros_counter[t]}"
        return Location(host=host, process=process, thread=thread,  name=Name)

    async def handle_connection(self, conn: asyncio.FrameConn) -> None:
        frames = await conn.recv()
        if frames is None:
            return
        magic_or_ping = frames[0]
        if magic_or_ping == b"ping":
            conn.send([b"ORBITV01"], None)
            await conn.recv()
            return
        assert magic_or_ping == b"ORBITV01", magic_or_ping
        loc_name = bytes(frames[1]).decode()
        if not self._free_to_join:
            assert loc_name in self._locations, loc_name
        else:
            if loc_name not in self._locations:
                loc = pickle.loads(bytes(frames[2]))
                print(f"free join: node loc: {loc}")
                self.add_loc(loc)
        context = self._locations[loc_name]
        location = context.location
        self._system_logger.info(f"{context.location} connected")
        context.set_writer(lambda cmd, data: conn.send([cmd.encode(), pickle.dumps(data)], None))
        logger = context.logger

        try:
            while True:
                frames = await conn.recv()
                if not frames:
                    self._system_logger.error(f"{location} disconnected")
                    self.put_packet((OrbitName, "drop_node", location))
                    del self._locations[loc_name]
                    return
                bcmd = bytes(frames[0])
                if bcmd == b'log':
                    level, loc_name, msg = bytes(frames[1]).decode(), bytes(frames[2]).decode(), bytes(frames[3]).decode()
                    getattr(logger, level)(msg)
                elif bcmd == b'ready':
                    context.set_ready()
                    self._system_logger.info(f"{location} ready")
                elif bcmd == b"query":
                    uuid_bytes, key = bytes(frames[1]), bytes(frames[2]).decode()
                    context.write_cmd("resp", (uuid_bytes, self._vars[key]))
                elif bcmd == b"set":
                    bkey, bvalue = bytes(frames[1]).decode(), pickle.loads(bytes(frames[2]))
                    self._vars[bkey] = bvalue
                else:
                    packet = (loc_name, bcmd.decode(), pickle.loads(bytes(frames[1])))
                    self.put_packet(packet)
        except Exception as e:
            self._system_logger.error(f"{location} disconnected with exception:{e}")
            self.put_packet((OrbitName, "drop_node", location))
            del self._locations[loc_name]
            # os._exit(-1)

    def add_loc(self, loc: Location) -> LocationContext:
        sublogger = PrefixLogger(self._root_logger, str(loc))
        context = LocationContext(loc, sublogger, asyncio.Event())
        self._locations[loc.name] = context
        return context

    def make_local_ctx(self, loc: Location):
        locationContext = self.add_loc(loc)
        locationContext.logger = LocalLogger(self._root_logger, str(loc), cast(Literal["debug", "info", "warning", "error"], self._log_level), self.start_time)
        context = LocalServiceContext(loc, locationContext.logger, self._network_hostname, self._vars)

        def node_to_orbit_writer(typ: str, data: Any):
            self.put_packet((loc.name, typ, data))
        context.add_writer(OrbitName, node_to_orbit_writer)
        locationContext.set_writer(lambda cmd, data: context.put_packet((OrbitName, cmd, data)))
        return locationContext, context

    async def init(self,
                   port: int = 7241,
                   satellites: Optional[Iterable[str]] = None,
                   timeout: Optional[float] = None
                   ) -> None:
        if satellites is None:
            satellites = tuple()
        self._port = int(port)
        unconnected = set(satellites)
        init_events = {satellite_id: asyncio.Event() for satellite_id in satellites}
        # if self.init_fn != "":
        #     loc = Location(OrbitName, 0, 0, 1, "Orbit-Init")
        #     _, ctx = self.make_local_ctx(loc)
        #     self._system_logger.info(f"Waiting Orbit-Init Ready...")
        #     self.launch_pad.launch(0, 0, 1, self.functions_dict[self.init_fn], ctx)
        #     await ctx.wait_ready_local()
        #     self._system_logger.info(f"Orbit-Init Ready")

        for satellite_id in satellites:
            satellite_name = SatelliteName(satellite_id)
            satellite_loc = Location(satellite_id, 0, 0,  satellite_name)
            self.add_loc(satellite_loc)
            # satellite_init_name = ""
            # if self.init_fn != "":
            #     satellite_init_name = f"Satellite-{satellite_id}-Init"
            #     satellite_init_loc = Location(satellite_id, 0, 0, 1, satellite_init_name)
            #     self.add_loc(satellite_init_loc)

            async def sub_waitor(host_name: str, name: str):
                await self._locations[name].wait_ready()
                self._system_logger.info(f"{name} ready")
                unconnected.remove(host_name)
                self._system_logger.info(f"still waiting {unconnected}")
                init_events[host_name].set()
            asyncio.create_task(sub_waitor(satellite_id, satellite_name))

        self._system_logger.info(f"Starting Orbit on port {port}, waiting Satellites: {satellites}")
        server_handler = await asyncio.get_event_loop().tcp_server(f":{self._port}", self.handle_connection)
        self._server_task = server_handler
        self._system_logger.info("Orbit server running.")

        await asyncio.wait_for(asyncio.gather(*(e.wait() for e in init_events.values())), timeout=timeout or None)
        self._system_logger.info("All satellites are ready.")

    @staticmethod
    def _task_runner_local_proc(target_fn: Callable[[Context], None], loc: Location, addr: str, arg_pack: Tuple[Tuple[Any], Dict[str, Any]], network_hostname: str):

        async def coro():
            print(f"{loc} @ {threading.get_ident()} of {os.getpid()}")
            ctx, _ = await make_remote_service_context(loc, addr, network_hostname)
            ctx.set_args(arg_pack=arg_pack)
            asyncio.get_event_loop().call_soon(target_fn, ctx)
        asyncio.create_task(coro())

    async def launch(self, loc: Location, fn_name: str, *args: Any, **kwargs: Any):
        self._system_logger.info(f"launch: {fn_name} @ {loc}")
        assert not loc.name in self._locations
        t = (loc.host, loc.process, loc.thread)
        self._coros_counter[t] += 1
        arg_pack = (args, kwargs)
        if loc.host == OrbitName:
            if loc.process == 0 and loc.thread == 0:
                _, lctx = self.make_local_ctx(loc)
                lctx.set_args(arg_pack)
                self.launch_pad.launch(loc.process, loc.thread,
                                       self.functions_dict[fn_name], args=(lctx,))
                await lctx.ready_fut
            else:
                ctx = self.add_loc(loc)
                args = (self.functions_dict[fn_name], loc, f"localhost:{self._port}", arg_pack, self._network_hostname,)
                self.launch_pad.launch(loc.process, loc.thread,
                                       self._task_runner_local_proc, args)
                await ctx.ready_fut()
        else:
            self._locations[f"Satellite-{loc.host}"].write_cmd("launch", (loc, fn_name, arg_pack))
            ctx = self.add_loc(loc)
            await ctx.ready_fut()

    async def launch_local(self, fn_name: str, func: Callable[[Context], None], *args: Any, **kwargs: Any):
        loc: Location = Location(name=fn_name)
        self._system_logger.info(f"launch: {fn_name} @ {loc}")
        assert not loc.name in self._locations
        t = (loc.host, loc.process, loc.thread)
        self._coros_counter[t] += 1
        arg_pack = (args, kwargs)
        _, lctx = self.make_local_ctx(loc)
        lctx.set_args(arg_pack)
        self.launch_pad.launch(loc.process, loc.thread, func, args=(lctx,))
        await lctx.ready_fut

    def drop(self, host: str = "", proc: Optional[int] = None):
        if host == "":
            host = OrbitName
        if proc == None:
            proc = 0
        if host != OrbitName:
            self._locations[SatelliteName(host)].write_cmd("drop", proc)
        else:
            assert proc != 0, "cannot drop main process"
            self.launch_pad.drop(proc)

    def send(self, loc: LocationByName, cmd: str, data: Any):
        w = self._locations[loc]
        w.write_cmd(cmd, data)

    async def recv(self) -> Packet:
        return await self._msg_queue.get()

    def put_packet(self, packet: Packet):
        if packet[1] in self._handlers:
            self._handlers[packet[1]](packet)
            return
        for l in self._listeners:
            l(packet)
        self._msg_queue.put_nowait(packet)

    def append_handler(self, packet_type: str, handler: Callable[[Packet], None]) -> None:
        self._handlers[packet_type] = handler

    def append_listener(self, listener: Callable[[Packet], None]) -> None:
        self._listeners.append(listener)

    def append_no_gc(self, no_gc: Any) -> None:
        self._no_gc.append(no_gc)
