import asyncio
import os
import time
import extend_asyncio as asyncio
from .logger import RemoteAndLocalLogger, RemoteLogger, RemoteStubLocalLogger

from .context import make_remote_service_context
from .launch_pad import LaunchPad


from .defines import Context, Location, LocationByName, Logger, Satellite, SatelliteName
from typing import Callable, Dict, Optional, Tuple, Any, Type, cast


class SimpleSatellite(Satellite):
    def __init__(self,
                 launch_pad: LaunchPad,
                 functions_dict:  Dict[str, Callable[..., None]],
                 host_id: str,
                 network_hostname: str = "",
                 stub_remote_log: bool = False
                 ) -> None:
        self.functions_dict = functions_dict
        self.satellite_id = host_id
        self.network_hostname = network_hostname
        self.init_fn: Optional[Callable[[LocationByName], None]] = None
        self.locations: Dict[LocationByName, Location] = {}
        self.logger: Logger
        self.launch_pad = launch_pad
        self.location = Location(host_id, 0, 0, SatelliteName(host_id))
        self.orbit_addr: str = ""
        self._ready: asyncio.Future[None] = asyncio.Future()
        self.logger_impl: Optional[Type[RemoteLogger]] = RemoteAndLocalLogger if not stub_remote_log else RemoteStubLocalLogger

    async def ready(self):
        await self._ready

    async def ping(self, orbit_addr: str):
        try:
            conn = await asyncio.get_event_loop().tcp_connect(orbit_addr)
            conn.send([b"ping"], None)
            frames = await conn.recv()
            if frames is None:
                conn.send([b"nop"], None)
                return Exception(f"protocol error: unknown protocol")
            if frames[0] != b'ORBITV01':
                conn.send([b"nop"], None)
                return Exception(f"protocol error: {frames[0]}")
            return None
        except Exception as e:
            return e

    # @staticmethod
    # async def _task_builder_remote(loc: Location, targetFn: Function, addr: str, arg_pack: Tuple[Tuple[Any], Dict[str, Any]], network_hostname: str):
    #     ctx, no_gc = await make_remote_service_context(loc, addr, network_hostname)
    #     ctx.set_args(arg_pack=arg_pack)
    #     await targetFn(ctx)

    @staticmethod
    def _task_runner_local_proc(target_fn: Callable[[Context], None], loc: Location, addr: str, arg_pack: Tuple[Tuple[Any], Dict[str, Any]], network_hostname: str, logger_impl: Optional[Type[RemoteLogger]]):
        async def coro():
            ctx, _ = await make_remote_service_context(loc, addr, network_hostname, logger_impl)
            ctx.set_args(arg_pack=arg_pack)
            asyncio.get_event_loop().call_soon(target_fn, ctx)
        asyncio.create_task(coro())

    def launch(self, target_fn: Callable[[Context], None], proc: int = 0, thread: int = 0, node_name: str = "", *args: Any, **kwargs: Any):
        if node_name == "":
            node_name = f"uname-{self.satellite_id}-P{proc}T{thread}C{self.launch_pad.count}"
        loc = Location(self.satellite_id, proc, thread, node_name)
        assert self.orbit_addr != ""
        self.launch_pad.launch(
            proc, thread,
            self._task_runner_local_proc, (target_fn, loc, self.orbit_addr, (args, kwargs), self.network_hostname, self.logger_impl))

    async def run(self, orbit_addr: str, wait: float = 0.0) -> None:
        exception = await self.ping(orbit_addr)
        if exception is not None:
            print(f"fail ping {orbit_addr}: {exception}")
            if wait == 0.0:
                raise exception
            start_time = time.time()
            while time.time()-start_time < wait and wait >= 0:
                print(f"retry after 1s")
                await asyncio.sleep(1)
                exception = await self.ping(orbit_addr)
                if exception is not None:
                    continue
                else:
                    break
            if exception is not None:
                exception = await self.ping(orbit_addr)
            if exception is not None:
                print(f"fail ping {orbit_addr}: {exception}")
                raise exception
        print(f"ping {orbit_addr} successful")

        ctx, _ = await make_remote_service_context(
            location=self.location,
            orbit_addr=orbit_addr,
            network_hostname=self.network_hostname,
            logger_impl=self.logger_impl
        )
        ctx.report_ready()
        print(f"connected to {orbit_addr}")
        self.logger = ctx.logger
        self.orbit_addr = orbit_addr
        self._ready.set_result(None)
        while True:
            source, cmd, data = await ctx.recv()
            assert source == "Orbit"
            # if cmd == "init_fn" and data != b'':
            #     init_fn = data.decode()
            #     init_loc = Location(self.satellite_id, 0, 0, 1, f"Satellite-{self.satellite_id}-Init")
            #     init_ctx, no_gc = await make_remote_service_context(
            #         location=init_loc,
            #         orbit_addr=orbit_addr,
            #         network_hostname=self.network_hostname,
            #     )
            #     self.logger.info("running init function")
            #     self.launch_pad.launch(0, 0, 1, self.functions_dict[init_fn], init_ctx)
            if cmd == "launch":
                loc = cast(Location, data[0])
                fn_name = cast(str, data[1])
                arg_pack = cast(Tuple[Tuple[Any], Dict[str, Any]], data[2])
                self.logger.info(f"launch {fn_name} @ {loc}")
                func = self.functions_dict[fn_name]
                args = (func, loc, orbit_addr, arg_pack, self.network_hostname)
                self.launch_pad.launch(loc.process, loc.thread, self._task_runner_local_proc, args)
            elif cmd == "drop":
                target: int = cast(int, data)
                if target == 0:
                    print("main process of satellite will drop, so all sub process will alse be dropped")
                    for proc_i in self.launch_pad.procs.keys():
                        self.launch_pad.drop(proc_i)
                    os._exit(0)
                else:
                    self.launch_pad.drop(target)

    def start(self, orbit_addr: str, wait: float = 0.0):
        self._task = asyncio.create_task(self.run(orbit_addr, wait))
