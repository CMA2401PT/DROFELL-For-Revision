from abc import ABC
from abc import abstractmethod
from typing import Awaitable, Callable, Dict, Iterable, Literal, Optional, Tuple, Any
from dataclasses import dataclass

from .args import ArgsKwargs

from .launch_pad import LaunchPad

LocationByName = str
OrbitName = "Orbit"


def SatelliteName(satellite_id: str):
    return f"Satellite-{satellite_id}"


def SatelliteID(satellite_name: str):
    return satellite_name.removeprefix("Satellite-")


@dataclass
class Location:
    host: str = OrbitName
    process: int = 0
    thread: int = 0
    name: LocationByName = ""

    def __post_init__(self):
        if self.host == "":
            self.host = OrbitName
        self._name = ""
        self._hash = 0

    def _ensure(self):
        if self._hash == 0:
            self._name = f"{self.host}.P{self.process}T{self.thread}:{self.name}"
            self._hash = self._name.__hash__()

    def __hash__(self) -> int:
        self._ensure()
        return self._hash

    def __str__(self) -> str:
        self._ensure()
        return self._name

    def __repr__(self) -> str:
        self._ensure()
        return self._name


class Logger(ABC):
    @abstractmethod
    def debug(self, data: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def info(self, data: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def warn(self, data: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def error(self, data: str) -> None:
        raise NotImplementedError


Packet = Tuple[LocationByName, str, Any]


class Context(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def location(self) -> Location:
        raise NotImplementedError

    @property
    def network_hostname(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def logger(self) -> Logger:
        raise NotImplementedError

    @property
    @abstractmethod
    def args(self) -> ArgsKwargs:
        raise NotImplementedError

    # network interface
    @abstractmethod
    async def recv(self) -> Packet:
        raise NotImplementedError

    @abstractmethod
    def send(self, packet: Packet, not_exist_ok: bool = False) -> None:
        raise NotImplementedError

    @abstractmethod
    async def try_connect(self, target: str) -> bool:
        raise NotImplementedError

    # network recv/send extend interface
    @abstractmethod
    def put_packet(self, packet: Packet) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_writer(self, target: str, writer: Callable[[str, Any], None]) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_default_writer(self, writer: Callable[[str, str, Any, bool], None]) -> None:
        raise NotImplementedError

    # network packet handler extend interface

    @abstractmethod
    def append_handler(self, packet_type: str, handler: Callable[[Packet], None]) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_listener(self, listener: Callable[[Packet], None]) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_no_gc(self, no_gc: Any) -> None:
        raise NotImplementedError

    # orbit interact
    @abstractmethod
    def report_ready(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def query_value(self, key: str) -> Awaitable[Any]:
        raise NotImplementedError

    @abstractmethod
    def set_value(self, key: str, value: Any) -> None:
        raise NotImplementedError


class LocHandler(ABC):
    @property
    @abstractmethod
    def location(self) -> Location:
        raise NotImplementedError

    @abstractmethod
    def write_cmd(self, cmd: str, data: Any) -> None:
        raise NotImplementedError


class Orbit(ABC):
    def __init__(self,
                 launch_pad: LaunchPad,
                 functions_dict: Dict[str, Callable[..., None]],
                 logger: Optional[Logger],
                 network_hostname: str,
                 log_level: Literal["debug", "info", "warning", "error"] = "info",
                 ) -> None:
        raise NotImplementedError

    async def init(self,
                   port: int,
                   satellites: Optional[Iterable[str]],
                   timeout: float = 0
                   ) -> None:
        raise NotImplementedError

    def launch(self,
               loc: Location,
               fn_name: str,
               args: Any = None,
               timeout: Optional[float] = None,
               ) -> Awaitable[Any]:
        raise NotImplementedError

    @abstractmethod
    def send(self, loc: LocationByName, cmd: str, data: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    async def recv(self) -> Packet:
        raise NotImplementedError

    @abstractmethod
    def append_handler(self, packet_type: str, handler: Callable[[Packet], None]) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_listener(self, listener: Callable[[Packet], None]) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_no_gc(self, no_gc: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, loc: LocationByName) -> LocHandler:
        raise NotImplementedError


class Satellite(ABC):
    def __init__(self,
                 launch_pad: LaunchPad,
                 functions_dict:  Dict[str, Callable[..., None]],
                 host_id: str,
                 network_hostname: str,
                 ) -> None:
        raise NotImplementedError

    async def run(self, orbit_addr: str, wait: float = 0.0) -> None:
        raise NotImplementedError
