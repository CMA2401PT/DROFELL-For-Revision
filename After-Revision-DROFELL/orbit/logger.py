import extend_asyncio as asyncio
import atexit
import time
from typing import Callable, Literal, cast
from .defines import LocationByName, Logger


class PrefixLogger(Logger):
    def __init__(self, base_logger: Logger, prefix: str):
        self._base = base_logger
        self._prefix = prefix+" "

    def debug(self, data: str) -> None: self._base.debug(self._prefix+data)
    def info(self, data: str) -> None: self._base.info(self._prefix+data)
    def warn(self, data: str) -> None: self._base.warn(self._prefix+data)
    def error(self, data: str) -> None: self._base.error(self._prefix+data)

    def __getattr__(self, key: str):
        ba = cast(Callable[[str], None], getattr(self._base, key))

        def fn(data: str) -> None:
            ba(self._prefix+data)
        self.__dict__[key] = fn
        return fn


class SimpleLogger(Logger):
    def __init__(self) -> None:
        pass

    def debug(self, data: str) -> None:
        print(f"\033[90m[DEBUG] {data}\033[0m")

    def info(self, data: str) -> None:
        print(f"[INFO] {data}")

    def warn(self, data: str) -> None:
        print(f"\033[33m[WARN] {data}\033[0m")

    def error(self, data: str) -> None:
        print(f"\033[31m[ERRO] {data}\033[0m")

    def __getattr__(self, key: str):
        def fn(data: str) -> None:
            print(f"\033[32m[{key}] {data}\033[0m")
        self.__dict__[key] = fn
        return fn


class FileLogger(Logger):
    def __init__(self, logfile: str, append: bool = True, terminal: bool = True) -> None:
        self.f = open(logfile, "a" if append else "w")
        self.t = terminal
        atexit.register(self.f.close)

    def debug(self, data: str) -> None:
        print(f"[DEBUG] {data}", file=self.f)
        if self.t:
            print(f"\033[90m[DEBUG] {data}\033[0m")

    def info(self, data: str) -> None:
        print(f"[INFO] {data}", file=self.f)
        if self.t:
            print(f"[INFO] {data}")

    def warn(self, data: str) -> None:
        print(f"[WARN] {data}", file=self.f)
        if self.t:
            print(f"\033[33m[WARN] {data}\033[0m")

    def error(self, data: str) -> None:
        print(f"[ERROR] {data}", file=self.f)
        if self.t:
            print(f"\033[31m[ERROR] {data}\033[0m")

    def __getattr__(self, key: str):
        def fn(data: str) -> None:
            print(f"\033[32m[{key}] {data}\033[0m", file=self.f)
            if self.t:
                print(f"\033[32m[{key}] {data}\033[0m")
        self.__dict__[key] = fn
        return fn


class LogLevel:
    def __init__(self, log_level: Literal["debug", "info", "warning", "error"] = "info") -> None:
        self.l = 1
        if log_level == "debug":
            self.l = 0
        elif log_level == "warning":
            self.l = 2
        elif log_level == "error":
            self.l = 3


def get_level(log_level: Literal["debug", "info", "warning", "error"] = "info"):
    return ["debug", "info", "warning", "error"].index(log_level)


class LocalLogger(Logger):
    def __init__(self, base_logger: Logger, prefix: str, log_level: Literal["debug", "info", "warning", "error"], start_time: int):
        self._base = base_logger
        self._prefix = prefix+" "
        self._l = get_level(log_level)
        self._t0 = start_time

    def debug(self, data: str) -> None:
        if self._l > 0:
            return
        self._base.debug(self._prefix+f"{(time.time_ns()-self._t0)/(10**9):.4f} "+data)

    def info(self, data: str) -> None:
        if self._l > 1:
            return
        self._base.info(self._prefix+f"{(time.time_ns()-self._t0)/(10**9):.4f} "+data)

    def warn(self, data: str) -> None:
        if self._l > 2:
            return
        self._base.warn(self._prefix+f"{(time.time_ns()-self._t0)/(10**9):.4f} "+data)

    def error(self, data: str) -> None:
        if self._l > 3:
            return
        self._base.error(self._prefix+f"{(time.time_ns()-self._t0)/(10**9):.4f} "+data)

    def __getattr__(self, key: str):
        ba = cast(Callable[[str], None], getattr(self._base, key))

        def fn(data: str) -> None:
            ba(self._prefix+f"{(time.time_ns()-self._t0)/(10**9):.4f} "+data)
        self.__dict__[key] = fn
        return fn


class RemoteLogger(Logger):
    def __init__(self, location: LocationByName, frames_sender: asyncio.FrameConnWriter, start_time: int):
        self._location = location
        self._location_b = self._location.encode()
        self._frames_sender = frames_sender
        self._l = 0
        self.start_time = start_time

    def set_level(self, log_level: Literal["debug", "info", "warning", "error"]):
        self._l = get_level(log_level)

    def _send(self, level: bytes, data: str) -> None:
        self._frames_sender.send([b"log", level, self._location_b, (f"{(time.time_ns()-self.start_time)/(10**9):.4f} {data}").encode()], None)

    def debug(self, data: str) -> None:
        if self._l > 0:
            return
        self._send(b'debug', data)

    def info(self, data: str) -> None:
        if self._l > 1:
            return
        self._send(b'info', data)

    def warn(self, data: str) -> None:
        if self._l > 2:
            return
        self._send(b'warn', data)

    def error(self, data: str) -> None:
        if self._l > 3:
            return
        self._send(b'error', data)

    def __getattr__(self, key: str):
        def fn(data: str) -> None:
            self._send(key.encode(), data)
        self.__dict__[key] = fn
        return fn


class RemoteAndLocalLogger(RemoteLogger):
    def __init__(self, location: LocationByName, frames_sender: asyncio.FrameConnWriter, start_time: int, local_log_level: Literal["debug", "info", "warning", "error"] = "info"):
        super().__init__(location=location, frames_sender=frames_sender, start_time=start_time)
        self._base = LocalLogger(SimpleLogger(), "", local_log_level, start_time=0)

    def set_level(self, log_level: Literal["debug", "info", "warning", "error"]):
        self._l = get_level(log_level)

    def debug(self, data: str) -> None:
        super().debug(data)
        self._base.debug(data)

    def info(self, data: str) -> None:
        super().info(data)
        self._base.info(data)

    def warn(self, data: str) -> None:
        super().warn(data)
        self._base.warn(data)

    def error(self, data: str) -> None:
        super().error(data)
        self._base.error(data)

    def __getattr__(self, key: str):
        rw = super().__getattr__(key)
        lw = self._base.__getattr__(key)

        def mw(data: str):
            rw(data)
            lw(data)
        return mw


class RemoteStubLocalLogger(RemoteLogger):
    def __init__(self, location: LocationByName, frames_sender: asyncio.FrameConnWriter, start_time: int, local_log_level: Literal["debug", "info", "warning", "error"] = "info"):
        self._base = LocalLogger(SimpleLogger(), "", local_log_level, start_time=0)

    def set_level(self, log_level: Literal["debug", "info", "warning", "error"]):
        self._l = get_level(log_level)

    def debug(self, data: str) -> None:
        self._base.debug(data)

    def info(self, data: str) -> None:
        self._base.info(data)

    def warn(self, data: str) -> None:
        self._base.warn(data)

    def error(self, data: str) -> None:
        self._base.error(data)

    def __getattr__(self, key: str):
        return self._base.__getattr__(key)
