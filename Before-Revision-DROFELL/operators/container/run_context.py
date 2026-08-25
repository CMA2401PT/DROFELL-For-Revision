import threading
from typing import Any, Callable, Dict, Literal, TypeVar
import torch

RetType = TypeVar("RetType")
Modes = Literal["inference", "no_grad", "normal", "grad"]


class RunContext:
    FAST = True  # FAST decide whether do full context switch in each operate

    def __init__(self, runner: "_ThreadContext", stream: torch.cuda.Stream) -> None:
        self._stream = stream
        self._device = self.stream.device
        self.device_name = f"{self.device}"
        self.cuda_stream = stream.cuda_stream
        self.stream_name = f"{self.cuda_stream}"
        self.runner = runner
        self._tid = threading.get_ident()

    def _check_tid(self):
        assert self._tid == threading.get_ident()

    def __call__(self, mode: Modes, step: Callable[..., RetType], **kwargs: Any) -> RetType:
        self._check_tid()
        if self.FAST:
            self.runner.switch_device(self.device_name, self._device)
            self.runner.switch_stream(self.stream_name, self._stream)
            self.runner.switch_mode(mode)
            return step(**kwargs)
        else:
            with torch.cuda.stream(self.stream):
                with torch.device(self.stream.device):
                    if mode == "inference":
                        with torch.inference_mode():
                            return step(**kwargs)
                    elif mode == "no_grad":
                        with torch.no_grad():
                            return step(**kwargs)
                    elif mode == "grad":
                        with torch.enable_grad():
                            return step(**kwargs)
                    else:
                        return step(**kwargs)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def stream(self) -> torch.cuda.Stream:
        return self._stream


class _ThreadContext:
    def __init__(self) -> None:
        self._running_mode: Modes = "normal"
        self._running_mode_exit = lambda: None
        self._running_device = "cpu"
        self._running_device_exit = lambda: None
        self._running_stream = ""
        self._cpu_device = torch.device("cpu")
        self._tid = threading.get_ident()

    def _check_tid(self):
        assert self._tid == threading.get_ident()

    def switch_device(self, new_device_name: str, device: torch.device):
        if new_device_name == self._running_device:
            return
        # print("SWITCH DEVICE!")
        self._running_device_exit()
        self._running_device = new_device_name
        device_ctx = torch.device(device)
        device_ctx.__enter__()
        self._running_device_exit = lambda: device_ctx.__exit__(None, None, None)

    def switch_cpu(self):
        self.switch_device("cpu", self._cpu_device)

    def switch_stream(self, new_stream: str, stream: torch.cuda.Stream):
        if new_stream == self._running_stream:
            return
        # print("SWITCH STREAM!")
        torch.cuda.set_stream(stream)
        self._running_stream = new_stream

    def switch_mode(self, new_mode: Literal["inference", "no_grad", "normal", "grad"]):
        if new_mode == self._running_mode:
            return
        # print("SWITCH MODE!")
        self._running_mode_exit()
        if new_mode == "inference":
            inference_context = torch.inference_mode()
            inference_context.__enter__()
            self._running_mode_exit = lambda: inference_context.__exit__(None, None, None)
        elif new_mode == "normal":
            self._running_mode_exit = lambda: None
        elif new_mode == "no_grad":
            no_grad_context = torch.no_grad()
            no_grad_context.__enter__()
            self._running_mode_exit = lambda: no_grad_context.__exit__(None, None, None)
        elif new_mode == "grad":
            grad_context = torch.enable_grad()
            grad_context.__enter__()
            self._running_mode_exit = lambda: grad_context.__exit__(None, None, None)
        else:
            raise ValueError(new_mode)
        self._running_mode = new_mode

    def make_env(self, stream: torch.cuda.Stream):
        self._check_tid()
        return RunContext(self, stream)


# thread_context = _ThreadContext()

_thread_contexts: Dict[int, _ThreadContext] = {}


def get_thread_context() -> _ThreadContext:
    tid = threading.get_ident()
    ctx = _thread_contexts.get(tid, None)
    if ctx is not None:
        return ctx
    ctx = _ThreadContext()
    _thread_contexts[tid] = ctx
    return ctx
