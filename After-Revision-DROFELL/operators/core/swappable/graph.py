import torch
from torch.autograd import Function
from typing import Any, Dict, List, Optional, Set, Tuple, cast
from cuda_extends.hack import tensor_as_leaf
from .real import CanSwapObject

CPU = torch.device("cpu")


class HijackGraph(CanSwapObject):
    class DropToCPUFunction(Function):
        actual_grad_lookup: Dict[int, torch.Tensor] = {}

        @staticmethod
        def forward(ctx: Any, actual_tensor: torch.Tensor) -> torch.Tensor:
            fake_holder: torch.Tensor = torch.empty((0,), device=CPU)
            name = id(fake_holder)
            ctx.name = name
            return fake_holder

        @staticmethod
        def backward(ctx: Any, fake_device_grad: torch.Tensor):  # type:ignore
            name = ctx.name
            actual_grad = HijackGraph.DropToCPUFunction.actual_grad_lookup[name]
            del HijackGraph.DropToCPUFunction.actual_grad_lookup[name]
            return actual_grad

        @staticmethod
        def set_actual_grad(name: int, grad: torch.Tensor):
            HijackGraph.DropToCPUFunction.actual_grad_lookup[name] = grad

    @staticmethod
    def get_module_address(module: torch.nn.Module, addrs: Optional[Set[int]] = None) -> Set[int]:
        if addrs is None:
            addrs = set()
        for _, parameter in module.named_parameters():
            addrs.add(parameter.data_ptr())
        for _, buffer in module.named_buffers():
            addrs.add(buffer.data_ptr())
        for _, submodule in module.named_children():
            HijackGraph.get_module_address(submodule, addrs)
        return addrs

    def __init__(self, non_free_addrs: Set[int]) -> None:
        self.pack_id = 0
        self.graph: Dict[int, torch.Tensor] = {}
        self.non_free_graph: Dict[int, torch.Tensor] = {}
        self.output_grad_idx: int
        self.output_fake_holder: torch.Tensor
        self._swapped = False
        self._expect_device = True
        self.enable_non_free_support = True
        self.non_free_addrs = non_free_addrs

    def _pack_hook(self, x: torch.Tensor):
        self.pack_id += 1
        pack_id: int = self.pack_id
        self.graph[pack_id] = x
        return pack_id

    def _unpack_hook(self, pack_id: int):
        if pack_id in self.non_free_graph:
            x = self.non_free_graph[pack_id]
        else:
            x = self.graph[pack_id]
        # print(f"unpack {pack_id} as {x}")
        return x

    def hijack_tensor(self):
        return torch.autograd.graph.saved_tensors_hooks(self._pack_hook, self._unpack_hook)

    def set_output_t(self, t: torch.Tensor):
        self.output_fake_holder = cast(torch.Tensor, HijackGraph.DropToCPUFunction.apply(t))
        name = id(self.output_fake_holder)
        self.output_grad_idx = name

    def backward(self, grad: torch.Tensor):
        HijackGraph.DropToCPUFunction.set_actual_grad(self.output_grad_idx, grad)
        torch.autograd.backward(self.output_fake_holder, torch.empty(size=(0,), device=CPU))

    def _swap_out(self):
        with torch.no_grad():
            if self.enable_non_free_support:
                shared_keys: List[int] = []
                for k, v in self.graph.items():
                    ptr = v.data_ptr()
                    # Empty tensors have data_ptr()==0 regardless of storage
                    # ownership. Never use that sentinel to identify a shared
                    # module buffer: Marlin's no-act-order sort index is an
                    # empty CUDA tensor and would otherwise collide with every
                    # other empty tensor captured by saved_tensors_hooks.
                    if ptr != 0 and ptr in self.non_free_addrs:
                        self.non_free_graph[k] = tensor_as_leaf(v)
                        shared_keys.append(k)
                    else:
                        self.graph[k] = v.to(CPU, non_blocking=True)
                # print(f"hit non free keys: {shared_keys} in swapping out")
                for k in shared_keys:
                    del self.graph[k]
            else:
                self.graph = {
                    k: v.to(CPU, non_blocking=True)
                    for k, v in self.graph.items()
                }

    def swap_out(self, stream: Optional[torch.cuda.Stream]):
        if stream is None:
            self._swap_out()
        else:
            with torch.cuda.stream(stream):
                self._swap_out()
        self._swapped = True
        self._expect_device = False

    def _swap_in(self, device: torch.device):
        with torch.no_grad():
            self.graph = {
                k: v.to(device, non_blocking=True)
                for k, v in self.graph.items()
            }

    def swap_in(self, device: torch.device, stream: Optional[torch.cuda.Stream]):
        if stream is None:
            self._swap_in(device)
        else:
            with torch.cuda.stream(stream):
                self._swap_in(device)
        self._swapped = True
        self._expect_device = True

    @property
    def swapped(self) -> bool:
        return self._swapped

    @property
    def expect_device(self) -> bool:
        return self._expect_device


class HijackGraphWithMultiGradTensors(HijackGraph):
    class ProxyFunction(Function):
        _tmp_t: Optional[torch.Tensor] = None
        _tmp_counter: Optional[int] = None
        grads: Dict[int, Any] = {}
        counter: int = 0

        @staticmethod
        def forward(ctx: Any, placeholder_tensor: torch.Tensor) -> torch.Tensor:
            assert HijackGraphWithMultiGradTensors.ProxyFunction._tmp_counter is not None
            assert HijackGraphWithMultiGradTensors.ProxyFunction._tmp_t is not None
            ctx.name = HijackGraphWithMultiGradTensors.ProxyFunction._tmp_counter
            actual_t = HijackGraphWithMultiGradTensors.ProxyFunction._tmp_t
            del HijackGraphWithMultiGradTensors.ProxyFunction._tmp_t
            del HijackGraphWithMultiGradTensors.ProxyFunction._tmp_counter
            ctx.device = actual_t.device
            return actual_t

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None, None]:  # type: ignore
            HijackGraphWithMultiGradTensors.ProxyFunction.grads[ctx.name] = grad_output
            victim = torch.empty((0,), dtype=torch.long, device=CPU)
            return victim, None, None, None

        def __init__(self, actual_t: torch.Tensor):
            assert actual_t.grad is None
            assert actual_t.requires_grad

            HijackGraphWithMultiGradTensors.ProxyFunction.counter += 1
            self.idx = HijackGraphWithMultiGradTensors.ProxyFunction.counter
            HijackGraphWithMultiGradTensors.ProxyFunction._tmp_counter = self.idx
            HijackGraphWithMultiGradTensors.ProxyFunction._tmp_t = actual_t

            proxy = torch.empty((0,), device=CPU)
            proxy.requires_grad_(True)
            self.proxied = cast(torch.Tensor, HijackGraphWithMultiGradTensors.ProxyFunction.apply(proxy))

            self.proxied_taken = False
            self.grad_taken = False

        @property
        def tensor(self) -> torch.Tensor:
            assert not self.proxied_taken
            self.proxied_taken = True
            p = self.proxied
            del self.proxied
            return p

        @property
        def grad(self) -> torch.Tensor:
            assert not self.grad_taken
            self.grad_taken = True
            t = HijackGraphWithMultiGradTensors.ProxyFunction.grads[self.idx]
            del HijackGraphWithMultiGradTensors.ProxyFunction.grads[self.idx]
            return t

    def __init__(self, tensors: List[torch.Tensor], addrs: Set[int]) -> None:
        super().__init__(addrs)
        self.proxy_tensors = [HijackGraphWithMultiGradTensors.ProxyFunction(t) for t in tensors]
        self.proxied_taken = False
        self.grad_taken = False

    @property
    def tensors(self) -> List[torch.Tensor]:
        assert not self.proxied_taken
        self.proxied_taken = True
        ps = [p.tensor for p in self.proxy_tensors]
        return ps

    @property
    def grads(self) -> List[torch.Tensor]:
        assert not self.grad_taken
        self.grad_taken = True
        gs = [p.grad for p in self.proxy_tensors]
        return gs


if __name__ == "__main__":
    from operators.helpers.cuda_utils.memory import MemoryDeltaStatus
    from torch import nn

    device = torch.zeros((0,), device="cuda").device
    cpu = torch.zeros((0,), device="cpu").device

    @torch.no_grad()
    def init_weights(m: torch.nn.Module):
        if type(m) == torch.nn.Linear:
            m.weight.fill_(1.0)

    ptuning_linear = torch.nn.Linear(2, 2, False, device=device).apply(init_weights)
    ptuning_drop = torch.nn.Dropout(0.5)

    addrs: Set[int] = set()

    addrs = HijackGraph.get_module_address(ptuning_linear)\
        .union(HijackGraph.get_module_address(ptuning_drop))
    print("module non free addrs: ", addrs)

    def ptuning_forward():
        x = (torch.ones((2), device=device)*2)
        hijack_graph = HijackGraph(addrs)

        with hijack_graph.hijack_tensor():
            torch.manual_seed(42)
            ptuning_kv = ptuning_drop(ptuning_linear(x))
            hijack_graph.set_output_t(ptuning_kv)
        return hijack_graph

    def warmup():
        hijack_graph = ptuning_forward()
        grad = torch.ones((2), device=device)*4
        hijack_graph.backward(grad)
        ptuning_linear.weight.grad.zero_()

    warmup()
    print("warm up done")
    print("="*80)

    torch.cuda.memory._record_memory_history()
    hijack_graph = ptuning_forward()
    # snapshot_after_forward = torch.cuda.memory._snapshot()
    # trace_after_forward = TraceRecord(snapshot_after_forward['device_traces'][0])
    # trace_after_forward.print_current_frames()

    hijack_graph.swap_out(stream=None)

    snapshot_after_swapout = torch.cuda.memory._snapshot()
    trace_after_swapout = MemoryDeltaStatus(snapshot_after_swapout['device_traces'][0])
    trace_after_swapout.print_current_frames()

    hijack_graph.swap_in(device=device, stream=None)

    grad = torch.ones((2), device=device)*4
    hijack_graph.backward(grad)
    print("grad:", ptuning_linear.weight.grad)
    print("="*80)

    class Trunk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_x1 = torch.nn.Linear(
                2, 2, False, device=device
            )
            self.linear_x2 = torch.nn.Linear(
                2, 2, False, device=device
            )
            self.linear_3 = torch.nn.Linear(
                2, 2, False, device=device
            )
            self.linear_4 = torch.nn.Linear(
                2, 2, False, device=device
            )

            @torch.no_grad()
            def init_weights(m: torch.nn.Module):
                if type(m) == torch.nn.Linear:
                    m.weight.fill_(10.0)

            self.apply(init_weights)
            self.freeze(self)

            self.non_free_addrs = HijackGraph.get_module_address(self)

        @staticmethod
        def freeze(module: nn.Module):
            module = module.eval()
            for _, parameter in module.named_parameters():
                parameter.requires_grad_(False)
            for _, buffer in module.named_buffers():
                buffer.requires_grad_(False)
            for _, submodule in module.named_children():
                Trunk.freeze(submodule)

        def forward(self, x1: torch.Tensor, x2: torch.Tensor):
            x1o = self.linear_x1(x1)
            x2o = self.linear_x2(x2)
            x = x1o*x2o
            x = self.linear_3(x)
            x = self.linear_4(x)
            return x

    trunk = Trunk()
    addrs = addrs.union(trunk.non_free_addrs)
    print("module non free addrs: ", addrs)

    def forward():
        x1 = (torch.ones((2,), device="cuda")*2).requires_grad_(True)
        x2 = (torch.ones((2,), device="cuda")*3).requires_grad_(True)

        graph_with_tensors = HijackGraphWithMultiGradTensors([x1, x2], addrs)
        with graph_with_tensors.hijack_tensor():
            x1p, x2p = graph_with_tensors.tensors
            o = trunk(x1p, x2p)
            graph_with_tensors.set_output_t(o)
        return graph_with_tensors

    def warmup():
        hijack_graph = forward()
        grad = torch.ones((2), device=device)*4
        hijack_graph.backward(grad)

    warmup()
    print("warm up done")
    print("="*80)

    torch.cuda.memory._record_memory_history()
    graph_with_tensors = forward()
    graph_with_tensors.swap_out(stream=None)
    snapshot_after_swapout = torch.cuda.memory._snapshot()
    trace_after_swapout = MemoryDeltaStatus(snapshot_after_swapout['device_traces'][0])
    trace_after_swapout.print_current_frames()

    graph_with_tensors.swap_in(device=device, stream=None)

    grad = torch.ones((2), device=device)*4
    graph_with_tensors.backward(grad)
    print(graph_with_tensors.grads)
    print("="*80)
