import math
# from auto_gptq.nn_modules.qlinear.qlinear_triton import QuantLinear as TritonQuantLinear  # type:ignore
# from auto_gptq.nn_modules.qlinear.qlinear_triton import quant_matmul_248, transpose_quant_matmul_248, quant_matmul_inference_only_248  # type:ignore
from typing import Any, Callable, List, Dict, Literal, cast
import torch
from .real import Linear, LinearMaker

TritonQuantLinear=Any
AutoGPTQQuantLinearsType = Literal["triton", "exllama_v2", "exllama_v1", "cuda_old", "cuda"]


class QuantLinearHolder(LinearMaker):
    def __init__(self, group_size: int, bits: int, desc_act: bool, quant_linear_type: AutoGPTQQuantLinearsType) -> None:
        self.maker = \
            self.select_quant_linear(desc_act=desc_act, group_size=group_size, bits=bits, quant_linear_type=quant_linear_type)
        self.group_size = group_size
        self.bits: int = bits
        self.use_act_order = desc_act
        self.quant_linear_type = quant_linear_type
        # self.trainable = (quant_linear_type != AutoGPTQQuantLinearsType.ExllamaV1) and (quant_linear_type != AutoGPTQQuantLinearsType.ExllamaV2)
        self.use_triton = quant_linear_type == "triton"
        self.quant_linears: list[Linear] = []

    def select_quant_linear(self, desc_act: bool, group_size: int, bits: int, quant_linear_type: AutoGPTQQuantLinearsType) -> Callable[[int, int, bool], Linear]:
        if quant_linear_type == "triton":
            from auto_gptq.nn_modules.qlinear.qlinear_triton import QuantLinear  # type:ignore

            def _make(infeatures: int, outfeatures: int, bias: bool) -> Linear:
                return cast(Linear, QuantLinear(bits=self.bits, group_size=self.group_size, infeatures=infeatures, outfeatures=outfeatures, bias=bias, trainable=True))
            return _make
        elif quant_linear_type == "exllama_v2":
            assert bits == 4
            from auto_gptq.nn_modules.qlinear.qlinear_exllamav2 import QuantLinear  # type:ignore

            def _make(infeatures: int, outfeatures: int, bias: bool) -> Linear:
                return cast(Linear, QuantLinear(bits=self.bits, group_size=self.group_size, infeatures=infeatures, outfeatures=outfeatures, bias=bias, trainable=False))
            return _make
        elif quant_linear_type == "exllama_v1":
            assert bits == 4
            from auto_gptq.nn_modules.qlinear.qlinear_exllama import QuantLinear  # type:ignore

            def _make(infeatures: int, outfeatures: int, bias: bool) -> Linear:
                return cast(Linear, QuantLinear(bits=self.bits, group_size=self.group_size, infeatures=infeatures, outfeatures=outfeatures, bias=bias, trainable=False))
            return _make
        elif quant_linear_type == "cuda_old":
            assert (not desc_act) or (group_size == -1)
            from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear  # type:ignore

            def _make(infeatures: int, outfeatures: int, bias: bool) -> Linear:
                return cast(Linear, QuantLinear(bits=self.bits, group_size=self.group_size, infeatures=infeatures, outfeatures=outfeatures, bias=bias, trainable=False))
            return _make
        elif quant_linear_type == "cuda":
            from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear  # type:ignore

            def _make(infeatures: int, outfeatures: int, bias: bool) -> Linear:
                return cast(Linear, QuantLinear(bits=self.bits, group_size=self.group_size, infeatures=infeatures, outfeatures=outfeatures, bias=bias, trainable=False))
            return _make

    def post_init(self, max_input_length: int) -> List[Any]:
        return autogptq_post_init(self.quant_linears, self.use_act_order, max_input_length)
        # if self.use_triton:
        #     warm_up_tritons(self.quant_linears,transpose=True,seqlen=2048)

    def warmup(self, seqlen: int, warmup_history: Dict[str, Any]):
        if not self.use_triton:
            return
        warm_up_tritons(cast(List[TritonQuantLinear], self.quant_linears), transpose=True, seqlen=seqlen, warmup_history=warmup_history)

    def make_linear(self, in_features: int, out_features: int, bias: bool = False) -> Linear:
        linear = self.maker(in_features, out_features, bias)
        self.quant_linears.append(linear)
        return linear


def warm_up_tritons(models: list[TritonQuantLinear], transpose: bool, seqlen: int, warmup_history: Dict[str, Any]):
    from tqdm import tqdm
    kn_values = {}
    device = None
    for m in models:
        k = m.infeatures
        n = m.outfeatures
        if (k, n) not in kn_values:
            warmup_key = f"triton_warmup_quant_linear_{k}_{n}"
            if warmup_key in warmup_history.keys():
                if warmup_history[warmup_key] >= seqlen:
                    continue
            warmup_history[warmup_key] = seqlen
            kn_values[(k, n)] = (m.qweight, m.scales, m.qzeros, m.g_idx, m.bits, m.maxq)
            if device is None:
                device = m.qweight.device
    if len(kn_values) == 0:
        return
    with torch.no_grad():
        for m in tqdm(range(0, math.ceil(math.log2(seqlen)) + 1)):
            m = 2 ** m
            for (k, n), (qweight, scales, qzeros, g_idx, bits, maxq) in kn_values.items():
                if transpose:
                    a = torch.randn(m, k, dtype=torch.float16, device=device)
                    quant_matmul_248(a, qweight, scales, qzeros, g_idx, bits, maxq)
                    a = torch.randn(m, n, dtype=torch.float16, device=device)
                    transpose_quant_matmul_248(a, qweight, scales, qzeros, g_idx, bits, maxq)
                else:
                    a = torch.randn(m, k, dtype=torch.float16, device=device)
                    quant_matmul_inference_only_248(a, qweight, scales, qzeros, g_idx, bits, maxq)
    del kn_values


def autogptq_post_init(models: List[Linear], use_act_order: bool, max_input_length: int = 2048) -> List[Any]:
    """
    The max_input_length argument is specific to the exllama backend, that requires to initialize a buffer temp_state.
    """
    persistent_datas: List[Any] = []
    device_to_buffers_size = {}

    model_uses_exllama = False
    for submodule in models:
        if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllama":
            model_uses_exllama = True
            device = submodule.qweight.device
            if device not in device_to_buffers_size:
                device_to_buffers_size[device] = {
                    "max_dq_buffer_size": 1,
                    "max_inner_outer_dim": 1
                }

            if not use_act_order:
                submodule._use_act_order = False  # type:ignore
            else:
                submodule._use_act_order = True  # type:ignore

            # Disable this heuristic for detecting act_order, but it could be used instead of the config.
            """
            if submodule.g_idx is None:
                submodule.act_order = False
            elif submodule.g_idx is not None and ((submodule.g_idx == 0).all() or torch.equal(submodule.g_idx.cpu(), torch.tensor([i // submodule.group_size for i in range(submodule.g_idx.shape[0])], dtype=torch.int32))):
                submodule.g_idx = None
                submodule.act_order = False
            else:
                submodule.act_order = True
            """

            device_to_buffers_size[device]["max_dq_buffer_size"] = max(device_to_buffers_size[device]["max_dq_buffer_size"], submodule.qweight.numel() * 8)

            if use_act_order:
                device_to_buffers_size[device]["max_inner_outer_dim"] = max(device_to_buffers_size[device]["max_inner_outer_dim"], submodule.infeatures, submodule.outfeatures)

    if model_uses_exllama:
        # To be honest this is quite ugly, not proud of this.
        from exllama_kernels import prepare_buffers, set_tuning_params

        device_to_buffers = {}

        if use_act_order:
            max_input_len = max_input_length
        else:
            if max_input_length is not None:
                print("Using exllama backend without act-order, the parameter max_input_length was set although not needed, it will be ignored.")
            max_input_len = 1

        for device, buffers_size in device_to_buffers_size.items():
            # The temp_state buffer is required to reorder X in the act-order case.
            # The temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
            device_to_buffers[device] = {
                "temp_state": torch.zeros((max_input_len, buffers_size["max_inner_outer_dim"]), dtype=torch.float16, device=device),
                "temp_dq": torch.zeros((1, buffers_size["max_dq_buffer_size"]), dtype=torch.float16, device=device),
                "max_dq_buffer_size": buffers_size["max_dq_buffer_size"],
                "max_inner_outer_dim": buffers_size["max_inner_outer_dim"],
            }

        persistent_datas.append(device_to_buffers)

        # Buffers need to be persistent to avoid any bug.
        for device, buffers in device_to_buffers.items():
            prepare_buffers(device, buffers["temp_state"], buffers["temp_dq"])

        # Using the default from exllama repo here.
        matmul_recons_thd = 8
        matmul_fused_remap = False
        matmul_no_half2 = False
        set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

        # The buffers need to have been initialized first before calling make_q4.
        for submodule in models:
            if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllama":
                submodule.post_init()

    # exllamav2
    fixed_bytes = {}
    model_uses_exllamav2 = False

    for submodule in models:
        if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllamav2":
            model_uses_exllamav2 = True
            device = submodule.qweight.device
            scratch_fixed = submodule.scratch_space_fixed()
            fixed_bytes[device] = max(scratch_fixed, fixed_bytes.get(device, 0))

    if model_uses_exllamav2:
        from auto_gptq.nn_modules.qlinear.qlinear_exllamav2 import ExLlamaV2DeviceTensors  # type:ignore
        device_tensors = {}
        for device, scratch_bytes in fixed_bytes.items():
            device_tensors[device] = ExLlamaV2DeviceTensors(device.index, scratch_bytes)

        # have persistent buffers, otherwise we will get OOM
        persistent_datas.append(device_tensors)

        for submodule in models:
            if hasattr(submodule, "QUANT_TYPE") and submodule.QUANT_TYPE == "exllamav2":
                device = submodule.qweight.device
                submodule.post_init(temp_dq=device_tensors[device])
    torch.cuda.empty_cache()

    return persistent_datas
