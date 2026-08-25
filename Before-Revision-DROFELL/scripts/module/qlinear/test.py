import json
import os
import pickle
import time
from typing import Any, Dict, List, Literal, Tuple, Callable, Union
import numpy
import torch
from safetensors import safe_open  # type:ignore
from operators.core.linear.real_auto_gptq import AutoGPTQQuantLinearsType
from operators.core.linear.real_auto_gptq import QuantLinearHolder as AutoGPTQLinearHolder
from operators.core.linear.real_quant import CacheV2, TuningParams
from operators.core.linear.real_quant import QuantLinearMaker as OurLinearHolder
from operators.helpers.configs import LlamaConfig
from operators import flags

qlinear_data = "scripts/converts/qlinear_data"
test_config = os.path.join(qlinear_data, "config.json")

with open(test_config, "r") as f:
    config_dict = json.load(f)
    config = LlamaConfig(**config_dict)

device = torch.zeros((0,)).cuda().device
max_len = 2**14  # 16K

persist: List[Any] = []


def make_qlinears(device: torch.device, method: Union[AutoGPTQQuantLinearsType, Literal["our_cache", "our_no_cache"]], persist: List[Any]):
    with torch.device(device):
        post_init: Callable[[], None] = lambda: None
        if method != "our_cache" and method != "our_no_cache":
            linear_maker = AutoGPTQLinearHolder(
                group_size=config.quantization_config.group_size,
                bits=config.quantization_config.bits,
                desc_act=config.quantization_config.desc_act,
                quant_linear_type=method,
            )

            def _post_init():
                persist.append(linear_maker.post_init(max_input_length=max_len))
                linear_maker.warmup(seqlen=max_len, warmup_history={})
            post_init = _post_init
        else:
            assert method.startswith("our")
            cache = CacheV2(device)
            tunning_params = TuningParams()
            linear_maker = OurLinearHolder(quantization_group_size=config.quantization_config.group_size, use_act_order=config.quantization_config.desc_act)

            def _post_init():
                linear_maker.post_init(tunning_params, cache)

                if method.endswith("no_cache"):
                    flags.REUSE_RECONSTRUCT_CACHE_WHEN_GRAD = False
                    # linear_maker.set_forward_method(QUANT_FORWARD_METHOD.EXLLAMA_NO_CACHE)
                elif method.endswith("cache"):
                    flags.REUSE_RECONSTRUCT_CACHE_WHEN_GRAD = True
                    # linear_maker.set_forward_method(QUANT_FORWARD_METHOD.EXLLAMA_WITH_CACHE)
                else:
                    assert False, "must be sth like our_cache, our_no_cache"
                cache.init_cache()
            post_init = _post_init

        def load_weight(weight_name: str, module: torch.nn.Module):
            state_dict: Dict[str, torch.Tensor] = {}
            with safe_open(os.path.join(qlinear_data, weight_name+".safetensors"), framework="pt") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
            dropped_dict: Dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                if k.endswith("bias"):
                    assert torch.sum(v) == 0
                    continue
                dropped_dict[k] = v
            module.load_state_dict(state_dict=dropped_dict)

        sized_linear: Dict[str, Tuple[int, int, torch.nn.Module]] = {}

        q_proj = linear_maker.make_linear(config.hidden_size, config.num_query_heads * config.head_dim, bias=False)
        load_weight("q_proj", q_proj)
        sized_linear["q"] = (config.hidden_size, config.num_query_heads * config.head_dim, q_proj)
        k_proj = linear_maker.make_linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        load_weight("k_proj", k_proj)
        sized_linear["k"] = (config.hidden_size, config.num_key_value_heads * config.head_dim, k_proj)
        v_proj = linear_maker.make_linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        load_weight("v_proj", v_proj)
        sized_linear["v"] = (config.hidden_size, config.num_key_value_heads * config.head_dim,  v_proj)
        o_proj = linear_maker.make_linear(config.num_query_heads * config.head_dim, config.hidden_size, bias=False)
        load_weight("o_proj", o_proj)
        sized_linear["o"] = (config.num_key_value_heads * config.head_dim, config.hidden_size, o_proj)
        gate_proj = linear_maker.make_linear(config.hidden_size, config.decoder_config.intermediate_size, bias=False)
        load_weight("gate_proj", gate_proj)
        sized_linear["gate"] = (config.hidden_size, config.decoder_config.intermediate_size, gate_proj)
        up_proj = linear_maker.make_linear(config.hidden_size, config.decoder_config.intermediate_size, bias=False)
        load_weight("up_proj", up_proj)
        sized_linear["up"] = (config.hidden_size, config.decoder_config.intermediate_size, up_proj)
        down_proj = linear_maker.make_linear(config.decoder_config.intermediate_size, config.hidden_size, bias=False)
        load_weight("down_proj", down_proj)
        sized_linear["down"] = (config.decoder_config.intermediate_size, config.hidden_size, down_proj)
        post_init()
        torch.cuda.empty_cache()
    return sized_linear


def test_multi_seqlen(test_once: Callable[[int], float], series: Dict[str, Any]):
    def test_one_size(seq_len: int) -> List[float]:
        a = [test_once(seq_len) for _ in range(10)]  # warm up
        b = [test_once(seq_len) for _ in range(10)]
        return [test_once(seq_len) for _ in range(10)]

    seq_len = 1
    records: List[Dict[str, Any]] = []
    while seq_len <= max_len:
        times = [v for v in test_one_size(seq_len)]
        records += [{
            "seq_len": seq_len,
            "time": t,
            **series
        } for t in times]
        mean = numpy.mean(a=times)
        std = numpy.std(times)
        print(f"{seq_len}: {mean/(10**3):.4f}({std/(10**3):.4f})µs ")
        seq_len = seq_len*2
    return records


def make_forward_test_once(module: torch.nn.Module, in_size: int, repeat_times: int, dtype: torch.dtype):
    def test_once(seq_len: int) -> float:
        module.eval()
        with torch.inference_mode():
            input = torch.empty((seq_len, in_size), dtype=dtype, device=device)
            st = time.time_ns()
            for _ in range(repeat_times):
                module(input)
            torch.cuda.synchronize()
            return (time.time_ns()-st)/repeat_times
    return test_once


def make_backward_test_once(module: torch.nn.Module, in_size: int, out_size: int, repeat_times: int, dtype: torch.dtype):
    def test_once(seq_len: int) -> float:
        module.train()
        input = torch.empty((seq_len, in_size), dtype=dtype, device=device, requires_grad=True)
        grad = torch.empty((seq_len, out_size), dtype=dtype, device=device)
        st = time.time_ns()
        for _ in range(repeat_times):
            out: torch.Tensor = module(input)
            torch.autograd.backward(out, grad)
        torch.cuda.synchronize()
        return (time.time_ns()-st)/repeat_times
    return test_once


all_record: List[Dict[str, Any]] = []


# workaround for Triton Error [CUDA]: invalid device context
# see https://github.com/pytorch/pytorch/issues/124565
torch.empty(1, device='cuda', requires_grad=True).backward()

print("our_cache")
in_size, out_size, module = make_qlinears(device, method="our_cache", persist=persist)["q"]
print(f"in_size: {in_size}, out_size: {out_size}")
test_once = make_forward_test_once(module, in_size, repeat_times=100, dtype=torch.float16)
all_record += test_multi_seqlen(test_once, {"method": "our_cache", "kind": "forward", "dtype": "float16"})
test_once = make_forward_test_once(module, in_size, repeat_times=100, dtype=torch.bfloat16)
all_record += test_multi_seqlen(test_once, {"method": "our_cache", "kind": "forward", "dtype": "bfloat16"})
test_once = make_backward_test_once(module, in_size, out_size=out_size, repeat_times=100, dtype=torch.float16)
all_record += test_multi_seqlen(test_once, {"method": "our_cache", "kind": "forward+backward", "dtype": "float16"})
test_once = make_backward_test_once(module, in_size, out_size=out_size, repeat_times=100, dtype=torch.bfloat16)
all_record += test_multi_seqlen(test_once, {"method": "our_cache", "kind": "forward+backward", "dtype": "bfloat16"})

print("our_no_cache")
in_size, out_size, module = make_qlinears(device, method="our_no_cache", persist=persist)["q"]
test_once = make_forward_test_once(module, in_size, repeat_times=100, dtype=torch.float16)
all_record += test_multi_seqlen(test_once, {"method": "our_no_cache", "kind": "forward", "dtype": "float16"})
test_once = make_forward_test_once(module, in_size, repeat_times=100, dtype=torch.bfloat16)
all_record += test_multi_seqlen(test_once, {"method": "our_no_cache", "kind": "forward", "dtype": "bfloat16"})
test_once = make_backward_test_once(module, in_size, out_size=out_size, repeat_times=100, dtype=torch.float16)
all_record += test_multi_seqlen(test_once, {"method": "our_no_cache", "kind": "forward+backward", "dtype": "float16"})
test_once = make_backward_test_once(module, in_size, out_size=out_size, repeat_times=100, dtype=torch.bfloat16)
all_record += test_multi_seqlen(test_once, {"method": "our_no_cache", "kind": "forward+backward", "dtype": "bfloat16"})

print("exllama_v1")
in_size, out_size, module = make_qlinears(device, "exllama_v1", persist)["q"]
test_once = make_forward_test_once(module, in_size, repeat_times=100, dtype=torch.float16)
all_record += test_multi_seqlen(test_once, {"method": "exllama_v1", "kind": "forward", "dtype": "float16"})


print("exllama_v2")
in_size, out_size, module = make_qlinears(device, "exllama_v2", persist)["q"]
test_once = make_forward_test_once(module, in_size, repeat_times=100, dtype=torch.float16)
all_record += test_multi_seqlen(test_once, {"method": "exllama_v2", "kind": "forward", "dtype": "float16"})

print("triton")
in_size, out_size, module = make_qlinears(device, "triton", persist)["q"]
test_once = make_forward_test_once(module, in_size, repeat_times=100, dtype=torch.float16)
all_record += test_multi_seqlen(test_once, {"method": "triton", "kind": "forward", "dtype": "float16"})
test_once = make_backward_test_once(module, in_size, out_size=out_size, repeat_times=100, dtype=torch.float16)
all_record += test_multi_seqlen(test_once, {"method": "triton", "kind": "forward+backward", "dtype": "float16"})

with open(os.path.join(os.path.dirname(__file__), "result.pkl"), "wb") as f:
    pickle.dump(all_record, f)
