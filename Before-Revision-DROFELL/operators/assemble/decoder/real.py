from typing import Callable, Dict, List, Optional, Tuple, Union
import torch

from cuda_extends.exllama import CacheV2, TuningParams
from operators import flags
from operators.container import Container
from operators.helpers.cuda_utils import torch_cuda
from ...core.block.real import LayerBlocksHolder
from ...core.decoder_wrap.real_cache import FuseCache
from ...core.decoder_wrap.real_common import DecoderWrap
from ...core.decoder_wrap.real_fuse import DecoderWrapFuse
from ...core.decoder_wrap.common import DecoderConfig, LoraHook
from ...core.attention.real_page import PagedAttention
from ...core.rope.real import RotaryEmbedding
from ...core.linear.real import LinearMaker
from ...core.attention.real_flash import FlashAttention


class Decoder(DecoderWrap):
    def __init__(self, config: DecoderConfig, linear_maker: LinearMaker, rotary_emb: RotaryEmbedding) -> None:
        DecoderWrap.__init__(self=self, config=config, linear_maker=linear_maker, rotary_emb=rotary_emb)
        self.flash_attention = FlashAttention(config=config.attention_config)
        self.paged_attention = PagedAttention(config=config.attention_config)
        self.fuse_wrap: Optional[DecoderWrapFuse] = None
        self.multi_prefill_streams: List[torch.cuda.Stream] = []
        self.config = config

    def fuse(self, fuse_cache: FuseCache, quant_linear_cache: CacheV2, tuning_params: TuningParams):
        self.fuse_wrap = DecoderWrapFuse(self, fuse_cache=fuse_cache, quant_linear_cache=quant_linear_cache, tuning_params=tuning_params)

    def inference_forward_without_attention(
        self,
        hidden_states: torch.Tensor, position_ids: torch.Tensor,
        atten: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        lora_hook: Optional[LoraHook],
        fuse: bool
    ):
        """
        atten(q,k,v)->o  
        common:
            hidden_states=[...,hidden_size]  
            q,k,v,o = [flatten_seq_len/seq_len,num_heads,head_dim]  
            out=[...,hidden_size] 
        fuse:
            hidden_states=[bsz,1,hidden_size]  
            q,k,v,o = [flatten_seq_len/seq_len,num_heads,head_dim]  
            out=[bsz,1,hidden_size]   
        """
        if fuse:
            assert self.fuse_wrap is not None
            return self.fuse_wrap.fuse_forward_without_attention(hidden_states=hidden_states, position_ids=position_ids, atten=atten, lora_hook=lora_hook)
        else:
            query_states, key_states, value_states, residual = self.inference_forward_before_stage(hidden_states, position_ids, lora_hook=lora_hook)
            out = atten(query_states, key_states, value_states)
            out = self.inference_forward_after_stage(out, residual)
            return out

    async def multi_seqs_finetuning_forward(
        self, container: Container,
        hidden_states: torch.Tensor, position_ids: torch.Tensor, input_lens: list[int],
        ptuning_kv_cache: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, None],
        lora_hook: Optional[LoraHook],
    ):
        atten = self.create_attention_enclosure(input_lens, ptuning_kv_cache)
        query_states, key_states, value_states, residual = await self.finetuning_forward_before_stage(container=container, hidden_states=hidden_states, position_ids=position_ids, lora_hook=lora_hook)
        out = await container.step("atten", atten, query_states=query_states, key_states=key_states, value_states=value_states)
        out = await self.finetuning_forward_after_stage(container=container, hidden_states=out, residual=residual)
        return out

    def create_attention_enclosure(self, input_lens: list[int], ptuning_kv_cache: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, None],):
        def attention(query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
            attention_outputs: List[torch.Tensor] = []
            start_pos = 0
            for input_len in input_lens:
                q = query_states[start_pos:start_pos + input_len]
                if ptuning_kv_cache is not None:
                    k = torch.cat([ptuning_kv_cache[0], key_states[start_pos:start_pos + input_len]], dim=0)
                    v = torch.cat([ptuning_kv_cache[1], value_states[start_pos:start_pos + input_len]], dim=0)
                else:
                    k = key_states[start_pos:start_pos + input_len]
                    v = value_states[start_pos:start_pos + input_len]
                attention_output: torch.Tensor = self.flash_attention.forward(q, k, v, multi_seq=False)
                attention_outputs.append(attention_output)
                start_pos += input_len
            return torch.concat(attention_outputs, dim=0)
        return attention

    def create_attention_forward_and_backward_enclosure(self, container: Container, input_lens: list[int], ptuning_kv_cache: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, None],):
        def single_seq_atten_fwd(query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, start_pos: int, input_len: int):
            q = query_states[start_pos:start_pos + input_len]
            if ptuning_kv_cache is not None:
                k = torch.cat([ptuning_kv_cache[0], key_states[start_pos:start_pos + input_len]], dim=0)
                v = torch.cat([ptuning_kv_cache[1], value_states[start_pos:start_pos + input_len]], dim=0)
            else:
                k = key_states[start_pos:start_pos + input_len]
                v = value_states[start_pos:start_pos + input_len]
            attention_output: torch.Tensor = self.flash_attention.forward(q, k, v, multi_seq=False)
            attention_outputd = attention_output.detach()  # .requires_grad_(True)
            return attention_output, attention_outputd

        def single_seq_atten_bwd(attention_output: torch.Tensor, grad: torch.Tensor, start_pos: int, input_len: int):
            attention_output.backward(gradient=grad[start_pos:start_pos + input_len])

        async def forward(query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor):
            attention_outputs: List[torch.Tensor] = []
            saved_seq: List[torch.Tensor] = []
            start_pos = 0
            for _, input_len in enumerate(input_lens):
                attention_output, attention_outputd = await container.step("single_seq_atten_fwd", single_seq_atten_fwd, query_states=query_states, key_states=key_states, value_states=value_states, start_pos=start_pos, input_len=input_len)
                attention_outputs.append(attention_outputd)
                saved_seq.append(attention_output)
                start_pos += input_len
            result = await container.step("concat_atten_fwd", torch.concat, tensors=attention_outputs, dim=0)
            return result, saved_seq

        async def backward(saved_qkv: List[torch.Tensor], grad: torch.Tensor):
            # result.backward(gradient=grad)
            # await flags.PTUNING_ASYNC_YIELD("per seq atten bwd split")
            start_pos = 0
            # Backward through each block if needed
            for seq_i, (input_len, attention_output) in enumerate(zip(input_lens, saved_qkv)):
                await container.step("single_seq_atten_bwd", single_seq_atten_bwd, attention_output=attention_output, grad=grad, start_pos=start_pos, input_len=input_len)
                start_pos += input_len
        return forward, backward

    def multi_seqs_prefill_forward(
        self,
        hidden_states: torch.Tensor, position_ids: torch.Tensor,
        input_lens: list[int],
        ptunings_getter: Callable[[int], Optional[Tuple[torch.Tensor, torch.Tensor]]],
        kv_cacher: Callable[[torch.Tensor, torch.Tensor], None],
        lora_hook: Optional[LoraHook],
    ):
        """
        hidden_states=[flatten_seq_len,hidden_size]
        """

        def _enclosure(seq_i: int, start_pos: int, input_len: int, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor):
            q = query_states[start_pos:start_pos + input_len]
            k = key_states[start_pos:start_pos + input_len]
            v = value_states[start_pos:start_pos + input_len]
            ptuning_kv_cache = ptunings_getter(seq_i)
            if ptuning_kv_cache is not None:
                k = torch.cat([ptuning_kv_cache[0], k], dim=0)
                v = torch.cat([ptuning_kv_cache[1], v], dim=0)
            attention_output: torch.Tensor = self.flash_attention.forward(q, k, v, multi_seq=False)
            return attention_output
        curr_stream = torch.cuda.current_stream()

        def attention(query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
            kv_cacher(key_states, value_states)
            attention_outputs: List[torch.Tensor] = []
            start_pos = 0

            if flags.MULTI_PREFILL_STREAM:
                events: List[torch.cuda.Event] = []
                for seq_i, input_len in enumerate(input_lens):
                    if seq_i == len(self.multi_prefill_streams):
                        self.multi_prefill_streams.append(torch_cuda.new_stream(self.device))
                    s = self.multi_prefill_streams[seq_i]
                    s.wait_stream(curr_stream)
                    with torch.cuda.stream(s):
                        attention_output: torch.Tensor = _enclosure(
                            seq_i, start_pos, input_len,
                            query_states, key_states, value_states)
                        attention_outputs.append(attention_output)
                        ev = torch_cuda.new_event(False)
                        ev.record()
                        events.append(ev)
                    start_pos += input_len
                for ev in events:
                    ev.wait()
            else:
                for seq_i, input_len in enumerate(input_lens):
                    attention_output: torch.Tensor = _enclosure(
                        seq_i, start_pos, input_len,
                        query_states, key_states, value_states)
                    attention_outputs.append(attention_output)
                    ev = torch_cuda.new_event(False)
                    ev.record()
                    start_pos += input_len
            return torch.concat(attention_outputs, dim=0)
        return self.inference_forward_without_attention(hidden_states=hidden_states, position_ids=position_ids, atten=attention, fuse=False, lora_hook=lora_hook)

    def generating_forward(
        self,
        hidden_states: torch.Tensor, position_ids: torch.Tensor,
        kv_cacher: LayerBlocksHolder,
        context_len_max: int,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        slot_mapping: torch.Tensor,
        lora_hook: Optional[LoraHook],
    ):
        def attention(query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
            kv_cacher.kv_tensor_to_slots_with_convert(key_states=key_states, val_states=value_states, slot_mapping=slot_mapping)
            return self.paged_attention.forward(query=query_states, context_len_max=context_len_max, context_lens=context_lens, block_tables=block_tables, kv_cacher=kv_cacher)
        ret = self.inference_forward_without_attention(hidden_states=hidden_states, position_ids=position_ids, atten=attention, fuse=self.fuse_wrap is not None, lora_hook=lora_hook)
        return ret

    def load_parameters(self, state_dict: Dict[str, torch.Tensor]):
        # check bias removed
        for k, v in state_dict.items():
            if k.endswith(".bias"):
                assert torch.sum(v) == 0
        raw_state_dict = {k: v for k, v in state_dict.items() if not k.endswith(".bias")}
        # check inv freq
        if "self_attn.rotary_emb.inv_freq" in raw_state_dict:
            assert torch.allclose(self.rotary_emb._buffers["inv_freq"], raw_state_dict["self_attn.rotary_emb.inv_freq"].to(self.device), atol=5e-4)  # type: ignore
            # print(rotary_emb._buffers["inv_freq"], raw_state_dict["self_attn.rotary_emb.inv_freq"])
            del raw_state_dict["self_attn.rotary_emb.inv_freq"]
        self.load_state_dict(raw_state_dict)
