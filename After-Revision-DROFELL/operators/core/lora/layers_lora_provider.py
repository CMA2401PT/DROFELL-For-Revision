import torch
from cuda_extends import hack
from cuda_extends.exllama import lora
import math
from typing import Dict, List, Optional, Tuple

from operators import flags

from .render import LoraHook, LoraRenderConfig
from ...assemble.decoders.common import LayersCacheProvider, LayersLoraHookProvider


class SimpleLayersLoraHookProvider(LayersLoraHookProvider):
    def __init__(
        self,
        seq_loras: List[Optional[Tuple[LoraRenderConfig, List[int]]]],  # [[(lora_config,blocks)]? * num_seqs],
        seq_lens: List[int],
        cache_provider: LayersCacheProvider,
    ) -> None:
        self.cache_provider = cache_provider

        self.grouped, self.loras = \
            self._squeeze_loras(seq_loras, seq_lens)

    def get_lora_hook(
        self,
        layer_i: int,
    ) -> Optional[LoraHook]:
        if len(self.loras) == 0:
            return None
        cacher = self.cache_provider.get_layer_with_event_ready(layer_i)
        device = cacher.layer_key_ptr_table_device.device

        lora_blocks: List[Tuple[LoraRenderConfig, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]] = []

        for cfg, lora_block_idx in self.loras:
            kv_size = cfg.kv_hidden_size
            kv_blocks = (cfg.index.total_saperate_blocks+1)//2

            blk_template = torch.empty((0,), device=device, dtype=torch.float16)
            bp: List[int] = [
                cacher.layer_key_ptr_table_list[lora_block_idx[i//2]]
                if i % 2 == 0 else
                cacher.layer_val_ptr_table_list[lora_block_idx[i//2]]
                for i in range(kv_blocks*2)
            ]
            bs: List[torch.Tensor] = [
                hack.tensor_from_pointer(p, blk_template, (flags.BLOCK_SIZE*kv_size,)).view(flags.BLOCK_SIZE, kv_size) for p in (bp)
            ]

            lq, lk, lv = None, None, None
            if cfg.index.q is not None:
                a_b, a_s, a_e, b_b, b_s, b_e = cfg.index.q[1]
                lq = bs[a_b][a_s:a_e].view(cfg.hidden_size, cfg.r_q), bs[b_b][b_s:b_e].view(cfg.r_q, cfg.hidden_size)
            if cfg.index.k is not None:
                a_b, a_s, a_e, b_b, b_s, b_e = cfg.index.k[1]
                lk = bs[a_b][a_s:a_e].view(cfg.hidden_size, cfg.r_k), bs[b_b][b_s:b_e].view(cfg.r_k, cfg.kv_hidden_size)
            if cfg.index.v is not None:
                a_b, a_s, a_e, b_b, b_s, b_e = cfg.index.v[1]
                lv = bs[a_b][a_s:a_e].view(cfg.hidden_size, cfg.r_v), bs[b_b][b_s:b_e].view(cfg.r_v, cfg.kv_hidden_size)
            lora_blocks.append((cfg, lq, lk, lv))

        def lora_fn(hidden_state: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, fuse: Optional[torch.Tensor]):
            hidden_state = hidden_state.view(-1, hidden_state.shape[-1])
            q = q.view(-1, q.shape[-1])
            k = k.view(-1, k.shape[-1])
            v = v.view(-1, v.shape[-1])
            if fuse is None:
                for li, s, e in self.grouped:
                    cfg, lq, lk, lv = lora_blocks[li]
                    if lq is not None:
                        q[s:e, :].add_((hidden_state[s:e, :]@lq[0])@lq[1])
                    if lk is not None:
                        k[s:e, :].add_((hidden_state[s:e, :]@lk[0])@lk[1])
                    if lv is not None:
                        v[s:e, :].add_((hidden_state[s:e, :]@lv[0])@lv[1])
                return None, None, None
            assert len(hidden_state.shape) == 2
            with torch.autograd.profiler.record_function("FuseLora"):
                use_half2 = True
                for li, s, e in self.grouped:
                    cfg, lq, lk, lv = lora_blocks[li]
                    _h = hidden_state[s:e, :]
                    x = _h.data_ptr()
                    seq_len = math.prod(_h.shape[:-1])
                    hidden_size = _h.shape[-1]
                    kv_size = cfg.kv_hidden_size
                    q_a, q_b, q_rank = 0, 0, cfg.r_q
                    k_a, k_b, k_rank = 0, 0, cfg.r_k
                    v_a, v_b, v_rank = 0, 0, cfg.r_v
                    lora_temp = fuse.data_ptr()
                    temp_q_output = q[s:e, :].data_ptr()
                    temp_k_output = k[s:e, :].data_ptr()
                    temp_v_output = v[s:e, :].data_ptr()
                    if lq is not None:
                        q_a, q_b = lq[0].data_ptr(), lq[1].data_ptr()
                    if lk is not None:
                        k_a, k_b = lk[0].data_ptr(), lk[1].data_ptr()
                    if lv is not None:
                        v_a, v_b = lv[0].data_ptr(), lv[1].data_ptr()

                    lora(
                        use_half2,
                        x,
                        seq_len, hidden_size, kv_size,
                        q_a, q_b, q_rank,
                        k_a, k_b, k_rank,
                        v_a, v_b, v_rank,
                        lora_temp,
                        temp_q_output, temp_k_output, temp_v_output
                    )
            return None, None, None
        return lora_fn

    def _squeeze_loras(self, seq_loras: List[Optional[Tuple[LoraRenderConfig, List[int]]]], seq_lens: List[int]):
        squeeze_loras: List[Tuple[LoraRenderConfig, List[int]]] = []
        new_index: Dict[Tuple[int, ...], int] = {}
        old2new_mapping: Dict[int, int] = {}
        for (old_i, lora) in enumerate(seq_loras):
            if lora is None:
                continue
            lora_blocks_tuple = tuple(lora[1])
            new_i = new_index.get(lora_blocks_tuple)
            if new_i is None:
                squeeze_loras.append(lora)
                new_i = len(new_index)
                new_index[lora_blocks_tuple] = new_i
            old2new_mapping[old_i] = new_i
        indexs: List[Tuple[int, int, int]] = []
        for seq_i, seq_len in enumerate(seq_lens):
            start = sum(seq_lens[:seq_i])
            end = start+seq_len
            if seq_i in old2new_mapping:
                lora_i = old2new_mapping[seq_i]
                if len(indexs) > 0 and indexs[-1][0] == lora_i and indexs[-1][2] == start:
                    indexs[-1] = (lora_i, indexs[-1][1], end)
                else:
                    indexs.append((lora_i, start, end))
        return indexs, squeeze_loras
