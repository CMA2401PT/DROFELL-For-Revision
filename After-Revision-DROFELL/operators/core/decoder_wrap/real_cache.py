from typing import Optional, Tuple
import torch
from .common import FuseConfig


class FuseCache(object):
    def __init__(self, device: torch.device, config: FuseConfig) -> None:
        self.cfg = config
        self._temp_rms_output: Optional[torch.Tensor] = None
        self._temp_lora_output: Optional[torch.Tensor] = None
        self._temp_gate_output: Optional[torch.Tensor] = None
        self._temp_up_output: Optional[torch.Tensor] = None
        self._temp_q_output: Optional[torch.Tensor] = None
        self._temp_k_output: Optional[torch.Tensor] = None
        self._temp_v_output: Optional[torch.Tensor] = None
        self.batch_size: int = 0
        self.seq_len: int = 0
        self.device: torch.device = device

    @property
    def temp_rms_output(self) -> torch.Tensor:
        assert self._temp_rms_output is not None
        return self._temp_rms_output

    @property
    def temp_lora_output(self) -> torch.Tensor:
        assert self._temp_lora_output is not None
        return self._temp_lora_output

    @property
    def temp_gate_output(self) -> torch.Tensor:
        assert self._temp_gate_output is not None
        return self._temp_gate_output

    @property
    def temp_up_output(self) -> torch.Tensor:
        assert self._temp_up_output is not None
        return self._temp_up_output

    @property
    def temp_q_output(self) -> torch.Tensor:
        assert self._temp_q_output is not None
        return self._temp_q_output

    @property
    def temp_k_output(self) -> torch.Tensor:
        assert self._temp_k_output is not None
        return self._temp_k_output

    @property
    def temp_v_output(self) -> torch.Tensor:
        assert self._temp_v_output is not None
        return self._temp_v_output

    def make_fuse_forward_temp(self, batch_size: int, seq_len: int):
        assert seq_len == 1
        if (
            self.batch_size == batch_size
            and self.seq_len == seq_len
            and self._temp_rms_output is not None
        ):
            return
        x_height = batch_size * seq_len
        self._temp_rms_output = torch.empty(
            (x_height, self.cfg.hidden_size), dtype=torch.float16, requires_grad=False, device=self.device)
        self._temp_lora_output = torch.empty(
            (x_height, self.cfg.hidden_size), dtype=torch.float16, requires_grad=False, device=self.device)
        self._temp_gate_output = torch.empty(
            (x_height, self.cfg.intermediate_size), dtype=torch.float16, requires_grad=False, device=self.device)
        self._temp_up_output = torch.empty(
            (x_height, self.cfg.intermediate_size), dtype=torch.float16, requires_grad=False, device=self.device)
        self._temp_q_output = torch.empty(
            (x_height, self.cfg.hidden_size), dtype=torch.float16, requires_grad=False, device=self.device)
        self._temp_k_output = torch.empty(
            (x_height, self.cfg.head_dim * self.cfg.num_key_value_heads), dtype=torch.float16, requires_grad=False, device=self.device)
        self._temp_v_output = torch.empty(
            (x_height, self.cfg.head_dim * self.cfg.num_key_value_heads), dtype=torch.float16, requires_grad=False, device=self.device)
        self.batch_size: int = batch_size
        self.seq_len: int = seq_len

    def get_qkv_with_std_view(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states = self.temp_q_output.view(
            self.batch_size, self.cfg.num_query_heads, self.cfg.head_dim)
        key_states = self.temp_k_output.view(
            self.batch_size, self.cfg.num_key_value_heads, self.cfg.head_dim)
        value_states = self.temp_v_output.view(
            self.batch_size, self.cfg.num_key_value_heads, self.cfg.head_dim)
        return query_states, key_states, value_states

    def remove_fuse_forward_temp(self):
        # Decode reuses these buffers on every layer and every token. Keeping
        # them resident removes allocator traffic and gives CUDA graphs stable
        # addresses. A shape change is handled by make_fuse_forward_temp.
        pass
