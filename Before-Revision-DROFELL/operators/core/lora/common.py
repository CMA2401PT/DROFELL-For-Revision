from dataclasses import dataclass

from operators import flags

from .render import qkv_blocks_arrangement

from . import LoraRenderConfig


@dataclass
class LoraConfig(LoraRenderConfig):
    num_layers: int = 0
    build_prompt_str: str = ""
    dropout_p: float = 0.0
    len_limit: int = 0

    @property
    def render_config(self):
        return LoraRenderConfig(
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            r_q=self.r_q,
            r_k=self.r_k,
            r_v=self.r_v,
            index=self.index.copy(),
        )

    def clone(self):
        return LoraConfig(
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            r_q=self.r_q,
            r_k=self.r_k,
            r_v=self.r_v,
            num_layers=self.num_layers,
            build_prompt_str=self.build_prompt_str,
            index=self.index.copy(),
            dropout_p=self.dropout_p,
            len_limit=self.len_limit,
        )

    def build_prompt(self, prompt: str) -> str:
        return self.build_prompt_str.format(prompt)

    @property
    def per_layer_elements(self) -> int:
        return self.index.total_len*self.kv_hidden_size

    @property
    def elements(self) -> int:
        return self.per_layer_elements*self.num_layers

    def check_or_set_decoders_config(
        self,
        num_layers: int,
        num_key_value_heads: int,
        num_query_heads: int,
        head_dim: int,
    ):
        if self.num_layers != 0:
            assert self.num_layers == num_layers
        else:
            self.num_layers = num_layers
        if self.num_key_value_heads != 0:
            assert self.num_key_value_heads == num_key_value_heads
        else:
            self.num_key_value_heads = num_key_value_heads
        if self.head_dim != 0:
            assert self.head_dim == head_dim
        else:
            self.head_dim = head_dim

        if self.num_heads != 0:
            assert self.num_heads == num_query_heads
        else:
            self.num_heads = num_query_heads
        if self.len_limit > 0:
            self.auto_set_r(self.len_limit)
        self.index = qkv_blocks_arrangement(flags.BLOCK_SIZE, self.r_q, self.r_k, self.r_v, self.query_groups)
        return self

    def auto_set_r(self, len_limit: int):
        factor = 0
        if self.r_q != 0:
            factor += self.query_groups*2
        if self.r_k != 0:
            factor += self.query_groups+1
        if self.r_v != 0:
            factor += self.query_groups+1
        assert len_limit % factor == 0, f"len_limit must be r*{factor}, current: {len_limit}"
        r = len_limit//factor
        print(f"auto set r to {r}")
        if self.r_q != 0:
            if self.r_q == -1:
                self.r_q = r
            else:
                assert self.r_q == r
        if self.r_k != 0:
            if self.r_k == -1:
                self.r_k = r
            else:
                assert self.r_k == r
        if self.r_v != 0:
            if self.r_v == -1:
                self.r_v = r
            else:
                assert self.r_v == r
