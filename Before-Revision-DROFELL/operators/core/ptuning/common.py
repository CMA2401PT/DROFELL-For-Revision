from dataclasses import dataclass


@dataclass
class PtuningConfig:
    num_layers: int = 0
    head_dim: int = 0
    num_key_value_heads: int = 0
    prefix_len: int = 128
    dropout_p: float = 0.1
    build_prompt_str: str = ""

    def clone(self):
        return PtuningConfig(
            num_layers=self.num_layers,
            head_dim=self.head_dim,
            num_key_value_heads=self.num_key_value_heads,
            prefix_len=self.prefix_len,
            dropout_p=self.dropout_p,
            build_prompt_str=self.build_prompt_str,
        )

    def build_prompt(self, prompt: str) -> str:
        return self.build_prompt_str.format(prompt)

    @property
    def key_value_output_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def combined_output_shape(self) -> int:
        return 2*self.num_layers * self.key_value_output_size

    @property
    def layer_kv_cache_elements(self) -> int:
        return self.prefix_len*self.key_value_output_size*2

    @property
    def kv_cache_elements(self) -> int:
        return self.num_layers*self.layer_kv_cache_elements

    @property
    def per_layer_elements(self) -> int:
        return self.prefix_len*self.key_value_output_size*2

    def check_or_set_decoders_config(
        self,
        num_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        num_query_heads: int = 0,
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
        return self
