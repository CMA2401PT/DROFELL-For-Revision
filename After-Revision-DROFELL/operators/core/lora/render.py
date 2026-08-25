import torch


from dataclasses import dataclass,field
from typing import Callable, List, Literal, Optional, Tuple

from operators import flags

LoraHook = \
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]  # (hidden) -> LoraQ,LoraK,LoraV , shape=[num_tokens,hidden_size]


@dataclass
class LoraBlockIndexer:
    q: Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int, int, int]]]
    k: Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int, int, int]]]
    v: Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int, int, int]]]
    total_len: int
    total_saperate_blocks: int

    def copy(self):
        return LoraBlockIndexer(
            self.q, self.k, self.v, self.total_len, self.total_saperate_blocks
        )

    def single_block_pos(self, name: Literal["q_a", "q_b", "k_a", "k_b", "v_a", "v_b"]) -> Optional[Tuple[int, int]]:
        i = self.__getattribute__(name[0])
        if i is None:
            return None
        is_a = name[2] == "a"
        if is_a:
            return i[0][0], i[0][1]
        else:
            return i[0][2], i[0][3]

    def separate_pos(self, name: Literal["q_a", "q_b", "k_a", "k_b", "v_a", "v_b"]) -> Optional[Tuple[int, int, int]]:  # b,s,e
        i = self.__getattribute__(name[0])
        if i is None:
            return None
        is_a = name[2] == "a"
        if is_a:
            return i[1][0], i[1][1], i[1][2]
        else:
            return i[1][3], i[1][4], i[1][5]

    def single_block_to_saperate_blocks(self, single_block: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        kv_size = single_block.shape[1]
        kv_blocks = (self.total_saperate_blocks+1)//2
        bs = [
            torch.zeros((flags.BLOCK_SIZE, kv_size), dtype=single_block.dtype, device=single_block.device)
            for _ in range(kv_blocks*2)
        ]
        for name in ("q_a", "q_b", "k_a", "k_b", "v_a", "v_b"):
            se = self.single_block_pos(name)
            nbse = self.separate_pos(name)
            if se is not None:
                assert nbse is not None
                s, e = se
                b, ns, ne = nbse
                bs[b][ns:ne] = single_block[s:e]
        return [
            (bs[i*2], bs[i*2+1])
            for i in range(kv_blocks)
        ]


def qkv_blocks_arrangement(block_size: int, r_q: int, r_k: int, r_v: int, query_groups: int) -> LoraBlockIndexer:
    ls = [r_q*query_groups, r_q*query_groups, r_k*query_groups, r_k, r_v*query_groups, r_v]
    os = [sum(ls[:i]) for i in range(len(ls))]
    oe = [sum(ls[:i+1]) for i in range(len(ls))]

    s, e, b = 0, block_size, 0

    def alloc_len(l: int) -> Tuple[int, int, int]:
        nonlocal s, e, b
        assert l <= block_size, f"{l}>{block_size}"
        if l > e-s:
            s, e, b = 0, block_size, b+1
            return alloc_len(l)
        _s, s = s, s+l
        return b, _s, _s+l
    separate_pos = [alloc_len(l) for l in ls]
    return LoraBlockIndexer(
        q=((os[0], oe[0], os[1], oe[1]), (*separate_pos[0], *separate_pos[1])) if r_q != 0 else None,
        k=((os[2], oe[2], os[3], oe[3]), (*separate_pos[2], *separate_pos[3])) if r_k != 0 else None,
        v=((os[4], oe[4], os[5], oe[5]), (*separate_pos[4], *separate_pos[5])) if r_v != 0 else None,
        total_len=sum(ls),
        total_saperate_blocks=b+1,
    )


@dataclass
class LoraRenderConfig:
    r_q: int
    r_k: int
    r_v: int
    num_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    index: LoraBlockIndexer = field(default_factory=lambda:LoraBlockIndexer(None, None, None, 0, 0))

    def copy(self):
        return LoraRenderConfig(
            self.r_q, self.r_k, self.r_v,
            self.num_heads, self.num_key_value_heads, self.head_dim,
            self.index.copy()
        )

    @property
    def hidden_size(self) -> int:
        return self.num_heads*self.head_dim

    @property
    def query_groups(self) -> int:
        return 0 if self.num_key_value_heads == 0 else self.num_heads//self.num_key_value_heads

    @property
    def kv_hidden_size(self) -> int:
        return self.num_key_value_heads*self.head_dim

    def __post_init__(self):
        if self.query_groups != 0:
            self.index = qkv_blocks_arrangement(flags.BLOCK_SIZE, self.r_q, self.r_k, self.r_v, self.query_groups)

    @property
    def lora_block_shape(self) -> Tuple[int, int]:
        return (self.index.total_len, self.kv_hidden_size)

    def get_q(self, lora_block: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.index.q is None:
            return None
        else:
            a_s, a_e, b_s, b_e = self.index.q[0]
            if len(lora_block.shape) == 2:
                q_a = lora_block[a_s: a_e, :].reshape((self.hidden_size, self.r_q))
                q_b = lora_block[b_s: b_e, :].reshape((self.r_q, self.hidden_size))
                return q_a, q_b
            else:
                q_a = lora_block[:, a_s: a_e, :].view((-1, self.hidden_size, self.r_q))
                q_b = lora_block[:, b_s: b_e, :].view((-1, self.r_q, self.hidden_size))
                return q_a, q_b

    def get_k(self, lora_block: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.index.k is None:
            return None
        else:
            a_s, a_e, b_s, b_e = self.index.k[0]
            if len(lora_block.shape) == 2:
                k_a = lora_block[a_s: a_e, :].reshape((self.hidden_size, self.r_k))
                k_b = lora_block[b_s: b_e, :].reshape((self.r_k, self.kv_hidden_size))
                return k_a, k_b
            else:
                k_a = lora_block[:, a_s: a_e, :].view((-1, self.hidden_size, self.r_k))
                k_b = lora_block[:, b_s: b_e, :].view((-1, self.r_k, self.kv_hidden_size))
                return k_a, k_b

    def get_v(self, lora_block: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.index.v is None:
            return None
        else:
            a_s, a_e, b_s, b_e = self.index.v[0]
            if len(lora_block.shape) == 2:
                v_a = lora_block[a_s: a_e, :].reshape((self.hidden_size, self.r_v))
                v_b = lora_block[b_s: b_e, :].reshape((self.r_v, self.kv_hidden_size))
                return v_a, v_b
            else:
                v_a = lora_block[:, a_s: a_e, :].view((-1, self.hidden_size, self.r_v))
                v_b = lora_block[:, b_s: b_e, :].view((-1, self.r_v, self.kv_hidden_size))
                return v_a, v_b

    def render_lora_hook(self, lora_block: torch.Tensor) -> LoraHook:
        assert lora_block.shape[0] >= self.lora_block_shape[0]
        assert lora_block.shape[1] == self.lora_block_shape[1]
        lq, lk, lv = self.get_q(lora_block), self.get_k(lora_block), self.get_v(lora_block)

        def lora_fn(hidden_state: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, fuse: Optional[torch.Tensor]):
            assert fuse is None
            assert len(hidden_state.shape) == 2
            assert hidden_state.size(1) == self.hidden_size
            if lq is not None:
                q = q+((hidden_state@lq[0])@lq[1])
            if lk is not None:
                k = k+((hidden_state@lk[0])@lk[1])
            if lv is not None:
                v = v+((hidden_state@lv[0])@lv[1])
            return q, k, v
        return lora_fn
