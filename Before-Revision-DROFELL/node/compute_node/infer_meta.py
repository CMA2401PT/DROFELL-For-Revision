from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

from dealers.common.base_types import DeviceTensorCarrier, ModelDescribe
from operators.core.lora.render import LoraRenderConfig
from operators.core.tokenizer.common import SequenceCore, SequenceLengthData
from operators.core.sampler.common import SamplingParams
UserModelDecoderIndex = int


@dataclass
class HeadInit:
    src: str
    dst: str
    init_step: int

    @property
    def model(self):
        return (self.src, self.dst)

    def shallow_copy(self):
        return HeadInit(
            src=self.src, dst=self.dst, init_step=self.init_step,
        )


@dataclass
class InferTailInit:
    src: str
    dst: str
    seq_id: str
    step: int
    seq_core: SequenceCore
    sampling_params: SamplingParams

    @property
    def model(self):
        return (self.src, self.dst)

    def shallow_copy(self):
        return InferTailInit(
            src=self.src, dst=self.dst, seq_id=self.seq_id,
            step=self.step, seq_core=self.seq_core.clone(),
            sampling_params=self.sampling_params,
        )


@dataclass
class InferBindingInit:
    src: str
    dst: str
    seq_id: str
    init_step: int
    length_info: SequenceLengthData
    allocate_slots: bool
    user_model: Optional[Tuple[
        str,
        List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]],
        Optional[LoraRenderConfig]
    ]]

    # [num_layers*[num_blocks*(k:torch.Tenser,v:torch.Tensor)]]
    init_blocks: Optional[List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]]] = None

    @property
    def received_blocks(self):
        if self.init_blocks is None:
            return 0
        return len(self.init_blocks[0])

    @property
    def model(self):
        return (self.src, self.dst)

    def shallow_copy(self):
        return InferBindingInit(
            src=self.src, dst=self.dst, seq_id=self.seq_id,
            init_step=self.init_step, length_info=self.length_info.clone(),
            allocate_slots=self.allocate_slots,
            user_model=self.user_model,
            init_blocks=self.init_blocks
        )


@dataclass
class Navigate:
    user_models: List[str]
    _current: int = 0
    _cache: Optional[Dict[int, Tuple[str, bool, bool]]] = None

    @property
    def current(self):
        return self._current

    def current_user_model(self, extend_tail: bool):
        if self._current == len(self.user_models):
            assert extend_tail
            return self.user_models[-1]
        else:
            return self.user_models[self._current]

    def target(self, offset: int = 0) -> Tuple[str, bool, bool]:
        ptr = self._current+offset
        if self._cache is not None:
            out = self._cache.get(ptr)
            if out is not None:
                return out
        else:
            self._cache = {}

        target = self.user_models[ptr]
        desc = ModelDescribe.from_str(target)
        out = self.user_models[ptr], desc.is_head, desc.is_tail
        self._cache[ptr] = out
        return out

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_cache']
        return state

    @property
    def is_last(self) -> bool:
        return self._current == len(self.user_models)-1

    @property
    def finished(self) -> bool:
        return self._current == len(self.user_models)

    def next(self):
        self._current += 1
        self._cache = None

    def index(self, user_model: str):
        return self.user_models.index(user_model)

    def shallow_copy(self):
        return Navigate(
            user_models=self.user_models,
            _current=self._current
        )


@dataclass
class InferTask:
    seq_id: str
    step: int
    navi: Navigate
    is_prefill: bool
    input: Union[DeviceTensorCarrier, List[int], int]
    output: Optional[Tuple[int, Dict[int, float]]]

    def __str__(self):
        return f"InferTask({self.step},{self.navi.current_user_model(True)})(prefill={self.is_prefill},finished={self.navi.finished})"

    @property
    def is_cpu(self):
        if isinstance(self.input, DeviceTensorCarrier):
            return self.input.tensor.is_cpu
        return True

    @property
    def unique_id(self):
        return (self.seq_id, self.step)

    @property
    def unique_id_str(self):
        return f"{self.seq_id}-{self.step}"

    def shallow_copy(self):
        shallow_copy_input = self.input
        if isinstance(shallow_copy_input, List):
            shallow_copy_input = shallow_copy_input.copy()
        elif isinstance(shallow_copy_input, DeviceTensorCarrier):
            shallow_copy_input = shallow_copy_input.shallow_copy()
        return InferTask(
            seq_id=self.seq_id,
            step=self.step,
            navi=self.navi.shallow_copy(),
            is_prefill=self.is_prefill,
            input=shallow_copy_input,
            output=self.output
        )


@dataclass
class BatchRecoverInferTask:
    seq_id: str
    navigate: List[str]
    input: List[Tuple[int, Union[int, DeviceTensorCarrier]]]


UserModelDataProvider = Callable[[Tuple[int, int]], Tuple[
    List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]],
    Optional[LoraRenderConfig]
]]


@dataclass
class InferInit:
    seq_id: str
    prompt_len: int
    head: Optional[HeadInit]
    tail: Optional[InferTailInit]
    bindings: List[InferBindingInit]

    def __str__(self) -> str:
        lst = ", ".join(f"{s}->{u}" for s, u in self.models)
        return f"InferInit({lst})"

    # def get_seq(self):
    #     seq: Union[None, SequenceCore, SequenceLengthData] = None
    #     if self.tail is not None:
    #         seq = self.tail.seq_core.clone()
    #     for b in self.bindings:
    #         assert b.length_info.output_count == b.init_step
    #         if seq is None:
    #             seq = b.length_info
    #         assert seq.output_count == b.init_step, f"{seq.output_count} != {b.init_step}"
    #     if self.head is not None:
    #         if seq is None:
    #             seq = SequenceLengthData(0, self.head.init_step)
    #         else:
    #             assert seq.output_count == self.head.init_step
    #     assert seq is not None
    #     return seq

    def get_part(self, seqs: List[Tuple[str, str]]):
        init = InferInit(self.seq_id, self.prompt_len, head=None, tail=None, bindings=[])
        if self.head is not None and self.head.model in seqs:
            init.head = self.head.shallow_copy()
        if self.tail is not None and self.tail.model in seqs:
            init.tail = self.tail.shallow_copy()
        for b in self.bindings:
            if b.model not in seqs:
                continue
            init.bindings.append(b.shallow_copy())
        return init

    @property
    def user_models(self) -> List[str]:
        return [u for b, u in self.models]

    @property
    def base_models(self) -> List[str]:
        return [b for b, u in self.models]

    @property
    def models(self) -> List[Tuple[str, str]]:
        models: List[Tuple[str, str]] = []
        if self.head is not None:
            models.append((self.head.src, self.head.dst))
        for b in self.bindings:
            models.append((b.src, b.dst))
        if self.tail is not None:
            models.append((self.tail.src, self.tail.dst))
        return models

    @property
    def num_models(self):
        return len(self.bindings)+(1 if self.head is not None else 0)+(1 if self.tail is not None else 0)

    @classmethod
    def gen(
        cls,
        seq_id: str,
        models: List[Tuple[str, str]],
        user_model_name_and_slicer: Optional[Tuple[str, UserModelDataProvider]],
        seq_data: Union[SequenceLengthData, SequenceCore],
        sampling_params: Optional[SamplingParams]
    ):
        head: Optional[HeadInit] = None
        tail: Optional[InferTailInit] = None
        bindings: List[InferBindingInit] = []
        for src, dst in models:
            desc = ModelDescribe.from_str(src)
            if desc.is_head:
                head = HeadInit(src, dst, init_step=0)
            elif desc.is_tail:
                assert isinstance(seq_data, SequenceCore)
                assert sampling_params is not None
                tail = InferTailInit(
                    src=src,
                    dst=dst,
                    seq_id=seq_id,
                    step=0,
                    seq_core=seq_data.clone(),
                    sampling_params=sampling_params,
                )
            else:
                user_model = None
                if user_model_name_and_slicer is not None:
                    user_model_name, slicer = user_model_name_and_slicer
                    user_model_full_name = UserModelDataName(user_model_name).full_name(desc)
                    blocks, cfg = slicer(desc.decoders_range)
                    user_model = (user_model_full_name, blocks, cfg)
                if isinstance(seq_data, SequenceCore):
                    length_data = seq_data.length_data
                else:
                    length_data = SequenceLengthData(seq_data.prompt_len, seq_data.output_count)
                binding = InferBindingInit(
                    src=src,
                    dst=dst,
                    seq_id=seq_id,
                    init_step=0,
                    length_info=length_data,
                    allocate_slots=False,
                    user_model=user_model,
                )
                bindings.append(binding)
        init_config = cls(seq_id=seq_id, prompt_len=seq_data.prompt_len, head=head, tail=tail, bindings=bindings)
        return init_config


@dataclass
class PutBlocks:
    user_model: str
    target_blocks: List[int]
    layers_blocks: List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]]

    def __str__(self) -> str:
        return f"infer_blocks(model={self.user_model},blocks={self.target_blocks})"

    def get_blocks(self, targets: List[int]):
        indexs = [self.target_blocks.index(t) for t in targets]
        return [
            [blocks[i] for i in indexs]
            for blocks in self.layers_blocks
        ]

    # def read_blocks(self, start_layer: int):
    #     for _layer_i, blocks in enumerate(self.layers_blocks):
    #         layer_i = start_layer+_layer_i
    #         for b_i, block in zip(self.target_blocks, blocks):
    #             yield b_i, layer_i, block


class UserModelDataName:
    def __init__(self, name: str) -> None:
        assert not "@" in name
        assert not ":" in name
        self.name = name  # e.g. my-model

    @staticmethod
    def is_full_res_name(res_name: str):
        return res_name.startswith("user-model-data-")

    @staticmethod
    def split_full_name(res_name: str):
        """
        res_name= user-model-data-my-model@llama-2-7B:000-008
        """
        assert UserModelDataName.is_full_res_name(res_name)
        desc = ModelDescribe.from_str(res_name)
        _user_model_name, _base = desc.base_model.split("@")
        s, e = desc.decoders_range
        base = ModelDescribe.new(_base, start=s, end=e)
        user_model_name = _user_model_name.removeprefix("user-model-data-")
        return user_model_name, base

    def full_name(self, desc: ModelDescribe):
        s, e = desc.decoders_range
        user_model_handle = ModelDescribe.new(f"user-model-data-{self.name}@{desc.base_model}", start=s, end=e).__str__()
        return user_model_handle
