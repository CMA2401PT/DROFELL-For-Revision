from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from dataset.common import ProcessedRecord
from dealers.common.base_types import DeviceTensorCarrier, ModelDescribe
from dealers.common.base_types import DeviceTensorCarrier
from dealers.common.brain import OptimizerConfig, SchedulerConfig, TrainForwardMeta, UserModelDataInitConfig
from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig
# from . import flags as mod_flags


UserModelDecoderIndex = int


@dataclass
class HeadInit:
    src: str
    dst: str
    version: int

    @property
    def model(self):
        return (self.src, self.dst)

    def shallow_copy(self):
        return HeadInit(
            src=self.src,
            dst=self.dst,
            version=self.version,
        )


@dataclass
class TrainTailInit:
    src: str
    dst: str
    impl: str
    version: int

    @property
    def model(self):
        return (self.src, self.dst)

    def shallow_copy(self):
        return TrainTailInit(
            src=self.src,
            dst=self.dst,
            impl=self.impl,
            version=self.version
        )


@dataclass
class TrainBindingInit:
    src: str
    dst: str
    version: int
    user_model_config: Union[PtuningConfig, LoraConfig]
    optimizer_config: OptimizerConfig
    scheduler_config: Optional[SchedulerConfig]
    data_init_method: Optional[UserModelDataInitConfig]
    recover_state_dict: Optional[Dict[str, Any]] = None

    @property
    def model(self):
        return (self.src, self.dst)

    def shallow_copy(self):
        return TrainBindingInit(
            src=self.src,
            dst=self.dst,
            version=self.version,
            user_model_config=self.user_model_config,
            optimizer_config=self.optimizer_config,
            scheduler_config=self.scheduler_config,
            data_init_method=self.data_init_method,
            recover_state_dict=self.recover_state_dict,
        )


@dataclass
class Navigate:
    dir_user_model: List[str]
    current: int

    @property
    def current_dir_user_model(self):
        return self.dir_user_model[self.current]

    def current_dir_user_model_with_extend(self):
        if self.current == len(self.dir_user_model):
            return self.dir_user_model[-1]
        else:
            return self.dir_user_model[self.current]

    @property
    def finished(self):
        return self.current == len(self.dir_user_model)

    @property
    def is_last(self):
        return self.current == len(self.dir_user_model)-1

    def target(self, offset: int = 0) -> Tuple[str, bool, bool, bool, bool]:  # dst, is_forward_binding, is_backward_binding,is_head,is_tail
        target = self.dir_user_model[self.current+offset]
        is_forward_binding, is_backward_binding, is_head, is_tail = False, False, False, False
        if target.endswith(",forward"):
            is_forward_binding = True
            target = target.removesuffix(",forward")
        elif target.endswith(",backward"):
            is_backward_binding = True
            target = target.removesuffix(",backward")
        else:
            desc = ModelDescribe.from_str(target)
            is_head, is_tail = desc.is_head, desc.is_tail
        return target, is_forward_binding, is_backward_binding, is_head, is_tail

    def target_dir_user_model(self, offset: int = 0):
        return self.dir_user_model[self.current+offset]

    @property
    def is_backward(self):
        target = self.dir_user_model[self.current]
        return target.endswith(",backward")

    @property
    def is_first_or_last_forward_binding(self) -> Tuple[bool, bool]:
        """
        indicate alloc/free aux device memory (e.g. pos idx)
        """
        _, is_forward_binding, _, _, _ = self.target()
        if not is_forward_binding:
            return False, False
        _is_first = (self.current == 0) or self.target(-1)[-2]
        _, _next_is_forward, _, _, _ = self.target(1)
        _is_last = (self.is_last) or (not _next_is_forward)
        return _is_first, _is_last

    @property
    def is_last_backward_binding(self) -> bool:
        _, _, is_backward_binding, _, _ = self.target()
        if not is_backward_binding:
            return False
        return self.is_last

    def next(self):
        self.current += 1


@dataclass
class TrainHeadInput:
    is_bfloat16: bool
    input_lens: List[int]
    position_ids: List[int]
    input_ids: List[int]
    label_ids: List[int]

    @property
    def input_ids_tensor(self):
        return torch.tensor(self.input_ids, dtype=torch.int32, device="cpu", requires_grad=False)

    @property
    def is_cpu(self):
        return True

    @classmethod
    def from_records(cls, samples: List[ProcessedRecord], is_bfloat16: bool):
        input_lens = [len(sample.input_ids) for sample in samples]
        return cls(
            is_bfloat16=is_bfloat16,
            input_lens=input_lens,
            position_ids=[i for input_len in input_lens for i in range(input_len)],
            input_ids=[e for sample in samples for e in sample.input_ids],
            label_ids=[e for sample in samples for e in sample.output_ids],
        )

    def shallow_copy(self):
        return self

    def debug_check_cpu(self):
        return


@dataclass
class TrainTailInput:
    carrier: DeviceTensorCarrier
    label_ids: List[int]

    @property
    def label_ids_tensor(self):
        return torch.tensor(self.label_ids, dtype=torch.long, device="cpu", requires_grad=False)

    @property
    def is_cpu(self):
        return self.carrier.is_cpu

    def shallow_copy(self):
        assert (self.carrier.tensor.is_cpu)
        return TrainTailInput(
            carrier=self.carrier.shallow_copy(),
            label_ids=self.label_ids
        )

    def debug_check_cpu(self):
        assert self.carrier.is_cpu


PossibleTrainInput = Union[TrainHeadInput, TrainForwardMeta, TrainTailInput, DeviceTensorCarrier]


@dataclass
class TrainTask:
    request_version: int
    minibatch_id: str
    batch_ids: List[str]
    navi: Navigate
    input: Optional[PossibleTrainInput]
    loss: Optional[float] = None

    @property
    def unique_id_str(self):
        return self.minibatch_id

    def shallow_copy(self):
        return TrainTask(
            request_version=self.request_version,
            minibatch_id=self.minibatch_id,
            batch_ids=self.batch_ids,
            navi=Navigate(
                dir_user_model=self.navi.dir_user_model,
                current=self.navi.current,
            ),
            input=self.input.shallow_copy() if self.input is not None else None,
            loss=self.loss
        )

    def debug_check_cpu(self):
        if self.input is None:
            return
        self.input.debug_check_cpu()

    @property
    def is_cpu(self):
        if self.input is None:
            return True
        return self.input.is_cpu

    def __str__(self):
        name = "end" if self.navi.current == len(self.navi.dir_user_model) else self.navi.dir_user_model[self.navi.current]
        return f"TrainTask({self.minibatch_id},{name})(@{self.batch_ids})"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class TrainInit:
    head: Optional[HeadInit]
    tail: Optional[TrainTailInit]
    bindings: List[TrainBindingInit]

    def __str__(self) -> str:
        lst = ", ".join(f"{s}->{u}" for s, u in self.models)
        return f"InferInit({lst})"

    def get_part(self, seqs: List[Tuple[str, str]]):
        init = TrainInit(head=None, tail=None, bindings=[])
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
    def dir_user_models(self):
        user_models = self.user_models
        dir_user_models: List[str] = []
        for m in user_models:
            desc = ModelDescribe.from_str(m)
            if desc.is_decoders:
                dir_user_models.append(m+",forward")
            else:
                dir_user_models.append(m)
        for m in user_models[::-1]:
            desc = ModelDescribe.from_str(m)
            if desc.is_decoders:
                dir_user_models.append(m+",backward")
        return dir_user_models

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
    def model_versions(self) -> List[Tuple[str, int]]:
        models: List[Tuple[str, int]] = []
        if self.head is not None:
            models.append((self.head.dst, self.head.version))
        for b in self.bindings:
            models.append((b.dst, b.version))
        if self.tail is not None:
            models.append((self.tail.dst, self.tail.version))
        return models

    @property
    def arch(self):
        has_head, has_tail = self.head is not None, self.tail is not None
        has_bindings = len(self.bindings)
        if has_head and has_tail:
            return "J"
        if has_head:
            if has_bindings:
                return "Ii"
            else:
                return 'ih'
        if has_tail and has_bindings:
            return "U"
        if has_tail:
            return "it"
        return "ii"

    def num_steps(self, is_backward: bool):
        if is_backward:
            assert self.tail is None
            return len(self.bindings)
        h = 1 if self.head is not None else 0
        if self.tail is None:
            return len(self.bindings)+h
        return 2*len(self.bindings)+h+1

    def first_user_model(self):
        has_head, has_tail = self.head is not None, self.tail is not None
        has_bindings = len(self.bindings)
        if has_head and has_tail:
            return self.head.dst, None  # type:ignore
        if has_head:
            if has_bindings:
                return self.head.dst, self.bindings[-1].dst  # type:ignore
            else:
                return self.head.dst, None  # type:ignore
        if has_tail and has_bindings:
            return self.bindings[0].dst, None  # type:ignore
        if has_tail:
            return self.tail.dst, None  # type:ignore
        return self.bindings[0].dst, self.bindings[-1].dst  # type:ignore

    @classmethod
    def gen(
        cls,
        models: List[Tuple[str, str]],
        user_model_config: Union[PtuningConfig, LoraConfig],
        optimizer_config: OptimizerConfig,
        scheduler_config: Optional[SchedulerConfig],
        user_model_data_provider: Callable[[str, str], UserModelDataInitConfig],
    ):

        head: Optional[HeadInit] = None
        tail: Optional[TrainTailInit] = None
        bindings: List[TrainBindingInit] = []

        for src, dst in models:
            desc = ModelDescribe.from_str(src)
            if desc.is_head:
                head = HeadInit(src, dst, version=0)
            elif desc.is_tail:
                tail = TrainTailInit(
                    src=src,
                    dst=dst,
                    impl="",
                    version=0
                )
            else:
                init_data = user_model_data_provider(src, dst)
                binding = TrainBindingInit(
                    src=src,
                    dst=dst,
                    version=0,
                    user_model_config=user_model_config,
                    optimizer_config=optimizer_config,
                    scheduler_config=scheduler_config,
                    data_init_method=init_data,
                )
                bindings.append(binding)
        train_config = TrainInit(head, tail, bindings)
        return train_config


@dataclass
class ModelCKPT:
    user_model: str
    version: int  # version after update, = req_version of input
    state: Dict[str, Any]

    def __str__(self) -> str:
        return f"model_ckpt(model={self.user_model},version={self.version})"
