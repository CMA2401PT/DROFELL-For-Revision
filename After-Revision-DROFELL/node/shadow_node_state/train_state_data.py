import hashlib
import os
import time
from typing import Dict, List, Optional, Tuple

import torch

from ..compute_node.train_meta import ModelCKPT, TrainInit, TrainTask, HeadInit, TrainTailInit, TrainBindingInit
from utils.size_metrics import retained_bytes
from utils.runtime_metrics import record_event


def _state_fingerprint(value: object) -> str:
    """Return a stable, exact digest for nested checkpoint state."""

    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(
                item, key=lambda key: (type(key).__qualname__, repr(key))
            ):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__qualname__.encode("ascii"))
            digest.update(b"\0")
            for child in item:
                update(child)
        elif item is None or isinstance(item, (bool, int, float, str)):
            digest.update(type(item).__qualname__.encode("ascii"))
            digest.update(b"\0")
            digest.update(repr(item).encode("utf-8"))
            digest.update(b"\0")
        else:
            raise TypeError(
                "unsupported checkpoint fingerprint value: "
                f"{type(item).__qualname__}"
            )

    update(value)
    return digest.hexdigest()


class TrainStateData:
    def __init__(self, init_config: TrainInit) -> None:
        self._init_config = init_config
        # direction: forward and backward
        self._inputs: Dict[int, Dict[Tuple[str, str], TrainTask]] = {}
        self._ckpts: Dict[int, Dict[str, ModelCKPT]] = {}
        self._ready_ckpts: Dict[str, ModelCKPT] = {}
        self._ready_ckpt_version: int = 0
        self._update_cpu_ns: int = 0
        self._task_input_bytes: int = 0
        self._checkpoint_bytes_received: int = 0

    @property
    def ready_ckpts(self) -> Dict[str, ModelCKPT]:
        return self._ready_ckpts

    def metrics_snapshot(self) -> Dict[str, int]:
        input_bytes = retained_bytes(self._inputs)
        pending_checkpoint_bytes = retained_bytes(self._ckpts)
        ready_checkpoint_bytes = retained_bytes(self._ready_ckpts)
        payload_bytes = input_bytes + pending_checkpoint_bytes + ready_checkpoint_bytes
        return {
            "metadata_bytes": max(0, retained_bytes(self) - payload_bytes),
            "pending_task_bytes": input_bytes,
            "pending_checkpoint_bytes": pending_checkpoint_bytes,
            "ready_checkpoint_bytes": ready_checkpoint_bytes,
            "total_retained_bytes": retained_bytes(self),
            "update_cpu_ns": self._update_cpu_ns,
            "task_input_bytes_observed": self._task_input_bytes,
            "checkpoint_bytes_received": self._checkpoint_bytes_received,
            "ready_checkpoint_version": self._ready_ckpt_version,
            "pending_version_count": len(self._inputs),
        }

    @property
    def model_seq(self) -> List[Tuple[str, str]]:  # src->dst
        seqs: List[Tuple[str, str]] = []
        init_config = self._init_config
        if init_config.head is not None:
            seqs.append(init_config.head.model)
        for b in init_config.bindings:
            seqs.append(b.model)
        if init_config.tail is not None:
            seqs.append(init_config.tail.model)
        return seqs

    def get_part(self, seqs: List[Tuple[str, str]]):
        orig_init = self._init_config
        new_init = orig_init.get_part(seqs)
        data = TrainStateData(new_init)
        target_models = [dst for _, dst in seqs]
        data._ready_ckpts = {
            model: ckpt for model, ckpt in self._ready_ckpts.items()
            if model in target_models
        }
        data._ready_ckpt_version = self._ready_ckpt_version
        if self.model_seq[0] in seqs:
            for v, inputs in self._inputs.items():
                for m, t in inputs.items():
                    if t.navi.is_backward:
                        continue
                    data._inputs.setdefault(v, {})[m] = t
        if self.model_seq[-1] in seqs:
            for v, inputs in self._inputs.items():
                for m, t in inputs.items():
                    if not t.navi.is_backward:
                        continue
                    data._inputs.setdefault(v, {})[m] = t
        return data

    @property
    def models(self) -> List[str]:
        seqs: List[str] = []
        init_config = self._init_config
        if init_config.head is not None:
            seqs.append(init_config.head.dst)
        for b in init_config.bindings:
            seqs.append(b.dst)
        if init_config.tail is not None:
            seqs.append(init_config.tail.dst)
        return seqs

    @property
    def decoders_group_seq(self) -> List[str]:
        return [b.dst for b in self._init_config.bindings]

    @property
    def has_head(self) -> bool:
        return self._init_config.head is not None

    @property
    def has_tail(self) -> bool:
        return self._init_config.tail is not None

    @property
    def decoders_count(self) -> int:
        return len(self.decoders_group_seq)

    @property
    def has_decoders(self) -> bool:
        return self.decoders_count > 0

    @property
    def organize(self) -> Tuple[str, bool]:  # bool=special-isolate
        if self.has_head and self.has_tail:
            return "J", False  # -> U -> ii / -> iI -> ii
        if not self.has_head and self.has_tail:
            return "U", (not self.has_decoders)
        if self.has_head and not self.has_tail:
            return "iI", (not self.has_decoders)
        else:
            return "ii", False

    def update_by_ckpt(self, ckpt: ModelCKPT):
        started = time.perf_counter_ns()
        self._checkpoint_bytes_received += retained_bytes(ckpt)
        if os.getenv("PEFT_RECOVER_STATE_FINGERPRINTS", "0") == "1":
            record_event(
                "train_checkpoint_state_fingerprint",
                user_model=ckpt.user_model,
                checkpoint_version=ckpt.version,
                state_fingerprint=_state_fingerprint(ckpt.state),
                component_fingerprints={
                    key: _state_fingerprint(value)
                    for key, value in sorted(ckpt.state.items())
                },
            )
        if ckpt.version < self._ready_ckpt_version:
            self._update_cpu_ns += time.perf_counter_ns() - started
            return
        if ckpt.version not in self._ckpts:
            self._ckpts[ckpt.version] = {}
        self._ckpts[ckpt.version][ckpt.user_model] = ckpt
        self._check_ckpts()
        self._update_cpu_ns += time.perf_counter_ns() - started

    def _check_ckpts(self):
        versions = sorted(self._ckpts.keys())
        for version in versions:
            ckpts = self._ckpts[version]
            if len(ckpts) != len(self.models):
                continue
            # print(f"stable node version update\n\t{set(self.models)}->{version}")
            assert set(ckpts.keys()) == set(self.models)
            self._ready_ckpts = ckpts
            self._ckpts = {k: v for k, v in self._ckpts.items() if k > version}
            self._ready_ckpt_version = version
            input_versions = sorted(self._inputs.keys())
            inputs_to_del: List[int] = []
            for v in input_versions:
                if v < version:
                    inputs_to_del.append(v)
            for v in inputs_to_del:
                del self._inputs[v]

    def update_by_task(self, task: TrainTask):
        started = time.perf_counter_ns()
        self._task_input_bytes += retained_bytes(task)
        # print(f"update data of {self.models} by\n\t{task}")
        req_version = task.request_version
        if req_version < self._ready_ckpt_version:
            # print(f"\treject  because: {req_version} < {self._ready_ckpt_version}")
            self._update_cpu_ns += time.perf_counter_ns() - started
            return True
        if req_version not in self._inputs:
            self._inputs[req_version] = {}
        minibatch_id = task.minibatch_id
        current_model = task.navi.current_dir_user_model
        if (minibatch_id, current_model) in self._inputs[req_version]:
            # print(f"\treject because: aready exist in {list(self._inputs[req_version].keys())}")
            self._update_cpu_ns += time.perf_counter_ns() - started
            return True
        else:
            # print(f"\taccept")
            self._inputs[req_version][(
                minibatch_id, current_model)] = task.shallow_copy()
            self._update_cpu_ns += time.perf_counter_ns() - started
            return False

    def gen_init_cfg(self):
        head: Optional[HeadInit] = None
        tail: Optional[TrainTailInit] = None
        bindings: List[TrainBindingInit] = []
        recover_extra_task: List[TrainTask] = []

        if self.has_head:
            assert self._init_config.head is not None
            head = self._init_config.head.shallow_copy()
            head.version = self._ready_ckpt_version
        if self.has_tail:
            assert self._init_config.tail is not None
            tail = self._init_config.tail.shallow_copy()
            tail.version = self._ready_ckpt_version
        if self.decoders_count > 0:
            for _config in self._init_config.bindings:
                _cfg = _config.shallow_copy()
                if _cfg.dst in self._ready_ckpts:
                    _ckpt = self._ready_ckpts[_cfg.dst]
                    _cfg.version = _ckpt.version
                    _cfg.recover_state_dict = _ckpt.state
                    _cfg.data_init_method = None
                bindings.append(_cfg)
        for _, tasks in self._inputs.items():
            for _, task in tasks.items():
                recover_extra_task.append(task.shallow_copy())
        return TrainInit(head=head, tail=tail, bindings=bindings), recover_extra_task
