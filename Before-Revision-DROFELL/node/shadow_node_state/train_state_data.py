from typing import Dict, List, Optional, Tuple
from ..compute_node.train_meta import ModelCKPT, TrainInit, TrainTask, HeadInit, TrainTailInit, TrainBindingInit


class TrainStateData:
    def __init__(self, init_config: TrainInit) -> None:
        self._init_config = init_config
        # direction: forward and backward
        self._inputs: Dict[int, Dict[Tuple[str, str], TrainTask]] = {}
        self._ckpts: Dict[int, Dict[str, ModelCKPT]] = {}
        self._ready_ckpts: Dict[str, ModelCKPT] = {}
        self._ready_ckpt_version: int = 0

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
        if ckpt.version < self._ready_ckpt_version:
            return
        if ckpt.version not in self._ckpts:
            self._ckpts[ckpt.version] = {}
        self._ckpts[ckpt.version][ckpt.user_model] = ckpt
        self._check_ckpts()

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
        # print(f"update data of {self.models} by\n\t{task}")
        req_version = task.request_version
        if req_version < self._ready_ckpt_version:
            # print(f"\treject  because: {req_version} < {self._ready_ckpt_version}")
            return True
        if req_version not in self._inputs:
            self._inputs[req_version] = {}
        minibatch_id = task.minibatch_id
        current_model = task.navi.current_dir_user_model
        if (minibatch_id, current_model) in self._inputs[req_version]:
            # print(f"\treject because: aready exist in {list(self._inputs[req_version].keys())}")
            return True
        else:
            # print(f"\taccept")
            self._inputs[req_version][(minibatch_id, current_model)] = task.shallow_copy()
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
