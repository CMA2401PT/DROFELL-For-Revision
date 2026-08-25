import collections
from typing import Dict, List, Optional, Tuple
import torch
from dealers.common.base_types import DeviceTensorCarrier, ModelDescribe
from dealers.common.provider import InferTaskReorderAndDropCtrl
from operators import flags
from operators.core.tokenizer.common import SequenceLengthData
from ..compute_node.infer_meta import InferTailInit, HeadInit, InferBindingInit, InferTask, InferInit, PutBlocks


class TailData:
    def __init__(self, tail_init: InferTailInit) -> None:
        self._seq_core = tail_init.seq_core.clone()

    def update_by_output(self, task: InferTask):
        assert task.output is not None
        if len(self._seq_core.output_token_ids) == task.step:
            self._seq_core.append_token_id(task.output[0])
            return True
        return False  # drop dup result

    @property
    def step(self):
        return len(self._seq_core.output_token_ids)

    @property
    def output_token_ids(self):
        return self._seq_core.output_token_ids.copy()


class InferStateData:
    def __init__(self, init_config: InferInit) -> None:
        self._init_config = init_config
        self._seq_id = init_config.seq_id
        self._prompt_len = init_config.prompt_len

        self._ctrl = InferTaskReorderAndDropCtrl(0)
        self._tail_data = TailData(init_config.tail) if init_config.tail else None

        self._prompt: Optional[InferTask] = None
        self._generatings:  List[Tuple[int, InferTask]] = []
        self._pendings: Dict[int, InferTask] = {}

        # {block_i:{dst_name:[layers Tuple[k:torch.Tensor,v:torch.Tensor]]}}
        self.pending_blocks: Dict[int, Dict[str, List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]]] =\
            collections.defaultdict(dict)

        self.blocks: List[Dict[str, List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]]] = []
        self._splitted_head: bool = False
        self._splitted_tail: bool = False
        self._actual_tail_step: int = 0

    def merged_prompt(self):
        assert self._prompt is not None
        if len(self._generatings) <= 1:
            return
        else:
            _prompt_input = self._prompt.input
            assert not isinstance(_prompt_input, int)
            if isinstance(_prompt_input, list):
                merged_prompt = _prompt_input.copy()
                for _inp in self._generatings[:-1]:
                    assert isinstance(_inp[1].input, int)
                    merged_prompt.append(_inp[1].input)
                self._prompt.input = merged_prompt
                self._prompt.step = self._generatings[-1][0]-1

            else:
                merged_prompt = [_prompt_input.tensor]
                for _inp in self._generatings[:-1]:
                    _generating_input = _inp[1].input
                    assert isinstance(_generating_input, DeviceTensorCarrier)
                    merged_prompt.append(_generating_input.tensor.view(1, -1))
                merged_prompt_t = torch.cat(merged_prompt)
                self._prompt.input = DeviceTensorCarrier(merged_prompt_t, None)
                self._prompt.step = self._generatings[-1][0]-1
            self._generatings = [self._generatings[-1]]

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

    @property
    def decoders_group_seq(self) -> List[str]:
        return [b.dst for b in self._init_config.bindings]

    def get_part(self, seqs: List[Tuple[str, str]]):
        orig_init = self._init_config
        new_init = orig_init.get_part(seqs)
        data = InferStateData(new_init)
        data.blocks = [{} for _ in self.blocks]
        if orig_init.head is not None and orig_init.head.model in seqs:
            data._splitted_head = (len(seqs) == 1) and (len(self.has_decoders) > 0)
        if orig_init.tail is not None and orig_init.tail.model in seqs:
            data._tail_data = self._tail_data
            data._splitted_tail = (len(seqs) == 1) and (len(self.has_decoders) > 0)
            data._actual_tail_step = self._ctrl.step
        for b in orig_init.bindings:
            if b.model not in seqs:
                continue
            for block_i, bs in enumerate(self.blocks):
                data.blocks[block_i][b.dst] = bs[b.dst]
        if self.model_seq[0] in seqs:
            data._prompt = self._prompt
            data._generatings = self._generatings
            data._ctrl = self._ctrl.shallow_copy()
        elif self._block_kv_cache_len == 0:
            data._ctrl = InferTaskReorderAndDropCtrl(0)
        else:
            steps = self._block_kv_cache_len-self.prompt_len+1
            data._ctrl = InferTaskReorderAndDropCtrl(steps)
        data._prompt_len = self._prompt_len
        data._seq_id = self._seq_id
        return data

    @property
    def seq_id(self):
        assert self._seq_id is not None
        return self._seq_id

    @property
    def has_head(self):
        return self._init_config.head is not None

    @property
    def has_tail(self):
        return self._init_config.tail is not None

    @property
    def has_decoders(self):
        return self._init_config.bindings

    @property
    def prompt_len(self) -> int:
        assert self._prompt_len is not None
        return self._prompt_len

    @property
    def is_isolate_head(self):
        return self.has_head and len(self.has_decoders) == 0

    @property
    def is_isolate_tail(self):
        return self.has_tail and len(self.has_decoders) == 0

    @property
    def _has_recover_task_data(self):
        return (self._prompt is not None) or (len(self._generatings) > 0)

    def update_by_output(self, task: InferTask):
        assert self._tail_data is not None
        return self._tail_data.update_by_output(task)

    def _keep_only_last_task(self):
        if len(self._generatings) > 1:
            self._generatings = [self._generatings[-1]]
        elif len(self._generatings) == 1:
            self._prompt = None

    def _update_task(self, task: InferTask):
        assert self._ctrl.can_operate(task.step, task.is_prefill)
        self._ctrl.add_complete_task(task.step, task.is_prefill)
        if self.is_isolate_head or self.is_isolate_tail:
            # keep only last task, as all previously task must be dispatched successfully
            if task.is_prefill:
                self._prompt = task.shallow_copy()
            else:
                self._prompt = None
                self._generatings.append((task.step, task.shallow_copy()))
                self._keep_only_last_task()
        else:
            if task.is_prefill:
                self._prompt = task.shallow_copy()
            else:
                assert len(self._generatings) == 0 or self._generatings[-1][0] == task.step-1
                self._generatings.append((task.step, task.shallow_copy()))
        if self._ctrl.step in self._pendings:
            task = self._pendings[self._ctrl.step]
            del self._pendings[self._ctrl.step]
            self._update_task(task)

    def update_by_task(self, task: InferTask):
        # print(f"update by task: {task}")
        if (task.step < self._ctrl.step) or (task.is_prefill and task.step != 0):
            return True  # reject

        if task.step > self._ctrl.step:
            if task.step in self._pendings:
                return True  # reject
            # handle later
            self._pendings[task.step] = task.shallow_copy()
            return False
        self._update_task(task)
        return False

    def update_by_block(self, blocks: PutBlocks):
        for i, block_i in enumerate(blocks.target_blocks):
            if block_i < len(self.blocks):
                # print(f"drop block {blocks.dst} {block_i} {len(self.blocks)}")
                continue
            # print("accept block")
            block_layers: List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]] = []
            for layer_blocks in blocks.layers_blocks:
                block_layers.append(layer_blocks[i])
            self.pending_blocks[block_i][blocks.user_model] = block_layers
        self._reoreder_block()

    def _reoreder_block(self):
        block_idxs = sorted(list(self.pending_blocks.keys()))
        for i in block_idxs:
            if i > len(self.blocks):
                return
            bs = self.pending_blocks[i]
            if len(bs) == len(self.decoders_group_seq):
                # has at least 1 task postpone so will not drop last task
                assert sorted(bs.keys()) == self.decoders_group_seq
                del self.pending_blocks[i]
                if i == len(self.blocks):
                    self.blocks.append(bs)
                else:
                    self.blocks[i] = bs
            else:
                return
        generating_steps = self._block_kv_cache_len-self.prompt_len+1
        if self._generatings[0][0] == generating_steps:
            return
        offset = generating_steps-self._generatings[0][0]
        assert offset >= 0
        assert offset < len(self._generatings)
        self._generatings = self._generatings[offset:]

    @property
    def _block_kv_cache_len(self) -> int:
        return flags.BLOCK_SIZE*len(self.blocks)

    def gen_init_cfg(self):
        head: Optional[HeadInit] = None
        tail: Optional[InferTailInit] = None
        bindings: List[InferBindingInit] = []

        recover_extra_task: List[InferTask] = []
        if self.has_head:
            assert self._init_config.head is not None
            head = self._init_config.head.shallow_copy()
            if self.is_isolate_head and (not self._splitted_head):
                # for last task replay, as we don't know if this task is dropped with node
                head.init_step = self._ctrl.step-1
                recover_extra_task = [self._prompt.shallow_copy() if self._prompt is not None else self._generatings[-1][1]]
            else:
                assert self._has_recover_task_data
                # align to recover task
                if len(self.blocks) > 0:
                    generating_steps = self._block_kv_cache_len-self.prompt_len+1
                    head.init_step = generating_steps
                    # self._init_config.head.init_step = generating_steps
                else:
                    head.init_step = 0
                # else:
                #     head.init_step = self._ctrl.step
                #     self._init_config.head.init_step = self._ctrl.step
        if self.has_tail:
            assert self._init_config.tail is not None
            tail = self._init_config.tail.shallow_copy()
            assert self._tail_data is not None
            tail.step = self._tail_data.step
            if self.is_isolate_tail and (not self._splitted_tail):
                # for last task replay, as we don't know if this task is dropped with node
                if self._ctrl.step != 0:
                    tail.step = self._ctrl.step-1
                else:
                    tail.step = self._actual_tail_step-1
                assert tail.step != 0
                tail.seq_core.output_token_ids = self._tail_data.output_token_ids[:tail.step]
            else:
                if len(self.blocks) > 0:
                    generating_steps = self._block_kv_cache_len-self.prompt_len+1
                    tail.step = generating_steps
                    tail.seq_core.output_token_ids = self._tail_data.output_token_ids[:tail.step]
                else:
                    tail.step = 0  # prompt+generating
                    outputs_to_keep = self._ctrl.step
                    if outputs_to_keep == 0:
                        outputs_to_keep = self._actual_tail_step
                    if outputs_to_keep != 0:
                        tail.seq_core.output_token_ids = self._tail_data.output_token_ids[:outputs_to_keep-2]
                    else:
                        tail.seq_core.output_token_ids = []

        if len(self._init_config.bindings) > 0:
            for _config in self._init_config.bindings:
                if self._block_kv_cache_len == 0:
                    # print("recover with merged prompt")
                    _recover_config = InferBindingInit(
                        src=_config.src,
                        dst=_config.dst,
                        seq_id=_config.seq_id,
                        init_step=0,
                        length_info=SequenceLengthData(
                            prompt_len=_config.length_info.prompt_len,
                            output_count=0,
                        ),
                        allocate_slots=False,
                        user_model=_config.user_model,
                    )
                else:
                    # print("recover with cache")
                    generating_steps = self._block_kv_cache_len-self.prompt_len+1
                    _init_blocks: List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]] = [
                        b[_config.dst]
                        for b in self.blocks
                    ]
                    # _pending_block_i = len(_init_blocks)
                    # while True:
                    #     if _config.dst not in self.pending_blocks[_pending_block_i]:
                    #         break
                    #     block = self.pending_blocks[_pending_block_i][_config.dst]
                    #     _init_blocks.append(block)

                    s, e = ModelDescribe.from_str(_config.dst).decoders_range
                    _init_blocks_by_layer: List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]] = [[] for _ in range(s, e)]
                    for layers_block in _init_blocks:
                        for layer_i, block in enumerate(layers_block):
                            _init_blocks_by_layer[layer_i].append(block)

                    _recover_config = InferBindingInit(
                        src=_config.src,
                        dst=_config.dst,
                        seq_id=_config.seq_id,
                        init_step=generating_steps,
                        length_info=SequenceLengthData(
                            prompt_len=_config.length_info.prompt_len,
                            output_count=generating_steps,
                        ),
                        allocate_slots=True,
                        user_model=_config.user_model,
                        init_blocks=_init_blocks_by_layer,
                    )
                bindings.append(_recover_config)

        if (self.is_isolate_head or self.is_isolate_tail) and (not self._splitted_head) and (not self._splitted_tail):
            # for last task replay, as we don't know if this task is dropped with node
            recover_extra_task = [self._prompt.shallow_copy() if self._prompt is not None else self._generatings[-1][1].shallow_copy()]
        elif self._block_kv_cache_len == 0:
            # merege into one prompt and one generating
            if self._prompt is not None:
                self.merged_prompt()
                recover_extra_task = [self._prompt.shallow_copy()] + [t.shallow_copy() for (_, t) in self._generatings]
            # wait previous model send prefill and generating
        else:
            recover_extra_task = [t.shallow_copy() for _, t in self._generatings]

        if self.is_isolate_head:
            self._keep_only_last_task()
        self._splitted_head = False
        self._splitted_tail = False
        self._actual_tail_step = 0
        init = InferInit(seq_id=self.seq_id, prompt_len=self.prompt_len, head=head, tail=tail, bindings=bindings)
        return init, recover_extra_task
