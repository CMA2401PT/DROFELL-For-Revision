import collections
import time
from typing import Dict, List, Optional, Tuple
import torch
from dealers.common.base_types import DeviceTensorCarrier, ModelDescribe
from dealers.common.provider import InferTaskReorderAndDropCtrl
from operators import flags
from operators.core.tokenizer.common import SequenceLengthData
from ..compute_node.infer_meta import InferTailInit, HeadInit, InferBindingInit, InferTask, InferInit, PutBlocks
from utils.size_metrics import retained_bytes


class TailData:
    def __init__(self, tail_init: InferTailInit) -> None:
        self._seq_core = tail_init.seq_core.clone()

    def update_by_output(self, task: InferTask):
        assert task.output is not None
        if len(self._seq_core.output_token_ids) == task.step:
            self._seq_core.append_token_id(task.output[0])
            return True
        return False  # drop dup result

    def update_by_recovery_output(
        self,
        step: int,
        output_token_ids: List[int],
    ) -> None:
        end = step + len(output_token_ids)
        existing = self._seq_core.output_token_ids
        overlap_end = min(len(existing), end)
        if overlap_end > step:
            if existing[step:overlap_end] != \
                    output_token_ids[:overlap_end - step]:
                raise AssertionError(
                    "recovery output history conflicts with tail shadow")
        if len(existing) < step:
            raise AssertionError(
                "recovery output history has a gap before tail shadow")
        for token_id in output_token_ids[max(0, len(existing) - step):]:
            self._seq_core.append_token_id(token_id)

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
        self._update_cpu_ns: int = 0
        self._task_input_bytes: int = 0
        self._received_kv_bytes: int = 0

    def metrics_snapshot(self) -> Dict[str, int]:
        prompt_bytes = retained_bytes(self._prompt)
        generating_bytes = retained_bytes(self._generatings)
        pending_task_bytes = retained_bytes(self._pendings)
        kv_bytes = retained_bytes(self.blocks) + retained_bytes(self.pending_blocks)
        payload_bytes = prompt_bytes + generating_bytes + pending_task_bytes + kv_bytes
        return {
            "metadata_bytes": max(0, retained_bytes(self) - payload_bytes),
            "prompt_bytes": prompt_bytes,
            "generating_bytes": generating_bytes,
            "pending_task_bytes": pending_task_bytes,
            "kv_bytes": kv_bytes,
            "total_retained_bytes": retained_bytes(self),
            "update_cpu_ns": self._update_cpu_ns,
            "task_input_bytes_observed": self._task_input_bytes,
            "kv_bytes_received": self._received_kv_bytes,
            "replay_task_count": len(self._generatings) + (1 if self._prompt is not None else 0),
            "checkpoint_block_count": len(self.blocks),
        }

    def recovery_snapshot(self) -> Dict[str, object]:
        return {
            "ctrl_step": self._ctrl.step,
            "tail_step": (
                self._tail_data.step if self._tail_data is not None else None
            ),
            "confirmed_output_count": len(self.confirmed_output_token_ids()),
            "prompt_step": (
                self._prompt.step if self._prompt is not None else None
            ),
            "generating_steps": [step for step, _ in self._generatings],
            "pending_steps": sorted(self._pendings),
            "checkpoint_block_count": len(self.blocks),
            "checkpoint_token_count": self._block_kv_cache_len,
            "has_head": self.has_head,
            "has_tail": self.has_tail,
        }

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
        started = time.perf_counter_ns()
        assert self._tail_data is not None
        result = self._tail_data.update_by_output(task)
        self._update_cpu_ns += time.perf_counter_ns() - started
        return result

    def update_by_recovery_output(self, task: InferTask) -> None:
        if not task.is_recovery_replay:
            raise ValueError("expected a recovery replay task")
        if self._tail_data is None:
            raise ValueError("recovery output update requires tail state")
        output_token_ids = task.recovery_output_token_ids
        if output_token_ids is None:
            raise ValueError("recovery chunk has no confirmed outputs")
        self._tail_data.update_by_recovery_output(
            task.step, output_token_ids)

    def confirmed_output_token_ids(self) -> List[int]:
        """Return the longest output prefix proven by tail results or inputs.

        In a redundant graph, a Backbone's inner tail result can lose the race
        to another outer path.  The next decode input nevertheless proves the
        preceding greedy output token.  Use those inputs to advance the local
        confirmation prefix without requiring the outer Agent to acknowledge
        which physical copy won.
        """

        confirmed = (
            self._tail_data.output_token_ids
            if self._tail_data is not None
            else []
        )
        for step, task in self._generatings:
            if step <= 0:
                continue
            token_id = task.decode_input_token_id
            if token_id is None and isinstance(task.input, int):
                token_id = task.input
            if token_id is None:
                continue
            output_index = step - 1
            if output_index < len(confirmed):
                if confirmed[output_index] != token_id:
                    raise AssertionError(
                        "decode input conflicts with confirmed output shadow: "
                        f"step={step}, output_index={output_index}, "
                        f"confirmed={confirmed[output_index]}, input={token_id}, "
                        f"confirmed_len={len(confirmed)}, "
                        f"generating_steps={[s for s, _ in self._generatings]}"
                    )
            elif output_index == len(confirmed):
                confirmed.append(token_id)
            else:
                break
        return confirmed

    def _keep_only_last_task(self):
        if len(self._generatings) > 1:
            self._generatings = [self._generatings[-1]]
        elif len(self._generatings) == 1:
            self._prompt = None

    def _update_task(self, task: InferTask):
        if task.is_recovery_decode:
            if task.is_prefill:
                raise ValueError("recovery decode chunk cannot be prefill")
            count = task.recovery_decode_count
            if isinstance(task.input, list):
                if len(task.input) != count:
                    raise ValueError(
                        "recovery token count does not match task count")
                inputs = task.input
            elif isinstance(task.input, DeviceTensorCarrier):
                tensor = task.input.tensor
                if tensor.shape[0] != count:
                    raise ValueError(
                        "recovery carrier rows do not match task count")
                inputs = [
                    DeviceTensorCarrier(tensor[i].view(-1), task.input.no_gc)
                    for i in range(count)
                ]
            else:
                raise TypeError("recovery decode chunk input cannot be scalar")
            input_token_ids = task.recovery_input_token_ids
            if input_token_ids is None or len(input_token_ids) != count:
                raise ValueError(
                    "recovery decode chunk requires every original token ID")
            for offset, inp in enumerate(inputs):
                expanded = task.shallow_copy()
                expanded.step = task.step + offset
                expanded.input = inp
                expanded.recovery_decode_count = 1
                expanded.recovery_output_token_ids = None
                expanded.decode_input_token_id = input_token_ids[offset]
                expanded.recovery_input_token_ids = None
                self._update_task(expanded)
            return
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
        started = time.perf_counter_ns()
        self._task_input_bytes += retained_bytes(task.input)
        # print(f"update by task: {task}")
        if (task.step < self._ctrl.step) or (task.is_prefill and task.step != 0):
            self._update_cpu_ns += time.perf_counter_ns() - started
            return True  # reject

        if task.step > self._ctrl.step:
            if task.step in self._pendings:
                self._update_cpu_ns += time.perf_counter_ns() - started
                return True  # reject
            # handle later
            self._pendings[task.step] = task.shallow_copy()
            self._update_cpu_ns += time.perf_counter_ns() - started
            return False
        self._update_task(task)
        self._update_cpu_ns += time.perf_counter_ns() - started
        return False

    def update_by_block(self, blocks: PutBlocks):
        started = time.perf_counter_ns()
        self._received_kv_bytes += retained_bytes(blocks.layers_blocks)
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
        self._update_cpu_ns += time.perf_counter_ns() - started

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
        # A replacement node can publish its first reconstructed KV block
        # before any post-recovery decode task has reached the new shadow.
        # There is then no replay window to trim yet; the block itself is
        # already retained above.
        if not self._generatings:
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

    def coalesce_recovery_tasks(
        self,
        recover_tasks: List[InferTask],
        confirmed_output_token_ids: Optional[List[int]] = None,
    ) -> List[InferTask]:
        """Combine the confirmed prefix of KV-backed decode replay tasks.

        Online recovery may include one in-flight task whose sampled output is
        not known yet.  That suffix must stay sequential.  A state without a
        local tail can still prove task ``i``'s output from task ``i+1``'s
        original decode input token.
        """
        if len(self.blocks) == 0 or len(recover_tasks) <= 1:
            return recover_tasks
        if any(task.is_prefill for task in recover_tasks):
            raise AssertionError(
                "KV checkpoint recovery cannot contain prefill tasks")

        first_step = recover_tasks[0].step
        expected_steps = list(
            range(first_step, first_step + len(recover_tasks)))
        actual_steps = [task.step for task in recover_tasks]
        if actual_steps != expected_steps:
            raise AssertionError(
                "recovery decode tasks must be contiguous before coalescing")

        input_token_ids: List[int] = []
        for task in recover_tasks:
            token_id = task.decode_input_token_id
            if token_id is None and isinstance(task.input, int):
                token_id = task.input
            if token_id is None:
                # State created before original-token metadata was available.
                # Sequential replay is slower but remains correct.
                return recover_tasks
            input_token_ids.append(token_id)

        output_history = confirmed_output_token_ids
        if self._tail_data is not None:
            local_history = self._tail_data.output_token_ids
            if output_history is None or len(local_history) > len(output_history):
                output_history = local_history

        if output_history is not None:
            confirmed_count = min(
                len(recover_tasks),
                max(0, len(output_history) - first_step),
            )
            output_token_ids = output_history[
                first_step:first_step + confirmed_count]
        else:
            # The next decode input is the previous task's confirmed output.
            confirmed_count = len(recover_tasks) - 1
            output_token_ids = input_token_ids[1:confirmed_count + 1]

        if confirmed_count <= 1:
            return recover_tasks
        if len(output_token_ids) != confirmed_count:
            raise AssertionError(
                "confirmed output history does not cover recovery chunk")

        chunk_sources = recover_tasks[:confirmed_count]
        first = chunk_sources[0].shallow_copy()
        inputs = [task.input for task in chunk_sources]
        if all(isinstance(inp, int) for inp in inputs):
            first.input = [int(inp) for inp in inputs]
        elif all(isinstance(inp, DeviceTensorCarrier) for inp in inputs):
            tensors: List[torch.Tensor] = []
            for inp in inputs:
                assert isinstance(inp, DeviceTensorCarrier)
                tensor = inp.tensor
                if isinstance(tensor, tuple):
                    tensor = tensor[1]()
                tensors.append(tensor.view(-1))
            first.input = DeviceTensorCarrier(torch.stack(tensors), None)
        else:
            # A single shadow state should be homogeneous, but retain the
            # correctness-preserving fallback if it is not.
            return recover_tasks

        first.recovery_decode_count = confirmed_count
        first.recovery_input_token_ids = \
            input_token_ids[:confirmed_count]
        first.recovery_output_token_ids = output_token_ids
        return [first] + recover_tasks[confirmed_count:]

    def gen_init_cfg(
        self,
        confirmed_output_token_ids: Optional[List[int]] = None,
    ):
        head: Optional[HeadInit] = None
        tail: Optional[InferTailInit] = None
        bindings: List[InferBindingInit] = []

        recover_extra_task: List[InferTask] = []
        effective_confirmed_outputs = self.confirmed_output_token_ids()
        if (
            confirmed_output_token_ids is not None
            and len(confirmed_output_token_ids) > len(effective_confirmed_outputs)
        ):
            effective_confirmed_outputs = confirmed_output_token_ids.copy()
        # A downstream decode input proves the preceding sampled token even
        # when this shadow has not observed that output yet (notably when a
        # parallel outer path won the race).  Persist that proven prefix in
        # the source state before handing it to the replacement handle.  The
        # replacement will merge replay results into this same state object;
        # leaving the local tail behind would create an artificial history gap
        # and can make an inner Backbone recovery fail after graph install.
        if self._tail_data is not None:
            self._tail_data.update_by_recovery_output(
                0,
                effective_confirmed_outputs,
            )
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
                tail.seq_core.output_token_ids = effective_confirmed_outputs[:tail.step]
            else:
                if len(self.blocks) > 0:
                    generating_steps = self._block_kv_cache_len-self.prompt_len+1
                    tail.step = generating_steps
                    tail.seq_core.output_token_ids = effective_confirmed_outputs[:tail.step]
                else:
                    tail.step = 0  # prompt+generating
                    outputs_to_keep = self._ctrl.step
                    if outputs_to_keep == 0:
                        outputs_to_keep = self._actual_tail_step
                    if outputs_to_keep != 0:
                        tail.seq_core.output_token_ids = effective_confirmed_outputs[:outputs_to_keep-2]
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
                # Full re-prefill reconstructs decoder KV from the complete
                # known prefix.  Its last output was already confirmed before
                # failure, so the replacement tail must restore that token
                # instead of sampling it again.  The following generating
                # task remains the independently dispatched live frontier.
                if self.has_tail:
                    assert self._tail_data is not None
                    prompt_step = self._prompt.step
                    confirmed_outputs = effective_confirmed_outputs
                    if prompt_step >= len(confirmed_outputs):
                        raise AssertionError(
                            "full re-prefill output is not confirmed by tail "
                            "shadow")
                    self._prompt.recovery_output_token_ids = [
                        confirmed_outputs[prompt_step]
                    ]
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
