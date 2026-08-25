import extend_asyncio as asyncio
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union
from dealers.common.base_types import DeviceTensorCarrier, ModelDescribe
from dealers.common.provider import Provider

from operators.core.lora.render import LoraRenderConfig
from operators.core.tokenizer.common import SequenceCore
from operators.core.sampler.common import SamplingParams
from node.compute_node import InferTask, InferNode, InferInit, InferHeadInit, InferBindingInit, InferTailInit, PutBlocks, InferNavigate, UserModelDataName
from node.shadow_node_state import InferStateData


@dataclass
class InferTestConfig:
    block_retrieve: bool = False
    recover: Union[Tuple[int, bool], List[Tuple[int, bool]]] = (0, False)

    def should_trigger_recover(self, step_i: int):
        if isinstance(self.recover, list):
            if (step_i, True) in self.recover:
                return True, True
            elif (step_i, False) in self.recover:
                return True, False
            return False, False
        else:
            return self.recover[0] > 0 and (step_i == self.recover[0]), self.recover[1]

    @property
    def could_recover(self):
        if isinstance(self.recover, list):
            return len(self.recover) > 0
        else:
            return self.recover[0] > 0


class UserModelDataProvider:
    def __init__(self) -> None:
        self.user_models: Dict[str, Tuple[
            List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]],
            Optional[LoraRenderConfig]
        ]] = {}

    def put_model(self, name: str, blocks: List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]], cfg: Optional[LoraRenderConfig]):
        self.user_models[name] = (blocks, cfg)

    def get_model(self, name: str, range: Tuple[int, int]) -> \
        Tuple[List[List[Tuple[DeviceTensorCarrier, DeviceTensorCarrier]]],
              Optional[LoraRenderConfig]]:
        blocks, cfg = self.user_models[name]
        s, e = range
        return blocks[s:e], cfg


class Breaker:
    def __init__(
        self,
        underlay_put_resources: Callable[[str, PutBlocks], Coroutine[Any, Any, None]],
        underlay_on_infer_task_done: Callable[[InferTask], Coroutine[Any, Any, None]],
    ) -> None:
        """
        temporal protect
        1. avoid data reach after node is dropped
        2. guard init data is sent earlier than task
        """
        self._breaked = False
        self._underlay_put_resources = underlay_put_resources
        self._underlay_on_infer_task_done = underlay_on_infer_task_done
        self._node: InferNode
        self._init_waitor: asyncio.Future[None] = asyncio.Future()

    def break_inbound(self):
        self._breaked = True
        self._node.dropped = True

    async def _put_resources(self, name: str, res: PutBlocks):
        if self._breaked:
            return
        else:
            await self._underlay_put_resources(name, res)

    async def _on_infer_task_done(self, infer_task: InferTask, cb: Callable[[], None]):
        cb()
        if self._breaked:
            return
        else:
            await self._underlay_on_infer_task_done(infer_task)

    async def _on_infer_task_discard(self, infer_task: InferTask, cb: Callable[[], None]):
        cb()

    async def on_infer_task(self, infer_task: InferTask):
        assert not self._breaked
        await self._init_waitor
        context_id = infer_task.unique_id_str

        def add_barrier():
            self._node.provider.brain.add_barrier(context_id)

        def remove_barrier():
            self._node.provider.brain.remove_barrier(context_id)
        add_barrier()
        updated_task = await self._node.infer(
            infer_task=infer_task,
            add_barrier=add_barrier,
            remove_barrier=remove_barrier
        )
        if updated_task is None:
            await self._on_infer_task_discard(infer_task, remove_barrier)
        else:
            await self._on_infer_task_done(updated_task, remove_barrier)

    async def infer_init(self, config: InferInit, block_retrieve: bool, provider: Provider):
        assert not self._breaked
        self._node = InferNode(
            block_retrieve=block_retrieve,
            provider=provider,
            put_resources=self._put_resources,
            # infer_task_dispatch=self._on_infer_task_done,
            # infer_task_discard=self._on_infer_task_discard
        )

        await self._node.infer_init(config)
        self._init_waitor.set_result(None)

    def remove(self):
        self._breaked = True
        self._node.remove()

    def __del__(self):
        self.remove()


class InferInstance:
    seq_counter: int = 0

    def __init__(self,
                 providers: List[Provider],
                 base_models_lookup: Dict[str, int],
                 user_model_provider: UserModelDataProvider,
                 groups: List[List[Tuple[str, str]]],
                 user_model_name: Optional[str],
                 seq_core: SequenceCore,
                 sampling_params: SamplingParams,
                 test_config: InferTestConfig,
                 ) -> None:

        self.providers = providers
        self.base_models_lookup = base_models_lookup
        self.groups = groups
        self.navi_list: List[str] = [
            dst
            for group in groups
            for (_, dst) in group
        ]
        # self.user_model = user_model
        self.user_model_provider = user_model_provider
        self.user_model_name: Optional[str] = user_model_name

        self.test_config: InferTestConfig = test_config
        self.sampling_params = sampling_params

        self.seq_core = seq_core.clone()
        self.generating_i: int = 0
        self.seq_id = f"seq-{InferInstance.seq_counter}"
        InferInstance.seq_counter += 1
        self.nodes_and_data: List[Tuple[Breaker, InferStateData]] = []
        self._model_look_up: Dict[str, Tuple[Breaker, InferStateData]] = {}
        # self.model_init_waitor: Dict[str, Set[asyncio.Future[None]]] = {}
        self.result_waitor: Dict[Tuple[str, int], asyncio.Future[InferTask]] = {}

    # async def _wait_and_get(self, name: str):
    #     ws = self.model_init_waitor.get(name)
    #     if ws is not None:
    #         fut: asyncio.Future[None] = asyncio.Future()
    #         ws.add(fut)
    #         await fut
    #     return self._model_look_up[name]

    # def _set_node_ready(self, name: str, pair: Tuple[Breaker, InferStateData]):
    #     self._model_look_up[name] = pair
    #     for w in self.model_init_waitor[name]:
    #         w.set_result(None)
    #     del self.model_init_waitor[name]

    # def _set_node_unready(self, name: str):
    #     del self._model_look_up[name]
    #     self.model_init_waitor[name] = set()

    async def _update_blocks(self, name: str, res: PutBlocks):
        name = name.removesuffix("-blocks")
        _, data = self._model_look_up[name]
        data.update_by_block(res)

    async def _on_resource(self, name: str, res: PutBlocks):
        await self._update_blocks(name, res)

    def _create_remote_node(self):
        return Breaker(
            self._on_resource,
            self.on_infer_task,
        )

    async def ainit(self):
        for group in self.groups:
            head: Optional[InferHeadInit] = None
            tail: Optional[InferTailInit] = None
            bindings: List[InferBindingInit] = []

            provider_i = self.base_models_lookup[group[0][0]]

            assert all(
                self.base_models_lookup[src] == provider_i
                for src, _ in group
            ), f"invalid split up {self.base_models_lookup} {[src for src, _ in group]}"

            for src, dst in group:
                desc = ModelDescribe.from_str(src)
                if desc.is_head:
                    head = InferHeadInit(src, dst, init_step=0)
                elif desc.is_tail:
                    tail = InferTailInit(
                        src=src,
                        dst=dst,
                        seq_id=self.seq_id,
                        step=0,
                        seq_core=self.seq_core.clone(),
                        sampling_params=self.sampling_params,
                    )
                else:
                    user_model = None
                    if self.user_model_name is not None and self.user_model_name != "":
                        user_model_full_name = UserModelDataName(self.user_model_name).full_name(desc)
                        blocks, cfg = self.user_model_provider.get_model(self.user_model_name, desc.decoders_range)
                        user_model = (user_model_full_name, blocks, cfg)
                    binding = InferBindingInit(
                        src=src,
                        dst=dst,
                        seq_id=self.seq_id,
                        init_step=0,
                        length_info=self.seq_core.length_data,
                        allocate_slots=False,
                        user_model=user_model,
                    )
                    bindings.append(binding)
            init_config = InferInit(seq_id=self.seq_id, prompt_len=self.seq_core.length_data.prompt_len, head=head, tail=tail, bindings=bindings)
            node = self._create_remote_node()
            await node.infer_init(init_config, self.test_config.block_retrieve, self.providers[provider_i])
            data = InferStateData(init_config=init_config)
            pair = (node, data)
            self.nodes_and_data.append(pair)
            for src, dst in group:
                self._model_look_up[dst] = pair

    async def infer(self):
        stopped, next_token, logprobs = await self.prefill()
        yield ("prefilling", next_token, logprobs)
        count = 0
        while not stopped:
            stopped, next_token, logprobs = await self.generating_step()
            yield ("generating", next_token, logprobs)
            count += 1

    async def prefill(self):
        return await self._infer_step(self.seq_core.prompt_token_ids.copy())

    async def generating_step(self):
        self.generating_i += 1
        ret = await self._infer_step(
            inp=self.seq_core.output_token_ids[-1],
        )
        return ret

    async def _infer_step(self, inp: Union[List[int], int]):
        infer_task = InferTask(
            seq_id=self.seq_id,
            step=self.generating_i,
            navi=InferNavigate(self.navi_list),
            is_prefill=isinstance(inp, list),
            input=inp,
            output=None
        )

        should_recover, recover_with_split = self.test_config.should_trigger_recover(self.generating_i)
        if should_recover:
            self.recover(recover_with_split)

        fut: asyncio.Future[InferTask] = asyncio.Future()
        self.result_waitor[infer_task.unique_id] = fut
        self._dispatch_infer_task(infer_task)
        infer_task = await fut
        # while not infer_task.navi.finished:
        #     _model, _, _ = infer_task.navi.target(0)
        #     _node = self.model_look_up[_model]
        #     _data = self.infer_state_datas[self.stateless_nodes.index(_node)]
        #     should_drop = _data.update_by_task(infer_task)
        #     assert not should_drop
        #     infer_task = await _node.infer(infer_task)
        assert infer_task.output is not None
        next_token, log_probs = infer_task.output
        self.seq_core.append_token_id(next_token)
        stopped = self.seq_core.stopped
        return (stopped, next_token, log_probs)

    async def on_infer_task(self, infer_task: InferTask):
        self._dispatch_infer_task(infer_task)

    def _dispatch_infer_task(self, infer_task: InferTask):
        async def on_dispatch():
            if infer_task.navi.finished:
                _t = infer_task.navi.user_models[-1]
                _, _data = self._model_look_up[_t]
                if _data.update_by_output(infer_task):
                    fut = self.result_waitor.get(infer_task.unique_id)
                    assert fut is not None
                    del self.result_waitor[infer_task.unique_id]
                    fut.set_result(infer_task)
                return
            _model, _, _ = infer_task.navi.target(0)
            _node, _data = self._model_look_up[_model]
            should_drop = _data.update_by_task(infer_task)
            if should_drop:
                self.on_drop(infer_task)
            else:
                await _node.on_infer_task(infer_task)
        asyncio.create_task(on_dispatch())

    def on_drop(self, task: InferTask):
        assert task.is_cpu
        print(f"task: {task} drop at shadow node!")

    def recover(self, split: bool = False):
        print(f"recover @ {self.generating_i} split={split}")
        pairs: List[Tuple[Breaker, InferStateData]] = []
        # dispatch_later: List[Tuple[Breaker, List[infer_node.InferTask]]] = []

        for orig_node, data in self.nodes_and_data:
            orig_node.break_inbound()
            for _m, (_n, _) in self._model_look_up.copy().items():
                if _n == orig_node:
                    del self._model_look_up[_m]
            if not split:
                cfg, recover_extra = data.gen_init_cfg()
                provider_i = self.base_models_lookup[cfg.base_models[0]]
                assert all(
                    self.base_models_lookup[src] == provider_i
                    for src in cfg.base_models
                ), "invalid split up"
                _node = self._create_remote_node()
                asyncio.create_task(_node.infer_init(
                    config=cfg,
                    block_retrieve=self.test_config.block_retrieve,
                    provider=self.providers[provider_i]
                ))
                pair = (_node, data)
                for _m in cfg.user_models:
                    self._model_look_up[_m] = pair
                for t in recover_extra:
                    asyncio.create_task(_node.on_infer_task(t))

                pairs.append(pair)
            else:
                seq = data.model_seq
                for i, m in enumerate(seq):
                    sub_data = data.get_part([m])
                    cfg, recover_extra = sub_data.gen_init_cfg()
                    provider_i = self.base_models_lookup[cfg.base_models[0]]
                    assert all(
                        self.base_models_lookup[src] == provider_i
                        for src in cfg.base_models
                    ), "invalid split up"
                    _node = self._create_remote_node()
                    asyncio.create_task(_node.infer_init(
                        config=cfg,
                        block_retrieve=self.test_config.block_retrieve,
                        provider=self.providers[provider_i]
                    ))
                    pair = (_node, sub_data)
                    for _m in cfg.user_models:
                        self._model_look_up[_m] = pair
                    for t in recover_extra:
                        asyncio.create_task(_node.on_infer_task(t))
                    pairs.append(pair)
            orig_node.remove()
        self.nodes_and_data = pairs

    def remove(self):
        for node, _ in self.nodes_and_data:
            node.remove()
        self.nodes_and_data = None  # type:ignore

    def __del__(self):
        assert self.nodes_and_data is None, "must call remove before free"
