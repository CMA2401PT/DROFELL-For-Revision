from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union
from dealers.common.base_types import ModelDescribe
from dealers.common.brain import OptimizerConfig, SchedulerConfig, UserModelDataInitConfig
from dealers.common.provider import Provider
from node.compute_node import TrainTask, TrainNode, TrainHeadInit, TrainBindingInit, TrainTailInit, TrainInit, ModelCKPT, TrainNavigate, TrainHeadInput
from node.shadow_node_state import TrainStateData
from operators.core.lora.common import LoraConfig
from operators.core.ptuning.common import PtuningConfig
from dealers.real_basic import flags as flags
import extend_asyncio as asyncio


@dataclass
class TrainTestConfig:
    backup_period: int = 0
    recover: Union[Tuple[int, bool], List[Tuple[int, bool]]] = (0, False)

    @property
    def could_recover(self):
        if isinstance(self.recover, list):
            return len(self.recover) > 0
        else:
            return self.recover[0] > 0

    def __post_init__(self):
        if self.backup_period == 0:
            assert not self.could_recover
        # if self.could_recover:
        #     flags.SIMULATE_DROP = True

    def should_trigger_recover(self, step_i: int):
        if isinstance(self.recover, list):
            if (step_i, True) in self.recover:
                return True, True
            elif (step_i, False) in self.recover:
                return True, False
            return False, False
        else:
            return self.recover[0] > 0 and (step_i == self.recover[0]), self.recover[1]


class Breaker:
    def __init__(
        self,
        underlay_put_resources: Callable[[str, ModelCKPT], Coroutine[Any, Any, None]],
        underlay_on_infer_task_done: Callable[[TrainTask], Coroutine[Any, Any, None]],
    ) -> None:
        """
        temporal protect
        1. avoid data reach after node is dropped,
        protect from error update
        node-1,U-arch(v0) -> downgrade to:
        node-2,ii-arch(v0) + node-3,U-arch(v0)
        # normal case
        node-2(v0) forward
        node-3 forward, (possible) upgrade to (v1)
        node-2(v0) backward, (possible) upgrade to (v1)
        # normal recover
        node-4,ii-arch(v0) + node-5,U-arch(v0)
        ...
        # temporal error recover
        model state data of previous dropped node-1
        arrives after downgraded, and make downgrade to:
        node-2,ii-arch(v0) + node-3,U-arch(v1)

        2. guard init data is sent earlier than task
        """
        self._breaked = False
        self._underlay_put_resources = underlay_put_resources
        self._underlay_on_train_task_done = underlay_on_infer_task_done
        self._node: TrainNode
        self._init_waitor: asyncio.Future[None] = asyncio.Future()

    def break_inbound(self):
        self._breaked = True
        self._node.dropped = True

    async def _put_resources(self, name: str, res: ModelCKPT):
        if self._breaked:
            return
        else:
            await self._underlay_put_resources(name, res)

    async def _on_train_task_done(self, train_task: TrainTask, cb: Callable[[], None]):
        cb()
        if self._breaked:
            return
        else:
            await self._underlay_on_train_task_done(train_task)

    async def _on_train_task_discard(self, train_task: TrainTask, cb: Callable[[], None]):
        cb()

    async def on_train_task(self, train_task: TrainTask):
        assert not self._breaked
        await self._init_waitor
        context_id = train_task.unique_id_str

        def add_barrier():
            self._node.provider.brain.add_barrier(context_id)

        def remove_barrier():

            self._node.provider.brain.remove_barrier(context_id)
        add_barrier()
        updated_task = await self._node.train(
            train_task=train_task,
            add_barrier=add_barrier,
            remove_barrier=remove_barrier
        )
        if updated_task is None:
            await self._on_train_task_discard(train_task, remove_barrier)
        else:
            await self._on_train_task_done(updated_task, remove_barrier)
            await self._node.try_send_model_update(updated_task.request_version)

    async def train_init(self, config: TrainInit, ckpt_period: int, provider: Provider):
        assert not self._breaked
        self._node = TrainNode(
            ckpt_period=ckpt_period,
            provider=provider,
            put_resources=self._put_resources,
        )

        await self._node.train_init(config)
        self._init_waitor.set_result(None)

    def remove(self):
        self._breaked = True
        self._node.remove()

    def __del__(self):
        self.remove()


class TrainInstance:
    def __init__(self,
                 providers: List[Provider],
                 base_models_lookup: Dict[str, int],
                 groups: List[List[Tuple[str, str]]],
                 user_model_config: Union[PtuningConfig, LoraConfig],
                 optimizer_config: OptimizerConfig,
                 scheduler_config: Optional[SchedulerConfig],
                 data_init_method: UserModelDataInitConfig,
                 test_config: TrainTestConfig,
                 ) -> None:
        self.providers = providers
        self.base_models_lookup = base_models_lookup
        self.groups = groups
        _navi_list: List[str] = [
            dst
            for group in groups
            for (_, dst) in group
        ]
        self.navi_list: List[str] = []
        for i, d in enumerate(_navi_list):
            if i != 0 and i != len(_navi_list)-1:
                d = d+",forward"
            self.navi_list.append(d)
        for i, d in enumerate(_navi_list[::-1]):
            if i != 0 and i != len(_navi_list)-1:
                d = d+",backward"
                self.navi_list.append(d)

        self.user_model_config = user_model_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.data_init_method = data_init_method

        self.train_config: TrainTestConfig = test_config
        self.resources: Dict[str, Any] = {}
        self.nodes_and_data: List[Tuple[Breaker, TrainStateData]] = []
        self._model_look_up: Dict[str, Tuple[Breaker, TrainStateData]] = {}
        # self.model_init_waitor: Dict[str, Set[asyncio.Future[None]]] = {}
        self.result_waitors: Dict[str, asyncio.Future[TrainTask]] = {}
        self.train_step = 0

    # async def _wait_and_get(self, name: str):
    #     ws = self.model_init_waitor.get(name)
    #     if ws is not None:
    #         fut: asyncio.Future[None] = asyncio.Future()
    #         ws.add(fut)
    #         await fut
    #     return self._model_look_up[name]

    # def _set_node_ready(self, name: str, pair: Tuple[Node, TrainStateData]):
    #     # print(f"set ready:{name}")
    #     self._model_look_up[name] = pair
    #     for w in self.model_init_waitor[name]:
    #         w.set_result(None)
    #     del self.model_init_waitor[name]

    # def _set_node_unready(self, name: str):
    #     # print(f"set unready:{name}")
    #     del self._model_look_up[name]
    #     assert name not in self.model_init_waitor
    #     self.model_init_waitor[name] = set()

    async def _get_resource(self, name: str) -> Any:
        if name in self.resources:
            return self.resources[name]
        raise ValueError(name)

    async def _update_ckpt(self, name: str, res: ModelCKPT):
        name = name.removesuffix("-ckpt")
        # print(f"update ckpt: {name} -> {res.version}")
        _, data = self._model_look_up[name]
        data.update_by_ckpt(res)
        return

    async def _on_resource(self, name: str, res: ModelCKPT):
        await self._update_ckpt(name, res)

    def _create_remote_node(self):
        return Breaker(
            self._on_resource,
            self.on_train_task
        )

    async def ainit(self):
        for group in self.groups:
            head: Optional[TrainHeadInit] = None
            tail: Optional[TrainTailInit] = None
            bindings: List[TrainBindingInit] = []

            provider_i = self.base_models_lookup[group[0][0]]

            assert all(
                self.base_models_lookup[src] == provider_i
                for src, _ in group
            ), f"invalid split up {self.base_models_lookup} {[src for src, _ in group]}"

            for src, dst in group:
                desc = ModelDescribe.from_str(src)
                if desc.is_head:
                    head = TrainHeadInit(src, dst, version=0)
                elif desc.is_tail:
                    tail = TrainTailInit(
                        src=src,
                        dst=dst,
                        impl="",
                        version=0
                    )
                else:
                    binding = TrainBindingInit(
                        src=src,
                        dst=dst,
                        version=0,
                        user_model_config=self.user_model_config,
                        optimizer_config=self.optimizer_config,
                        scheduler_config=self.scheduler_config,
                        data_init_method=self.data_init_method,
                    )
                    bindings.append(binding)
            init_config = TrainInit(head, tail, bindings)
            _node = self._create_remote_node()
            await _node.train_init(init_config, self.train_config.backup_period, provider=self.providers[provider_i])
            data = TrainStateData(init_config=init_config)
            pair = (_node, data)
            self.nodes_and_data.append(pair)
            for src, dst in group:
                self._model_look_up[dst] = pair

    def _dispatch_train_task(self, train_task: TrainTask):
        if train_task.navi.finished:
            fut = self.result_waitors.get(train_task.minibatch_id)
            if fut is None:
                return
            del self.result_waitors[train_task.minibatch_id]
            fut.set_result(train_task)
            return
        _model = train_task.navi.target(0)[0]

        async def _dispatch():
            _node, _data = self._model_look_up[_model]
            should_drop = _data.update_by_task(train_task)
            if should_drop:
                self._on_drop(train_task)
                return
            await _node.on_train_task(train_task)
        asyncio.create_task(_dispatch())

    def _on_drop(self, train_task: TrainTask):
        assert train_task.is_cpu
        print(f"task: {train_task} drop on shadow node!")

    async def on_train_task(self, train_task: TrainTask):
        self._dispatch_train_task(train_task)

    async def do_mini_batch(
        self,
        req_version: int,
        full_batch_ids: List[str],
        minibatch_id: str,
        sample: TrainHeadInput,
    ):
        train_task = TrainTask(
            request_version=req_version,
            minibatch_id=minibatch_id,
            batch_ids=full_batch_ids,
            navi=TrainNavigate(
                dir_user_model=self.navi_list,
                current=0,
            ),
            input=sample,
        )

        do_recover, split = self.train_config.should_trigger_recover(self.train_step)
        if do_recover:
            asyncio.create_task(self._recover(split))

        fut: asyncio.Future[TrainTask] = asyncio.Future()
        self.result_waitors[train_task.minibatch_id] = fut
        self._dispatch_train_task(train_task)
        train_task = await fut
        self.train_step += 1

        assert train_task.loss is not None
        return train_task.loss

    async def _recover(self, split: bool = False):
        print(f"recover @ {self.train_step} split={split}")
        pairs: List[Tuple[Breaker, TrainStateData]] = []
        for node_i, (orig_node, data) in enumerate(self.nodes_and_data):
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
                node = self._create_remote_node()
                asyncio.create_task(node.train_init(
                    config=cfg,
                    ckpt_period=self.train_config.backup_period,
                    provider=self.providers[provider_i]
                ))
                pair = (node, data)
                for _m in cfg.user_models:
                    self._model_look_up[_m] = pair
                for task in recover_extra:
                    asyncio.create_task(node.on_train_task(task))
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
                    node = self._create_remote_node()
                    asyncio.create_task(node.train_init(
                        cfg,
                        self.train_config.backup_period,
                        provider=self.providers[provider_i]
                    ))
                    pair = (node, sub_data)
                    for _m in cfg.user_models:
                        self._model_look_up[_m] = pair
                    for task in recover_extra:
                        asyncio.create_task(node.on_train_task(task))
                    pairs.append(pair)
        self.nodes_and_data = pairs
        # for node, tasks in dispatch_later:
        #     for t in tasks:
        #         # print(f"dispatch task: {t}")
        #         asyncio.create_task(node.on_train_task(t))

    def remove(self):
        for node, _ in self.nodes_and_data:
            node.remove()
        self._model_look_up.clear()
        self.resources.clear()
        self.result_waitors.clear()
