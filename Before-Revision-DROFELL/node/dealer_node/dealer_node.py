import atexit
from dataclasses import dataclass, field
import json
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cache
from dealers.common.base_types import ModelDescribe
import extend_asyncio as asyncio

from dealers.common.provider import Provider
from dealers.common.brain import Brain, Profile
from ..infer.sender import InferBatchSender
from orbit.link.define import RecvPref, SendPerf
from utils.format import format_size
from ..base import BaseNode
from ..aux_node.protocols import API_Announce, API_RegBackboneNode, Announce, ExtraInfo, req_backbone_nodes, req_extra_info, req_tps
from node.compute_node import TrainNode, ModelCKPT
from node.compute_node import InferNode, PutBlocks
from .protocols import API_CreateInferNode, API_CreateTrainNode, API_GroupedInferTask, API_InferTask, API_InferPutBlocks, API_NodeSuicide, API_RemoveNode, API_TrainModelCKPT, API_TrainTask, CreateInferNode, CreateTrainNode, GroupedInferTask, InferPutBlocks, MultiPathInferTask, MultiPathTrainTask, RemoveNode, TrainModelCKPT


@dataclass
class ModelTargets:
    specific: bool
    decoders: List[str]
    bundle: Dict[str, List[str]] = field(default_factory=dict)
    # only when specific= False
    memory_limit: int = 0
    # only when specific= False
    decoders_limit: int = 0
    auto: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.specific:
            assert self.memory_limit == 0
            assert self.decoders_limit == 0
        else:
            assert self.memory_limit != 0 or self.decoders_limit != 0

    def __str__(self):
        if self.specific:
            s = "ModelLoadTarget (Specificed):\n  "
            targets: List[str] = []
            for d in self.decoders:
                targets.append(", ".join([d, *self.bundle.get(d, [])]))
            s += "\n  ".join(targets)
            return s
        else:
            s = "ModelLoadTarget (Auto):\n  " +\
                "candidates:\n    " +\
                "\n    ".join(self.auto) +\
                f"memory limit: {format_size(self.memory_limit)}"
            return s

    @classmethod
    def from_specific(cls, decoders: List[str], bundle: Dict[str, List[str]]):
        return ModelTargets(specific=True, decoders=decoders, bundle=bundle)

    @classmethod
    def from_auto(cls, model_names: List[str], memory_limit: int = 0, decoders_limit: int = 0):
        return ModelTargets(specific=False, decoders=[], bundle={}, memory_limit=memory_limit, decoders_limit=decoders_limit, auto=model_names)

    @property
    def base_model_names(self):
        base_models: List[str] = []
        for model in self.decoders:
            base_model = ModelDescribe.from_str(model).base_model
            base_models.append(base_model)
        base_models = list(set(base_models))
        return base_models or self.auto

    def add_bundle(self, base: str, extra: str):
        if not base in self.bundle:
            self.bundle[base] = []
        assert not extra in self.bundle[base]
        self.bundle[base].append(extra)

    def expand_bundle(self, models: List[str]) -> List[str]:
        expand_model: List[str] = []
        for m in models:
            if m in self.bundle:
                expand_model += self.bundle[m]
            expand_model.append(m)
        return expand_model

    def shallow_copy(self, **kwargs: Any):
        args = self.__dict__.copy()
        args.update(kwargs)
        return ModelTargets(**args)


@dataclass
class BackboneSelectConfig:
    auto: bool = True
    send_latency: float = 3
    recv_latency: float = 3
    send_bandwidth: float = 128
    recv_bandwidth: float = 128
    specific: Set[str] = field(default_factory=set)

    def can_select(self, node: str, send_perf: Optional[SendPerf] = None, recv_perf: Optional[RecvPref] = None) -> bool:
        if not self.auto:
            return node in self.specific
        assert send_perf is not None
        assert recv_perf is not None
        if self.send_latency > 0:
            if send_perf.latency_ms > self.send_latency:
                return False
        if self.send_bandwidth > 0:
            if send_perf.bandwidth_m < self.send_bandwidth:
                return False
        if self.recv_latency > 0:
            if recv_perf.latency_ms > self.recv_latency:
                return False
        if self.recv_bandwidth > 0:
            if recv_perf.bandwidth_m < self.recv_bandwidth:
                return False
        return True

    @classmethod
    def from_specific(cls, specific: Set[str]):
        return BackboneSelectConfig(auto=False, specific=specific)

    @classmethod
    def from_auto(cls, send_perf: SendPerf = SendPerf(latency_ms=3, bandwidth_mbps=128), recv_perf: RecvPref = SendPerf(latency_ms=3, bandwidth_mbps=128)):
        return BackboneSelectConfig(
            auto=True,
            send_latency=send_perf.latency_ms,
            send_bandwidth=send_perf.bandwidth_m,
            recv_latency=recv_perf.latency_ms,
            recv_bandwidth=recv_perf.bandwidth_m
        )


class DealerNode:
    def __init__(
        self,
        base_node: BaseNode,
        backbone_select_config: Optional[BackboneSelectConfig] = None
    ) -> None:
        self.base = base_node
        self.infer_batch_sender = InferBatchSender(base_node.node_name, base_node.send)
        if backbone_select_config is None:
            backbone_select_config = BackboneSelectConfig()
        self.backbone_select_config = backbone_select_config
        self.base.tags.add("dealer")
        self.base.apis[API_CreateInferNode] = self.on_create_infer_node
        self.base.apis[API_CreateTrainNode] = self.on_create_train_node
        self.base.apis[API_InferTask] = self.on_infer_task
        self.base.apis[API_TrainTask] = self.on_train_task
        self.base.apis[API_GroupedInferTask] = self.on_grouped_infer_task
        self.base.apis[API_RegBackboneNode] = self.on_new_backbone
        self.base.apis[API_RemoveNode] = self.on_remove_node
        self.base.link_change_cbs.append(self._on_link_change)
        self.brain: Brain
        self.extra_info: ExtraInfo
        # self.models: ModelTargets

        self.provider: Provider
        self.announce_memory: Tuple[int, int]
        self.announce_profile: Dict[str, Profile] = {}
        self.announce: Optional[Announce] = None

        self.connected_backbones: Set[str] = set()

        self.models_loaded: Set[str] = set()
        # {infer/train_node_name:(node,handle_host_node)}
        self.name_to_nodes: Dict[str, Tuple[Union[InferNode, TrainNode], str]] = {}
        self.name_to_node_init_waitor: Dict[str, asyncio.Future[None]] = {}
        self.handle_node_to_nodes: Dict[str, Set[str]] = {}
        # multiple nodes could have same first user model
        # so {user_model:set(multiple_nodes)}
        self.user_model_to_nodes: Dict[str, List[str]] = {}

        atexit.register(self._debug_check_resource_release)

    async def ainit(self, brain: Brain, models: ModelTargets):
        self.brain = brain
        self.extra_info = await req_extra_info(self.base)
        self.provider = Provider(self.brain)
        # self.models = models

        self.base.logger.info("node inited, start computing profiles")
        await self.init_profile(models.base_model_names)
        self.base.logger.info("profile done, looking up backbones")
        await self.lookup_backbones()
        self.base.logger.info("looking up done, decide models to load")
        if models.specific:
            target_models = models.expand_bundle(models.decoders)
        else:
            target_models = await self.decide_models_to_load(models)
        if len(target_models) != 0:
            self.base.logger.info(f"decide to load: \n  "+'\n  '.join(target_models))
        else:
            self.base.logger.info(f"no model to load")
        self.pre_log_base_model_memory(target_models)
        self.base.logger.info("staring loading...")
        await self._load_models(target_models)
        self.models_loaded = set(target_models)
        self.announce_memory = self.brain.memory
        self.base.logger.info(f"Total memory: {format_size(self.announce_memory[0])}")
        self.base.logger.info(f"Base model memory: {format_size(self.announce_memory[1])}")
        detail_memory = self.provider.brain.query_extra_info("detail_memory")
        if detail_memory is not None:
            self.base.logger.info("detail_memory")
            self.base.logger.info("\n"+detail_memory)
        detail_memory_restriction = self.provider.brain.query_extra_info("detail_memory_restriction")
        if detail_memory_restriction is not None:
            self.base.logger.info("detail_memory_restriction")
            self.base.logger.info("\n"+detail_memory_restriction)
        self.base.logger.info("load done, announcing")
        self.announce = Announce(*self.announce_memory, self.announce_profile, self.models_loaded)
        self.base.send(("aux", API_Announce, self.announce))
        for backbone in self.connected_backbones:
            self.base.send((backbone, API_Announce, self.announce))

    async def init_profile(
        self,
        models: List[str],
        cache_dir: Optional[str] = os.path.join(cache.cache_path, "profile")
    ):
        if cache_dir is None:
            for model in models:
                profile = await self.brain.gen_profile(model)
                self.announce_profile[model] = profile
            return
        os.makedirs(cache_dir, exist_ok=True)
        for model in models:
            store_name = f"device-{self.brain.device.index}-{model}"
            store_path = os.path.join(cache_dir, store_name+".json")
            if os.path.exists(store_path):
                with open(store_path, "r") as f:
                    self.base.logger.info(f"load profile from cache: {store_path}")
                    data = json.load(f)
                    profile = Profile(**data)
                    self.announce_profile[model] = profile
            else:
                profile = await self.brain.gen_profile(model)
                self.announce_profile[model] = profile
                with open(store_path, "w") as f:
                    json.dump(profile.__dict__, f, indent="  ")
            self.base.logger.info(f"Profile of :{model} (Per-Decoder)\n  "+str(profile).replace("\n", "\n  "))

    async def check_backbone(self, node: str) -> bool:
        if not self.backbone_select_config.auto:
            return self.backbone_select_config.can_select(node)
        else:
            if not await self.base.try_connect(node):
                return False
            send, recv = await self.base.perf_net(node)
            return self.backbone_select_config.can_select(node, send, recv)

    async def lookup_backbones(self):
        backbones = await req_backbone_nodes(self.base)
        self.base.logger.info(f"checking backbones: {backbones}")
        self.connected_backbones = set([node for node in backbones if await self.check_backbone(node)])
        self.base.logger.info(f"connected backbones: {self.connected_backbones}")

    def on_new_backbone(self, src: str, node: str):
        async def coro():
            self.base.logger.info(f"checking new backbone: {node}")
            if await self.check_backbone(node):
                self.base.logger.info(f"new backbone:{node}, all backbones: {self.connected_backbones}")
                self.base.send((node, API_Announce, self.announce))
            else:
                self.base.logger.info(f"reject backbone: {node}")
        asyncio.create_task(coro())

    async def decide_models_to_load(self, models: ModelTargets) -> List[str]:
        if len(models.decoders) == 0:
            for base_model in models.auto:
                extra_info = self.extra_info.models[base_model]
                models.decoders.extend(extra_info.decoders_groups)
                models.add_bundle(extra_info.decoders_groups[0], extra_info.head)
                models.add_bundle(extra_info.decoders_groups[-1], extra_info.tail)
        profiles = await req_tps(self.base, models.decoders)
        # print(profiles)
        priority: List[Tuple[str, int]] = []
        for (model, kind), tps in profiles.items():
            if kind != "generating":
                continue
            if tps < 0:
                # head or tail, not sort
                continue
            priority.append((model, tps))
        priority.sort(key=lambda pair: (pair[1], pair[0]))
        target_models: List[str] = []
        decoders_count: int = 0
        current_memory: int = 0
        for model, tps in priority:
            desc = ModelDescribe.from_str(model)
            decoder_memory = self.announce_profile[desc.base_model].perdecoder_memory
            group_memory = desc.num_decoders*decoder_memory
            decoders_count += desc.num_decoders
            if models.memory_limit > 0 and current_memory+group_memory >= models.memory_limit:
                break
            if models.decoders_limit > 0 and decoders_count > models.decoders_limit:
                break
            target_models += models.expand_bundle([model])
            current_memory += group_memory
        return target_models

    def pre_log_base_model_memory(self, target_models: List[str]):
        total_memory, _ = self.brain.memory
        s = f"total memory in this device: {format_size(total_memory,2)}"
        extra_models: List[str] = []
        grouped_model: Dict[str, List[str]] = {}
        for model in target_models:
            desc = ModelDescribe.from_str(model)
            if not desc.is_decoders:
                extra_models.append(model)
                continue
            grouped_model.setdefault(desc.base_model, []).append(model)
        s += "\nbelow is estimated memory, usually smaller than actual required. Accurate memory usage will be given after model loaded if provider is set correctly."
        total_decoders_memory = 0
        for base_model, models in grouped_model.items():
            decoder_memory = self.announce_profile[base_model].perdecoder_memory
            total_group_memory = decoder_memory*sum(ModelDescribe.from_str(m).num_decoders for m in models)
            total_decoders_memory += total_group_memory
            s += f"\n  base model: {base_model} estimate memory: {format_size(total_group_memory,2)}"
            for model in models:
                desc = ModelDescribe.from_str(model)
                group_memory = desc.num_decoders*decoder_memory
                s += f"\n    {model} {format_size(group_memory,2)}"
        s += f"\ntotal estimated decoders memory: {format_size(total_decoders_memory,2)}"
        for model in extra_models:
            s += f"\n  extra models: {extra_models}"

        self.base.logger.info(s)

    async def _load_models(self, models: List[str]):
        for i, model in enumerate(models):
            self.base.logger.info(f"loading: {i+1}/{len(models)} {model}")
            st = time.time()
            self.provider[model]
            total, base_model = self.provider.brain.memory
            self.base.logger.info(f"{time.time()-st:.2f}s {format_size(base_model)}/{format_size(total)} consumed")

    async def _on_infer_put_blocks(self, reciver_node: str, infer_node: str, res: PutBlocks):
        self.base.send((reciver_node, API_InferPutBlocks, InferPutBlocks(infer_node_name=infer_node, blocks=res)))

    async def _on_train_put_ckpts(self, reciver_node: str, train_node: str,  res: ModelCKPT):
        self.base.send((reciver_node, API_TrainModelCKPT, TrainModelCKPT(train_node_name=train_node, ckpt=res)))

    def on_create_infer_node(self, handle_host_node: str, cfg: CreateInferNode):
        infer_node_name = cfg.name
        # assert infer_node_name not in self.name_to_node_init_waitor
        # assert infer_node_name not in self.name_to_nodes
        assert infer_node_name not in self.name_to_node_init_waitor, infer_node_name
        assert infer_node_name not in self.name_to_nodes, infer_node_name
        fut: asyncio.Future[None] = asyncio.Future()
        self.name_to_node_init_waitor[infer_node_name] = fut
        self.handle_node_to_nodes.setdefault(handle_host_node, set()).add(infer_node_name)
        for dst in cfg.config.user_models:
            assert not dst in self.user_model_to_nodes.get(dst, [])
            self.user_model_to_nodes.setdefault(dst, []).append(infer_node_name)

        init_config = cfg.config
        if cfg.block_retrieve:
            async def put_res(model_name: str, res: PutBlocks):
                await self._on_infer_put_blocks(reciver_node=handle_host_node, infer_node=infer_node_name, res=res)
        else:
            async def put_res(model_name: str, res: PutBlocks): pass
        _node = InferNode(
            block_retrieve=cfg.block_retrieve,
            provider=self.provider,
            put_resources=put_res,
        )

        async def coro():
            self.base.logger.info(f"creating infer node: {infer_node_name} with {init_config}")
            await _node.infer_init(init_config)
            self.name_to_nodes[infer_node_name] = (_node, handle_host_node)
            fut.set_result(None)
            del self.name_to_node_init_waitor[infer_node_name]
            self.base.logger.info(f"create infer node: {infer_node_name} complete")
        asyncio.create_task(coro())

    def on_grouped_infer_task(self, src: str, grouped_task: GroupedInferTask):
        for t in grouped_task.tasks:
            self.on_infer_task(src, t, True)

    async def handle_infer_task(self, node_name: str, cpy_task: MultiPathInferTask, cpy_metrics: Optional[List[Tuple[str, int]]] = None):
        await self.wait_node_init(node_name)
        node, _ = self.name_to_nodes.get(node_name, (None, ""))
        if node is None:
            return
        assert isinstance(node, InferNode)
        context_id = cpy_task.task.unique_id_str
        if cpy_task.trace_override:
            this_len = len(node.user_models)
            override_len = len(cpy_task.trace_override)
            if override_len == this_len:
                cpy_task.trace.extend(cpy_task.trace_override)
                cpy_task.trace_override = None
            elif override_len > this_len:
                cpy_task.trace.extend(cpy_task.trace_override[:this_len])
                cpy_task.trace_override = cpy_task.trace_override[this_len:]
            else:
                raise ValueError(f"override len: {override_len} < this len: {this_len}")
        else:
            cpy_task.trace.extend([self.base.node_name]*len(node.user_models))

        def add_barrier():
            self.provider.brain.add_barrier(context_id)

        def remove_barrier():
            self.provider.brain.remove_barrier(context_id)
        add_barrier()
        updated_task = await node.infer(
            infer_task=cpy_task.task,
            add_barrier=add_barrier,
            remove_barrier=remove_barrier,
            metrics=cpy_metrics
        )
        remove_barrier()
        if updated_task is None:
            return
        cpy_task.task = updated_task
        current_targets = cpy_task.resource_loc[updated_task.navi.current]

        def dispatch(target: str):
            add_barrier()
            forked_task = cpy_task.detail_trace_fork()
            _start_trap = remove_barrier
            _end_trap = add_barrier
            _send_cb = remove_barrier

            if cpy_metrics is not None:
                forked_metrics = cpy_metrics.copy()

                def start_trap():
                    forked_metrics.append(("start_trap", time.time_ns()))
                    remove_barrier()
                _start_trap = start_trap

                def end_trap():
                    forked_metrics.append(("end_trap", time.time_ns()))
                    add_barrier()
                _end_trap = end_trap

                def send_cb():
                    forked_metrics.append(("leave", time.time_ns()))
                    forked_task.extend_steps(forked_metrics)
                    remove_barrier()
                _send_cb = send_cb

            self.infer_batch_sender.send(
                target=target,
                task=forked_task,
                start_trap_cb=_start_trap,
                end_trap_cb=_end_trap,
                send_cb=_send_cb
            )

        for target in current_targets:
            dispatch(target)

    def on_infer_task(self, src: str, task: MultiPathInferTask, from_group: bool = False):
        # print(f"dealer node {self.node_name} recv: {task.task} from {src}")
        model, _, _ = task.task.navi.target(0)
        node_names = self.user_model_to_nodes.get(model, [])
        if len(node_names) == 0:
            self.base.logger.error(f"cannot find {model} for task: {task.task.unique_id}, discard")
            return
        metrics = None
        if task.detail_trace is not None:
            metrics = [("dealer-recv", time.time_ns())]

        for node_name in node_names:
            asyncio.create_task(self.handle_infer_task(node_name, task.shallow_copy(), None if metrics is None else metrics.copy()))

    def on_create_train_node(self, handle_host_node: str, cfg: CreateTrainNode):
        node_name = cfg.name
        assert node_name not in self.name_to_node_init_waitor
        assert cfg.name not in self.name_to_nodes
        fut: asyncio.Future[None] = asyncio.Future()
        self.name_to_node_init_waitor[node_name] = fut
        self.handle_node_to_nodes.setdefault(handle_host_node, set()).add(node_name)
        for dst in cfg.config.user_models:
            assert not dst in self.user_model_to_nodes.get(dst, [])
            self.user_model_to_nodes.setdefault(dst, []).append(node_name)

        init_config = cfg.config
        if cfg.ckpt_period != 0:

            async def put_res(model_name: str, res: ModelCKPT):
                await self._on_train_put_ckpts(reciver_node=handle_host_node, train_node=node_name, res=res)
        else:
            async def put_res(model_name: str, res: ModelCKPT): pass
        _node = TrainNode(
            ckpt_period=cfg.ckpt_period,
            provider=self.provider,
            put_resources=put_res,
        )

        async def coro():
            await _node.train_init(init_config)
            self.name_to_nodes[node_name] = (_node, handle_host_node)
            fut.set_result(None)
            del self.name_to_node_init_waitor[node_name]
        asyncio.create_task(coro())

    async def handle_train_task(self, node_name: str, cpy_task: MultiPathTrainTask, is_backward: bool, cpy_metrics: Optional[List[Tuple[str, int]]] = None):
        await self.wait_node_init(node_name)
        node, _ = self.name_to_nodes.get(node_name, (None, ""))
        if node is None:
            return
        assert isinstance(node, TrainNode)
        context_id = cpy_task.task.unique_id_str
        this_len = node.model_sequence_len(is_backward=is_backward)
        if cpy_task.trace_override:

            override_len = len(cpy_task.trace_override)
            if override_len == this_len:
                cpy_task.trace.extend(cpy_task.trace_override)
                cpy_task.trace_override = None
            elif override_len > this_len:
                cpy_task.trace.extend(cpy_task.trace_override[:this_len])
                cpy_task.trace_override = cpy_task.trace_override[this_len:]
            else:
                raise ValueError(f"override len: {override_len} < this len: {this_len}")
        else:
            cpy_task.trace.extend([self.base.node_name]*this_len)

        def add_barrier():
            self.provider.brain.add_barrier(context_id)

        def remove_barrier():
            self.provider.brain.remove_barrier(context_id)
        add_barrier()
        updated_task = await node.train(
            train_task=cpy_task.task,
            add_barrier=add_barrier,
            remove_barrier=remove_barrier,
            metrics=cpy_metrics
        )
        remove_barrier()
        if updated_task is None:
            return
        cpy_task.task = updated_task
        await node.try_send_model_update(updated_task.request_version)
        current_targets = cpy_task.resource_loc[updated_task.navi.current]

        def dispatch(target: str):
            forked_task = cpy_task.detail_trace_fork()

            if cpy_metrics is not None:
                forked_metrics = cpy_metrics.copy()
                forked_metrics.append(("leave", time.time_ns()))
                forked_task.extend_steps(forked_metrics)

            # self.base.logger.error(f"send: {forked_task} -> {target}")
            self.base.send((target, API_TrainTask, forked_task), not_exist_ok=True)

        for target in current_targets:
            dispatch(target)

    def on_train_task(self, src: str, task: MultiPathTrainTask):
        # self.base.logger.error(f"{self.base.node_name} receives {task} from {src}")
        model = task.task.navi.current_dir_user_model
        is_backward = model.endswith(",backward")
        model = model.removesuffix(",forward").removesuffix(",backward")
        node_names = self.user_model_to_nodes.get(model, [])
        if len(node_names) == 0:
            self.base.logger.error(f"cannot find {model} for task: {task.task.unique_id_str}, discard")
            return
        metrics = None
        if task.detail_trace is not None:
            metrics = [("dealer-recv", time.time_ns())]

        for node_name in node_names:
            asyncio.create_task(self.handle_train_task(node_name, task.shallow_copy(), is_backward, None if metrics is None else metrics.copy()))

    async def wait_node_init(self, node: str):
        if node not in self.name_to_node_init_waitor:
            return
        fut = self.name_to_node_init_waitor[node]
        await fut

    def _remove_node(self, name: str):
        self.base.logger.info(f"remove node: {name}")
        if name in self.name_to_node_init_waitor:
            self.name_to_node_init_waitor[name].set_result(None)
            del self.name_to_node_init_waitor[name]
        target = self.name_to_nodes.get(name)
        if target is None:
            return
        node, handle_host_node = target
        node.remove()
        user_models = node.user_models
        del self.name_to_nodes[name]
        self.handle_node_to_nodes[handle_host_node].remove(name)
        if not self.handle_node_to_nodes[handle_host_node]:
            del self.handle_node_to_nodes[handle_host_node]
        for model in user_models:
            self.user_model_to_nodes[model] = [n for n in self.user_model_to_nodes[model] if n != name]
            if not self.user_model_to_nodes[model]:
                del self.user_model_to_nodes[model]

    def on_remove_node(self, src: str, data: RemoveNode):
        name = data.name
        self._remove_node(name)

    def suicide_node(self, node_name: str):
        self.base.logger.error(f"node suicide: {node_name}")
        # TODO finish slow snail suicide logic
        _, handler_host_node = self.name_to_nodes[node_name]
        self._remove_node(node_name)
        self.base.send((handler_host_node, API_NodeSuicide, node_name))

    def _on_link_change(self, node: str, drop: bool, dsts: Set[str]):
        if not drop:
            return
        nodes = self.handle_node_to_nodes.get(node, None)
        if not nodes:
            return
        for n in list(nodes):
            self._remove_node(n)
        self.handle_node_to_nodes.pop(node, None)

    def _debug_check_resource_release(self):
        assert not self.name_to_nodes, f"resource not release @ {self.base.node_name}: {list(self.name_to_nodes.keys())}"
        assert not self.name_to_node_init_waitor
        assert not self.handle_node_to_nodes
        assert not self.user_model_to_nodes
        print("dealer node all released")
