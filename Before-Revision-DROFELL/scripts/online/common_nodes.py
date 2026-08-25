import time
from typing import Optional, cast

import psutil
import torch

from dealers.analysis.recorder import TimeRecorder
from dealers.analysis.frame import AnalysisFrame
from dealers.memory_schedule.restriction import MemoryConfig
import extend_asyncio as asyncio
from dealers.memory_schedule.brain import Brain
from node.aux_node.aux_node import AuxNode
from node.backbone import BackboneNode
from node.base import BaseNode
from node.dealer_node.dealer_node import BackboneSelectConfig, DealerNode, ModelTargets
from operators.container.base_container import SharedBaseContainer
from orbit.defines import Context
from orbit.link.extender import ContextNetExtender

from .models import extra_info, lib_path


def aux_node(ctx: Context):
    async def coro(ctx: Context):
        ctx.report_ready()
        net_extend = ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        node = AuxNode(base_node, extra_info)

        def _on_request_telemetric_ablitity(node: str, ticket: int):
            base_node.send_resp(node, ticket, [])

        base_node.apis["telemetric-ablities"] = _on_request_telemetric_ablitity

        while True:
            base_node.on_packet(await ctx.recv())

    asyncio.get_event_loop().create_task(coro(ctx))


def dealer_coro(
    ctx: Context,
    device_str: str,
    models: ModelTargets,
    memory_config: Optional[MemoryConfig] = None,
    backbone: Optional[BackboneSelectConfig] = None,
    time_trace: bool = False,
    time_trace_with_sync_cuda: bool = False,
):
    if time_trace_with_sync_cuda:
        assert time_trace

    net_extend = ContextNetExtender(ctx)
    device = torch.zeros((0,), device=device_str).device
    container = SharedBaseContainer(cast(torch.cuda.Stream, torch.cuda.Stream(device)))

    base_node = BaseNode(ctx)
    base_node.perf_net = net_extend.pref_net
    dealer = DealerNode(base_node, backbone_select_config=backbone)

    time_recorder = TimeRecorder(sync_cuda=time_trace_with_sync_cuda, enabled=False) if time_trace else None

    def _on_start_time_telemetric(node: str, ticket: int):
        assert time_recorder is not None
        time_recorder.reset(True)
        ctx.logger.warn("time tracing started")
        base_node.send_resp(node, ticket, time.time_ns())

    def _on_end_time_telemetric(node: str, ticket: int):
        assert time_recorder is not None
        info = time_recorder.dump_info()
        time_recorder.reset(False)
        ctx.logger.warn("time tracing end")
        base_node.send_resp(node, ticket, info)

    base_node.apis["telemetric-time-start"] = _on_start_time_telemetric
    base_node.apis["telemetric-time-end"] = _on_end_time_telemetric

    async def init_coro():
        init_time = time.time()
        brain = Brain(
            models_path=lib_path,
            device=device,
            container=container,
            memory_config=memory_config,
        )
        if time_trace:
            base_node.tags.add("time_trace")
            ctx.logger.warn("time tracing embedded")
            assert time_recorder is not None
            brain = AnalysisFrame(
                brain=brain,
                recorder=time_recorder
            )
        await dealer.ainit(
            brain=brain,  # type:ignore
            models=models,
        )

        p = psutil.Process()
        start_time = p.create_time()
        elapsed = time.time() - start_time
        ctx.logger.info(f"init in {time.time()-init_time}s, process time: {elapsed}s")
        ctx.report_ready()
    return base_node, dealer, init_coro


def dealer_node(ctx: Context):
    async def coro(
        ctx: Context,
        device_str: str,
        models: ModelTargets,
        memory_config: Optional[MemoryConfig] = None,
        backbone: Optional[BackboneSelectConfig] = None,
        time_trace: bool = False,
        time_trace_with_sync_cuda: bool = False,
    ):
        base_node, dealer, init_coro = dealer_coro(ctx, device_str, models, memory_config, backbone, time_trace, time_trace_with_sync_cuda)
        asyncio.create_task(init_coro())

        while True:
            packet = await ctx.recv()
            base_node.on_packet(packet)

    asyncio.get_event_loop().create_task(coro(ctx, *ctx.args.args, **ctx.args.kwargs))


def backbone_node(ctx: Context):
    async def coro(ctx: Context):
        ctx.report_ready()
        net_extend = ContextNetExtender(ctx)

        base_node = BaseNode(ctx)
        BackboneNode(base_node)

        def _on_request_telemetric_ablitity(node: str, ticket: int):
            base_node.send_resp(node, ticket, [])

        base_node.apis["telemetric-ablities"] = _on_request_telemetric_ablitity

        while True:
            packet = await ctx.recv()
            base_node.on_packet(packet)

    asyncio.get_event_loop().create_task(coro(ctx))
