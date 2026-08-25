import time
import os
from typing import Optional, cast

import psutil
import torch

from dealers.analysis.recorder import TimeRecorder
from dealers.analysis.frame import AnalysisFrame
from dealers.memory_schedule.restriction import (
    MemoryConfig,
    MemoryConfigPercentage,
    MemoryConfigStatic,
)
import operators.flags as operator_flags
import extend_asyncio as asyncio
from dealers.memory_schedule.brain import Brain
from node.aux_node.aux_node import AuxNode
from node.aux_node.protocols import ExtraInfo, req_extra_info
from node.backbone import BackboneNode
from node.base import BaseNode
from node.dealer_node.dealer_node import BackboneSelectConfig, DealerNode, ModelTargets
from operators.container.base_container import SharedBaseContainer
from operators.container.container import NonAsyncContainer
from orbit.defines import Context
from orbit.link.extender import ContextNetExtender

from .models import extra_info, lib_path
from .runtime_config import (
    API_DECODE_GRAPH_STATS,
    OnlineInferenceConfig,
)
from utils.runtime_metrics import record_event


def resolve_memory_config(memory_config: Optional[MemoryConfig]) -> MemoryConfig:
    if memory_config is not None:
        record_event(
            "memory_policy_config",
            config_type=type(memory_config).__name__,
            values={
                name: getattr(memory_config, name)
                for name in (
                    "operate_reserved",
                    "graph_budget",
                    "unmanaged_reserved",
                    "minor_minimum",
                    "minor_maximum",
                    "major_percentage",
                )
                if hasattr(memory_config, name)
            },
            requested_type="explicit",
        )
        return memory_config

    requested_type = os.getenv("MEMORY_CONFIG_TYPE", "existing")
    if requested_type == "existing":
        config = MemoryConfigStatic()
    elif requested_type == "percentage":
        config = MemoryConfigPercentage()
    elif requested_type == "static":
        config = MemoryConfigStatic()
    else:
        raise ValueError(
            f"MEMORY_CONFIG_TYPE must be existing, percentage, or static, got {requested_type!r}")
    integer_fields = {
        "MEMORY_OPERATE_RESERVED": "operate_reserved",
        "MEMORY_GRAPH_BUDGET": "graph_budget",
        "MEMORY_UNMANAGED_RESERVED": "unmanaged_reserved",
        "MEMORY_MINOR_MINIMUM": "minor_minimum",
        "MEMORY_MINOR_MAXIMUM": "minor_maximum",
    }
    float_fields = {"MEMORY_MAJOR_PERCENTAGE": "major_percentage"}
    for env_name, field_name in integer_fields.items():
        if env_name in os.environ:
            setattr(config, field_name, int(os.environ[env_name]))
    for env_name, field_name in float_fields.items():
        if env_name in os.environ:
            setattr(config, field_name, float(os.environ[env_name]))
    record_event(
        "memory_policy_config",
        config_type=type(config).__name__,
        values={
            name: getattr(config, name)
            for name in (
                "operate_reserved",
                "graph_budget",
                "unmanaged_reserved",
                "minor_minimum",
                "minor_maximum",
                "major_percentage",
            )
            if hasattr(config, name)
        },
        requested_type=requested_type,
    )
    return config


def configure_operator_policy_from_env() -> None:
    train_method = os.getenv("DECODERS_TRAIN_METHOD",
                             operator_flags.DECODERS_TRAIN_METHOD)
    if train_method not in {"grad", "each_decoder", "first.grad", "first.2pass"}:
        raise ValueError(f"invalid DECODERS_TRAIN_METHOD: {train_method!r}")
    downgrade = os.getenv("DECODERS_TRAIN_DOWNGRAD_METHOD",
                          operator_flags.DECODERS_TRAIN_DOWNGRAD_METHOD)
    if downgrade not in {"first.grad", "first.2pass"}:
        raise ValueError(
            f"invalid DECODERS_TRAIN_DOWNGRAD_METHOD: {downgrade!r}")
    checkpoint = os.getenv("USER_MODEL_ENABLE_CHECKPOINT", "1").strip(
    ).lower() not in {"0", "false", "no", "off"}
    reuse_reconstruct_cache = os.getenv(
        "REUSE_RECONSTRUCT_CACHE_WHEN_GRAD",
        "1" if operator_flags.REUSE_RECONSTRUCT_CACHE_WHEN_GRAD else "0",
    ).strip().lower() not in {"0", "false", "no", "off"}
    math_attention_value = os.getenv(
        "USE_PYTORCH_MATH_ATTENTION",
        "1" if operator_flags.USE_PYTORCH_MATH_ATTENTION else "0",
    ).strip().lower()
    if math_attention_value not in {
        "0", "false", "no", "off", "", "1", "true", "yes", "on",
    }:
        raise ValueError(
            "USE_PYTORCH_MATH_ATTENTION must be a boolean value, got "
            f"{math_attention_value!r}"
        )
    math_attention = math_attention_value in {"1", "true", "yes", "on"}
    # type: ignore[assignment]
    operator_flags.DECODERS_TRAIN_METHOD = train_method
    # type: ignore[assignment]
    operator_flags.DECODERS_TRAIN_DOWNGRAD_METHOD = downgrade
    operator_flags.USER_MODEL_ENABLE_CHECKPOTIN = checkpoint
    operator_flags.REUSE_RECONSTRUCT_CACHE_WHEN_GRAD = (
        reuse_reconstruct_cache
    )
    operator_flags.USE_PYTORCH_MATH_ATTENTION = math_attention
    record_event(
        "operator_policy_config",
        train_method=train_method,
        downgrade_method=downgrade,
        user_model_checkpoint=checkpoint,
        reuse_reconstruct_cache_when_grad=reuse_reconstruct_cache,
        pytorch_math_attention=math_attention,
    )


def aux_node(ctx: Context):
    async def coro(
        ctx: Context,
        runtime_extra_info: Optional[ExtraInfo] = None,
    ):
        ctx.report_ready()
        net_extend = ContextNetExtender(ctx)
        base_node = BaseNode(ctx)
        node = AuxNode(
            base_node,
            extra_info if runtime_extra_info is None else runtime_extra_info,
        )

        def _on_request_telemetric_ablitity(node: str, ticket: int):
            base_node.send_resp(node, ticket, [])

        base_node.apis["telemetric-ablities"] = _on_request_telemetric_ablitity

        while True:
            base_node.on_packet(await ctx.recv())

    asyncio.get_event_loop().create_task(
        coro(ctx, *ctx.args.args, **ctx.args.kwargs)
    )


def dealer_coro(
    ctx: Context,
    device_str: str,
    models: ModelTargets,
    memory_config: Optional[MemoryConfig] = None,
    backbone: Optional[BackboneSelectConfig] = None,
    time_trace: bool = False,
    time_trace_with_sync_cuda: bool = False,
    inference_config: Optional[OnlineInferenceConfig] = None,
    expected_network_name: str = "",
    expected_network_spec_hash: str = "",
    execution_container: str = "shared",
):
    if time_trace_with_sync_cuda:
        assert time_trace

    net_extend = ContextNetExtender(ctx)
    device = torch.zeros((0,), device=device_str).device
    stream = torch.cuda.Stream(device=device)
    if execution_container == "shared":
        container = SharedBaseContainer(stream)
    elif execution_container == "nonasync":
        container = NonAsyncContainer(stream)
    else:
        raise ValueError(
            "execution_container must be 'shared' or 'nonasync', got "
            f"{execution_container!r}"
        )

    base_node = BaseNode(ctx)
    base_node.perf_net = net_extend.pref_net
    dealer = DealerNode(base_node, backbone_select_config=backbone)

    time_recorder = TimeRecorder(
        sync_cuda=time_trace_with_sync_cuda, enabled=False) if time_trace else None

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
        if expected_network_name or expected_network_spec_hash:
            network_info = await req_extra_info(base_node)
            if network_info.network_name != expected_network_name:
                raise RuntimeError(
                    "network name mismatch: "
                    f"expected={expected_network_name!r}, "
                    f"actual={network_info.network_name!r}"
                )
            if network_info.network_spec_hash != expected_network_spec_hash:
                raise RuntimeError(
                    "network spec hash mismatch: "
                    f"expected={expected_network_spec_hash}, "
                    f"actual={network_info.network_spec_hash}"
                )
        configure_operator_policy_from_env()
        configured_memory = resolve_memory_config(memory_config)
        configured_inference = (
            OnlineInferenceConfig.from_env()
            if inference_config is None
            else inference_config
        )
        brain = Brain(
            models_path=lib_path,
            device=device,
            container=container,
            memory_config=configured_memory,
            max_seq_len=configured_inference.max_seq_len,
            max_concurrency=configured_inference.max_concurrency,
            # type: ignore[arg-type]
            decoder_implement=configured_inference.decoder_backend,
        )
        if configured_inference.enable_cuda_graph:
            brain.capture_decode_graph()
        inference_details = dict(configured_inference.as_dict())
        inference_details["container"] = execution_container
        inference_details["graph_batch_sizes"] = (
            brain.decode_graph_runner.stats()["configured_batch_sizes"]
        )
        record_event(
            "online_inference_config", **inference_details
        )
        base_node.tags.add(
            f"decoder_backend:{configured_inference.decoder_backend}"
        )
        if configured_inference.enable_cuda_graph:
            base_node.tags.add("cuda_graph")
        ctx.logger.info(
            f"online inference config: {inference_details}"
        )
        if time_trace:
            base_node.tags.add("time_trace")
            ctx.logger.warn("time tracing embedded")
            assert time_recorder is not None
            brain = AnalysisFrame(
                brain=brain,
                recorder=time_recorder
            )

        def _on_request_decode_graph_stats(node: str, ticket: int):
            stats = brain.query_extra_info("decode_graph")
            base_node.send_resp(node, ticket, stats)

        base_node.apis[API_DECODE_GRAPH_STATS] = (
            _on_request_decode_graph_stats
        )
        await dealer.ainit(
            brain=brain,  # type:ignore
            models=models,
        )

        p = psutil.Process()
        start_time = p.create_time()
        elapsed = time.time() - start_time
        ctx.logger.info(
            f"init in {time.time()-init_time}s, process time: {elapsed}s")
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
        inference_config: Optional[OnlineInferenceConfig] = None,
        expected_network_name: str = "",
        expected_network_spec_hash: str = "",
        execution_container: str = "shared",
    ):
        base_node, dealer, init_coro = dealer_coro(
            ctx, device_str, models, memory_config, backbone, time_trace,
            time_trace_with_sync_cuda, inference_config,
            expected_network_name, expected_network_spec_hash,
            execution_container,
        )
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
