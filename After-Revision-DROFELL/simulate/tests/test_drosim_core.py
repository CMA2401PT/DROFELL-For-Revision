#!/usr/bin/env python3
"""Regression tests for the DROSIM core (clock / fabric / context / brain).

These are the CP1 checks from docs/05 that do not need a full topology. Run from
the repository root:

    /home/dai/projects/LLM/dflash_related/.venv/bin/python \
        simulate/tests/test_drosim_core.py

Each test prints PASS/FAIL and the script exits non-zero if anything fails, so it
can be used as a smoke gate before longer runs.
"""

from __future__ import annotations

import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    from simulate.drosim.clock import VirtualClock, install, run_simulation

    clock = VirtualClock()
    loop = install(clock)

    import extend_asyncio as asyncio
    import torch

    from dealers.common.base_types import DeviceTensorCarrier
    from dealers.common.brain import DecodeGeneratingResult, OperateDoGenerating
    from node.base import BaseNode
    from simulate.drosim.brain import DeviceState, SimBrain
    from simulate.drosim.context import SimContext
    from simulate.drosim.cost import CostModel, ServiceTerm, TransferTerm
    from simulate.drosim.fabric import (
        LOOPBACK, WAN, HostCpuCost, Link, SimFabric, wire_bytes,
    )

    def schedule(when_ns: int, callback) -> None:
        loop.call_at(when_ns / 1e9, callback)

    def carrier() -> DeviceTensorCarrier:
        return DeviceTensorCarrier(
            torch.empty((1, 4096), device="meta", dtype=torch.float16), None
        )

    # ---- 1. virtual clock -------------------------------------------------
    print("\n1. virtual clock")

    async def clock_scenario():
        marks = []
        async def worker(tag, delays):
            for delay in delays:
                await asyncio.sleep(delay)
                marks.append((round(loop.time(), 6), tag))
        await asyncio.gather(
            worker("a", [1.0, 10.0]),
            worker("b", [0.5, 3600.0]),
        )
        return marks

    wall_started = time.monotonic()
    marks = run_simulation(loop, clock_scenario())
    wall_elapsed = time.monotonic() - wall_started
    check("advances to 3600.5 virtual seconds", abs(loop.time() - 3600.5) < 1e-6,
          f"t={loop.time():.3f}s")
    check("costs almost no real time", wall_elapsed < 1.0,
          f"{wall_elapsed * 1000:.2f} ms wall")
    check("preserves causal order", marks == sorted(marks), f"{marks}")

    async def background_failure_scenario():
        async def fail():
            await asyncio.postpone()
            raise ValueError("background failure sentinel")

        asyncio.create_task(fail())
        await asyncio.sleep(1.0)

    surfaced = False
    try:
        run_simulation(loop, background_failure_scenario())
    except ValueError as error:
        surfaced = "background failure sentinel" in str(error)
    check("background task failures reach the root immediately", surfaced)

    async def deadlock_scenario():
        await asyncio.Future()

    deadlock_surfaced = False
    try:
        run_simulation(loop, deadlock_scenario())
    except RuntimeError as error:
        deadlock_surfaced = "DROSIM deadlock" in str(error)
    check("an empty virtual scheduler fails instead of spinning", deadlock_surfaced)

    # ---- 2. fabric latency and bandwidth ---------------------------------
    print("\n2. fabric: latency, bandwidth, per-link serialisation")
    fabric = SimFabric(clock, schedule, default_link=WAN)
    ctx_a = SimContext("node-a", fabric, clock, host="h0")
    ctx_b = SimContext("node-b", fabric, clock, host="h1")
    ctx_c = SimContext("node-c", fabric, clock, host="h0")
    node_a, node_b = BaseNode(ctx_a), BaseNode(ctx_b)
    node_b.tags.add("dealer")

    async def pump(ctx, node):
        while True:
            node.on_packet(await ctx.recv())

    async def rpc_scenario():
        asyncio.create_task(pump(ctx_a, node_a))
        asyncio.create_task(pump(ctx_b, node_b))
        started = clock.now_ns
        tags = await node_a.req_tags("node-b")
        return tags, (clock.now_ns - started) / 1e6

    tags, rtt_ms = run_simulation(loop, rpc_scenario())
    check("real BaseNode RPC works over SimFabric", tags == {"dealer"}, f"{tags}")
    check("cross-host RTT is 2x link latency", 99.0 < rtt_ms < 102.0,
          f"{rtt_ms:.2f} ms for 2x50ms")
    check("same-host link is loopback",
          fabric.link("node-a", "node-c") is LOOPBACK)
    check("cross-host link is the default (tc) link",
          fabric.link("node-a", "node-b") is WAN)
    check("wire bytes are counted", fabric.stats.bytes_total > 0,
          f"{fabric.stats.bytes_total} bytes / {fabric.stats.packets} packets")

    meta_payload = DeviceTensorCarrier(
        torch.empty((1, 4096), device="meta", dtype=torch.float16), None
    )
    check("meta tensor wire size includes logical storage",
          wire_bytes(meta_payload) >= 4096 * 2,
          f"{wire_bytes(meta_payload)} bytes for an 8192-byte tensor")

    forced = SimFabric(
        clock, schedule, default_link=WAN, force_default_link=True
    )
    SimContext("forced-a", forced, clock, host="same-host")
    SimContext("forced-b", forced, clock, host="same-host")
    check("explicit shaping overrides same-host loopback",
          forced.link("forced-a", "forced-b") is WAN,
          forced.report()["link_policy"])

    shared_link = Link(latency_ms=0.0, bandwidth_mbit=8.0)
    shared = SimFabric(
        clock, schedule, default_link=shared_link,
        force_default_link=True, share_host_interfaces=True,
        host_cpu_cost=HostCpuCost(
            cores=1, serialize_fixed_us=100.0,
            deserialize_fixed_us=100.0,
        ),
    )
    SimContext(
        "shared-a", shared, clock, host="shared-host",
        cpu_domain="shared-process",
    )
    SimContext(
        "shared-b", shared, clock, host="shared-host",
        cpu_domain="shared-process",
    )
    SimContext(
        "shared-c", shared, clock, host="shared-host",
        cpu_domain="shared-process",
    )
    shared.send("shared-a", "shared-c", "payload", b"x" * 1000)
    shared.send("shared-b", "shared-c", "payload", b"x" * 1000)
    check("logical peers share their host interface",
          shared.stats.queue_delay_ns > 0,
          f"{shared.stats.queue_delay_ns / 1e6:.3f} ms queued")
    check("logical peers share finite host CPU serialization",
          sum(shared.stats.host_cpu_queue_ns.values()) > 0,
          str(shared.report()["host_cpu_queue_ms"]))

    async def flush_shared_packets():
        await asyncio.sleep(0.01)

    run_simulation(loop, flush_shared_packets())

    big = fabric.link("node-a", "node-b").transmit_ns(100 * 1024 * 1024 // 8)
    check("bandwidth model: 12.5 MB over 100 Mbit ~ 1 s",
          0.9e9 < big < 1.1e9, f"{big / 1e9:.3f} s")

    # ---- 3. unreachable peers --------------------------------------------
    print("\n3. fabric: reachability and node drop")
    check("known peer is connectable",
          run_simulation(loop, ctx_a.try_connect("node-b")))
    ctx_b.drop()
    check("dropped node becomes unreachable",
          not run_simulation(loop, ctx_a.try_connect("node-b")))

    # ---- 4. SimBrain merging --------------------------------------------
    print("\n4. SimBrain: operate queues, merging, device serialisation")
    cost = CostModel.measured()
    device = DeviceState("cuda:0", 24 * (1 << 30))

    async def brain_scenario():
        brain = SimBrain("d0", clock, cost, device, max_concurrency=8)

        async def submit(binding: str, count: int):
            operates = []
            for index in range(count):
                operate = OperateDoGenerating(
                    binding=binding, extra={}, fut=asyncio.Future(), sync=None,
                    context_id=f"ctx-{index}", cpu_in=False, cpu_out=False,
                    begin_metric=None, finish_metric=None,
                    carrier=carrier(), head_task=None, tail_task=None,
                )
                operate._drosim_result = lambda: DecodeGeneratingResult(
                    carrier=carrier())
                brain.operates_do_generating.append(operate)
                operates.append(operate)
            return [await item.fut for item in operates]

        started = clock.now_ns
        await submit("bindingA", 8)
        merged_ms = (clock.now_ns - started) / 1e6

        started = clock.now_ns
        for index in range(8):
            await submit(f"binding-{index}", 1)
        serial_ms = (clock.now_ns - started) / 1e6

        brain.close()
        return brain, merged_ms, serial_ms

    brain, merged_ms, serial_ms = run_simulation(loop, brain_scenario())
    check("8 same-binding operates merge into one device operation",
          1.5 < merged_ms < 2.5, f"{merged_ms:.2f} ms")
    check("8 distinct bindings serialise on the device",
          serial_ms > 6 * merged_ms, f"{serial_ms:.2f} ms")
    check("merge advantage matches the measured flat service time",
          7.0 < serial_ms / merged_ms < 9.0, f"{serial_ms / merged_ms:.1f}x")
    check("device busy time is accounted", brain.stats.device_busy_ns > 0,
          f"{brain.stats.device_busy_ns / 1e6:.2f} ms")

    # ---- 5. unmodelled paths raise --------------------------------------
    print("\n5. SimBrain: unmodelled paths fail loudly")
    raised_queue = False
    try:
        _ = brain.operates_forward
    except NotImplementedError:
        raised_queue = True
    check("train-path queue raises NotImplementedError", raised_queue)

    raised_attr = False
    try:
        _ = brain.some_unmodelled_thing
    except AttributeError:
        raised_attr = True
    check("unknown attribute raises AttributeError", raised_attr)

    # ---- 6. cost model provenance ---------------------------------------
    print("\n6. cost model: calibration and extrapolation warning")
    term = cost.term("decoder_generating")
    # p12 fitted alpha' = 1918.8 us on 8-LAYER shards, so that constant bundled
    # eight layers of work into the intercept. p16 separated the two using the
    # 1/2/4/8-segment sweep, so the model is now alpha + beta*n + gamma*layers.
    # The check is therefore that the DECOMPOSITION reproduces the original
    # measurement at the topology it was measured on -- not that alpha still
    # equals the old bundled number.
    check("per-layer term was separated from per-invocation term",
          term.gamma_us_per_layer > 0.0,
          f"alpha={term.alpha_us:.1f}us gamma={term.gamma_us_per_layer:.1f}us/layer")
    check("reproduces the p12 measurement at 8 layers",
          abs(term.duration_us(batch=1, layers=8) - 1918.8) < 5.0,
          f"{term.duration_us(batch=1, layers=8):.1f}us vs measured 1918.8us")
    check("beta is negligible against alpha",
          abs(term.beta_us_per_item) < 5.0)
    interaction = ServiceTerm(100.0, 2.0, 3.0, 4.0)
    check("batch-by-layer interaction is represented explicitly",
          interaction.duration_us(batch=2, layers=5) == 159.0,
          f"{interaction.duration_us(batch=2, layers=5):.1f}us")
    transfer = TransferTerm(
        fixed_us=10.0, per_tensor_us=2.0, bandwidth_gbps=8.0
    )
    check("host/device transfer separates group, tensor and byte terms",
          abs(transfer.duration_us(1000, transfers=2) - 15.0) < 1e-9,
          f"{transfer.duration_us(1000, transfers=2):.1f}us")
    # And the segment-count axis must actually be expressed: splitting a fixed
    # model over more segments has to cost more in total, or the simulator could
    # not answer 2.7 at all.
    total_1 = 1 * term.duration_us(batch=1, layers=32)
    total_8 = 8 * term.duration_us(batch=1, layers=4)
    check("splitting a fixed model over more segments costs more",
          total_8 > total_1,
          f"L=1: {total_1 / 1000:.2f}ms vs L=8: {total_8 / 1000:.2f}ms")
    before = dict(cost.extrapolations)
    cost.duration_ns("decoder_generating", batch=64)
    check("batch beyond the calibrated range is flagged",
          cost.extrapolations != before, f"{cost.extrapolations}")
    before = dict(cost.extrapolations)
    cost.duration_ns("decoder_generating", batch=1, layers=999)
    check("layer count beyond the calibrated range is flagged",
          cost.extrapolations != before, f"{cost.extrapolations}")

    # ---- 7. real memory twin --------------------------------------------
    print("\n7. real memory twin attached, zero CUDA allocation")
    from dealers.real_basic.memory import MemoryBrain, MemoryConfigBasic
    import split_qllm_models

    lib = os.path.dirname(split_qllm_models.__file__)
    twin_config = MemoryConfigBasic()
    twin_config.root_memory = 24 * (1 << 30)
    twin = MemoryBrain(
        models_path=lib, basic_memory_config=twin_config, max_seq_len=2048,
        decoder_implement="marlin", device=None,
    )
    cuda_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    model = "teared-Llama-2-7B-GPTQ"
    twin.make_model_head(model)
    twin.make_model_decoders(model, (0, 8))
    twin.make_model_tail(model)
    cuda_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    check("twin models real shards", twin.total > 1 << 30,
          f"{twin.total / 2**30:.3f} GiB")
    check("twin allocates zero CUDA bytes", cuda_after == cuda_before,
          f"{cuda_after - cuda_before} bytes")

    # ---- 8. simplified path selection ------------------------------------
    print("\n8. routing: node-disjoint, linear in segment count")
    from simulate.drosim.routing import logical_nodes, select_paths

    def topology(segments: int, redundancy: int) -> list[list[str]]:
        return [
            [f"d-g{group}-r{index}" for index in range(redundancy)]
            for group in range(segments)
        ]

    selection = select_paths(topology(4, 2), redundancy=2)
    check("yields the requested disjoint paths", selection.achieved == 2)
    check("paths share no node",
          not (set(selection.paths[0]) & set(selection.paths[1])))
    check("keypoints cover every segment of every path",
          len(selection.keypoints) == 8, f"{len(selection.keypoints)}")

    single = select_paths([["dealer-g0"]] * 3, redundancy=1)
    check("degenerate single-provider topology works",
          single.achieved == 1 and len(single.keypoints) == 1)

    over = select_paths(topology(4, 2), redundancy=5)
    check("over-request is reported rather than silently wrong",
          over.achieved < over.requested and over.shortfall_reason is not None,
          f"got {over.achieved}/{over.requested}")

    started = time.perf_counter_ns()
    wide = select_paths(topology(256, 2), redundancy=2)
    wide_ms = (time.perf_counter_ns() - started) / 1e6
    check("scales to 256 segments where the prototype DP OOMs",
          wide.achieved == 2 and wide_ms < 50.0,
          f"{logical_nodes(topology(256, 2))} nodes in {wide_ms:.2f} ms")

    # ---- 9. clock shim: the mixed-clock deadlock ------------------------
    # Regression for the bug that made the first end-to-end run hang: a module
    # that did `from time import time_ns` kept the REAL function, so
    # `node/infer/sender.py` compared a virtual `now` against a wall-clock
    # deadline and its packet-grouping trap could never expire.
    print("\n9. time shim reaches module-level aliases, not just `time.time_ns`")
    import time as _time_module

    from node.infer import sender as _sender

    check("time.time_ns is virtual under simulation",
          _time_module.time_ns() == clock.now_ns,
          f"{_time_module.time_ns()} vs clock {clock.now_ns}")
    check("`from time import time_ns` alias is virtual too",
          _sender.time_ns() == clock.now_ns,
          f"{_sender.time_ns()} vs clock {clock.now_ns}")
    check("perf_counter_ns is deliberately NOT virtualised",
          _time_module.perf_counter_ns() > clock.now_ns,
          "shadow CPU cost must stay a real measurement (docs/03 §5)")

    # ---- 10. sub-nanosecond timer deadlines ----------------------------
    # Regression for the infinite spin: CPython timer deadlines are floats, so
    # `_when` routinely carries a sub-nanosecond tail. Truncating it left the
    # clock a hair BELOW the deadline, the timer never became ready, and the loop
    # advanced to the same instant forever (measured 7.5M iterations).
    print("\n10. virtual clock always reaches a fractional timer deadline")

    async def fractional_deadline():
        marks = []
        # 0.010116693999999999-style values: not representable as whole ns.
        for delay in (0.010116693999999999, 1e-9, 3.3333333333e-6):
            await asyncio.sleep(delay)
            marks.append(clock.now_ns)
        return marks

    fractional_started = time.monotonic()
    marks = run_simulation(loop, fractional_deadline())
    check("a sub-nanosecond deadline still fires",
          len(marks) == 3, f"{marks}")
    check("and it does so promptly in real time",
          time.monotonic() - fractional_started < 5.0,
          f"{time.monotonic() - fractional_started:.3f}s wall")
    check("virtual time advanced strictly",
          marks == sorted(marks) and marks[0] < marks[-1], f"{marks}")

    # ---- 11. topology states its placement ------------------------------
    print("\n11. topology distinguishes dedicated from shared devices")
    from simulate.drosim.topology import Placement, build

    dedicated = build("teared-Llama-2-7B-GPTQ", 32, 4096, segments=4)
    shared = build(
        "teared-Llama-2-7B-GPTQ", 32, 4096, segments=8,
        placement=Placement.SHARED, device_count=2,
    )
    check("segments match the measured network spec",
          dedicated.segment_layers == [8, 8, 8, 8],
          f"{dedicated.segment_layers}")
    check("head and tail sit with the first/last segment",
          dedicated.dealers[0].is_head_holder
          and dedicated.dealers[-1].is_tail_holder)
    check("dedicated placement gives one device per node",
          dedicated.report()["nodes_per_device"] == 1.0)
    check("shared placement packs nodes onto fewer devices",
          shared.report()["nodes_per_device"] == 4.0,
          f"{shared.report()['nodes_per_device']}")
    check("every result states which placement produced it",
          "placement" in dedicated.report()
          and "placement_meaning" in dedicated.report())
    uneven = False
    try:
        build("teared-Llama-2-7B-GPTQ", 32, 4096, segments=5)
    except ValueError:
        uneven = True
    check("an uneven split is refused rather than silently rounded", uneven)

    # ---- 12. carrier row counting ---------------------------------------
    # Regression: a single-token carrier is 1-D `(hidden,)`, so reading shape[0]
    # returned 4096 and charged prefill for 4096 tokens.
    print("\n12. carrier row count is not the hidden size")
    from simulate.drosim.backend import SimBackend

    one_row = DeviceTensorCarrier(
        torch.empty((4096,), device="meta", dtype=torch.float16), None
    )
    many_rows = DeviceTensorCarrier(
        torch.empty((7, 4096), device="meta", dtype=torch.float16), None
    )
    check("1-D carrier counts as one token row",
          SimBackend._rows(one_row) == 1, f"{SimBackend._rows(one_row)}")
    check("2-D carrier reports its real row count",
          SimBackend._rows(many_rows) == 7, f"{SimBackend._rows(many_rows)}")

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {FAILURES}")
        return 1
    print("all core checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
