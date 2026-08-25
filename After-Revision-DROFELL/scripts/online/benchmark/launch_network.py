"""Launch an online benchmark network from a logical JSON specification."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, Literal, cast

import extend_asyncio as asyncio
from orbit import Orbit, Satellite, SimpleLogger
from orbit.defines import Location
from orbit.launch_pad import LaunchPad
from orbit.link.cpn import NetModel, OrbitCPNCommander

from scripts.online.common_nodes import aux_node, dealer_node
from scripts.online.benchmark.network_nodes import agent_node, failure_dealer_node
from scripts.online.benchmark.network_spec import DealerSpec, NetworkSpec


_NO_GC: list[object] = []
AsyncBackend = Literal["py", "chimera", "golang"]


@dataclass(frozen=True)
class Placement:
    dealer_name: str
    process: int
    thread: int
    device: str


@dataclass(frozen=True)
class AgentPlacement:
    agent_name: str
    dealer_name: str


def initialize_process(
    async_backend: str,
    busy_wait: bool,
) -> asyncio.Loop:
    def on_exception(_: asyncio.Loop, error: Exception) -> None:
        raise error

    if async_backend not in {"py", "chimera", "golang"}:
        raise ValueError(f"invalid async backend: {async_backend!r}")
    asyncio.config_impl(
        cast(AsyncBackend, async_backend),
        busy_wait=busy_wait,
    )
    loop = asyncio.get_event_loop()
    loop.on_except(on_exception)
    return loop


def positive_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be in [1, 65535]")
    return port


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def placement(value: list[str]) -> Placement:
    name, process_text, thread_text, device = value
    try:
        process = int(process_text)
        thread = int(thread_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "placement PROCESS and THREAD must be integers"
        ) from error
    if process < 0 or thread < 0:
        raise argparse.ArgumentTypeError(
            "placement PROCESS and THREAD must be non-negative"
        )
    if not name or not device:
        raise argparse.ArgumentTypeError(
            "placement NAME and DEVICE must be non-empty"
        )
    return Placement(name, process, thread, device)


def agent_placement(value: list[str]) -> AgentPlacement:
    agent_name, dealer_name = value
    if not agent_name or not dealer_name:
        raise argparse.ArgumentTypeError(
            "agent NAME and DEALER must be non-empty"
        )
    return AgentPlacement(agent_name, dealer_name)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--async-backend",
        choices=("py", "chimera", "golang"),
        default="golang",
    )
    parser.add_argument(
        "--busy-wait",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--container",
        choices=("shared", "nonasync"),
        default="shared",
        help="Dealer CUDA execution container (default: shared).",
    )


def _add_placement_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--place",
        action="append",
        nargs=4,
        metavar=("DEALER", "PROCESS", "THREAD", "DEVICE"),
        required=True,
        help=(
            "Physical placement. Repeat once per launched dealer, e.g. "
            "--place dealer-a 0 0 cuda:0."
        ),
    )
    parser.add_argument(
        "--agent",
        action="append",
        nargs=2,
        metavar=("NAME", "DEALER"),
        default=[],
        help=(
            "Launch a persistent benchmark Agent in the same process and "
            "thread as DEALER. Repeat for multiple Agents."
        ),
    )
    parser.add_argument(
        "--agent-path-search-redundancy",
        type=positive_int,
        default=1,
        help=(
            "Provider-search redundancy for colocated persistent Agents; "
            "each segment considers at most this value plus one providers."
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a persistent online LLM network with optional colocated "
            "Agents; temporary Request nodes are started separately."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    local = commands.add_parser(
        "local", help="Launch Orbit, aux and every dealer on this host."
    )
    local.add_argument("--network", type=Path, required=True)
    local.add_argument("--port", type=positive_port, default=7241)
    local.add_argument("--advertise-host", default="")
    _add_runtime_arguments(local)
    _add_placement_arguments(local)

    orbit = commands.add_parser(
        "orbit", help="Launch only Orbit and aux for a distributed network."
    )
    orbit.add_argument("--network", type=Path, required=True)
    orbit.add_argument("--port", type=positive_port, default=7241)
    orbit.add_argument("--advertise-host", default="")
    _add_runtime_arguments(orbit)

    satellite = commands.add_parser(
        "satellite", help="Attach this host and launch a subset of dealers."
    )
    satellite.add_argument("--network", type=Path, required=True)
    satellite.add_argument("--orbit", required=True, metavar="HOST:PORT")
    satellite.add_argument("--host-id", required=True)
    satellite.add_argument("--advertise-host", default="")
    _add_runtime_arguments(satellite)
    _add_placement_arguments(satellite)

    args = parser.parse_args()
    if hasattr(args, "place"):
        try:
            args.place = [placement(item) for item in args.place]
            args.agent = [agent_placement(item) for item in args.agent]
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
    return args


def _validate_placements(
    spec: NetworkSpec,
    placements: Iterable[Placement],
    require_all: bool,
) -> tuple[Placement, ...]:
    result = tuple(placements)
    names = [item.dealer_name for item in result]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(
            f"dealers placed more than once in this command: {duplicate_names}"
        )
    unknown = sorted(set(names) - set(spec.dealer_names))
    if unknown:
        raise ValueError(f"placements reference unknown dealers: {unknown}")
    if require_all and set(names) != set(spec.dealer_names):
        missing = sorted(set(spec.dealer_names) - set(names))
        raise ValueError(f"local launch is missing dealer placements: {missing}")
    return result


def _dealer_function(dealer: DealerSpec):
    return failure_dealer_node if dealer.failure_injection else dealer_node


def _validate_agents(
    agents: Iterable[AgentPlacement],
    placements: Iterable[Placement],
    reserved_node_names: Iterable[str] = tuple(),
) -> tuple[AgentPlacement, ...]:
    result = tuple(agents)
    names = [item.agent_name for item in result]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"Agents named more than once: {duplicate_names}")
    dealer_names = {item.dealer_name for item in placements}
    missing = sorted(
        item.dealer_name
        for item in result
        if item.dealer_name not in dealer_names
    )
    if missing:
        raise ValueError(
            "Agent placements reference Dealers absent from this command: "
            f"{missing}"
        )
    reserved = dealer_names | set(reserved_node_names) | {"aux", "Orbit"}
    conflicts = sorted(set(names) & reserved)
    if conflicts:
        raise ValueError(f"Agent names conflict with network nodes: {conflicts}")
    return result


def _dealer_arguments(
    spec: NetworkSpec,
    dealer: DealerSpec,
    execution_container: str,
) -> dict:
    return {
        "device_str": None,
        "models": spec.model_targets(dealer),
        "memory_config": dealer.memory.make_runtime_config(),
        "time_trace": dealer.time_trace,
        "time_trace_with_sync_cuda": dealer.time_trace_sync_cuda,
        "inference_config": dealer.inference_config(),
        "expected_network_name": spec.name,
        "expected_network_spec_hash": spec.spec_hash(),
        "execution_container": execution_container,
    }


def _launch_pad(args: argparse.Namespace) -> LaunchPad:
    return LaunchPad(partial(initialize_process, args.async_backend, args.busy_wait))


async def launch_local_network(
    *,
    spec: NetworkSpec,
    placements: Iterable[Placement],
    launch_pad: LaunchPad,
    port: int,
    advertise_host: str,
    agents: Iterable[AgentPlacement] = tuple(),
    agent_path_search_redundancy: int = 1,
    execution_container: str = "shared",
) -> Orbit:
    resolved_placements = _validate_placements(
        spec,
        placements,
        require_all=True,
    )
    resolved_agents = _validate_agents(
        agents,
        resolved_placements,
        spec.dealer_names,
    )
    functions = {"aux": aux_node}
    for dealer in spec.dealers:
        functions[f"dealer:{dealer.name}"] = _dealer_function(dealer)
    if resolved_agents:
        functions["agent"] = agent_node
    orbit = Orbit(
        launch_pad=launch_pad,
        functions_dict=functions,
        logger=SimpleLogger(),
        network_hostname=advertise_host,
        log_level="info",
    )
    commander = OrbitCPNCommander(orbit, NetModel())
    _NO_GC.append(commander)
    await orbit.init(port=port, satellites=tuple())
    await orbit.launch(
        Location(name="aux"),
        fn_name="aux",
        runtime_extra_info=spec.extra_info(),
    )

    dealers = {dealer.name: dealer for dealer in spec.dealers}
    tasks = []
    for item in resolved_placements:
        dealer = dealers[item.dealer_name]
        kwargs = _dealer_arguments(spec, dealer, execution_container)
        kwargs["device_str"] = item.device
        tasks.append(
            asyncio.create_task(
                orbit.launch(
                    Location(
                        name=dealer.name,
                        process=item.process,
                        thread=item.thread,
                    ),
                    fn_name=f"dealer:{dealer.name}",
                    **kwargs,
                )
            )
        )
    await asyncio.gather(*tasks)
    placement_by_dealer = {
        item.dealer_name: item for item in resolved_placements
    }
    for agent in resolved_agents:
        dealer_placement = placement_by_dealer[agent.dealer_name]
        await orbit.launch(
            Location(
                name=agent.agent_name,
                process=dealer_placement.process,
                thread=dealer_placement.thread,
            ),
            fn_name="agent",
            path_search_redundancy=agent_path_search_redundancy,
        )
    return orbit


def _run_local(args: argparse.Namespace, spec: NetworkSpec) -> None:
    launch_pad = _launch_pad(args)

    async def start() -> None:
        await launch_local_network(
            spec=spec,
            placements=args.place,
            launch_pad=launch_pad,
            port=args.port,
            advertise_host=args.advertise_host,
            agents=args.agent,
            agent_path_search_redundancy=args.agent_path_search_redundancy,
            execution_container=args.container,
        )
        print(
            f"network {spec.name!r} ready at "
            f"{args.advertise_host or 'localhost'}:{args.port}"
        )
        await asyncio.Future()

    launch_pad.loop.create_task(start())
    launch_pad.loop.run_forever()


def _run_orbit(args: argparse.Namespace, spec: NetworkSpec) -> None:
    launch_pad = _launch_pad(args)

    async def start() -> None:
        orbit = Orbit(
            launch_pad=launch_pad,
            functions_dict={"aux": aux_node},
            logger=SimpleLogger(),
            network_hostname=args.advertise_host,
            log_level="info",
        )
        commander = OrbitCPNCommander(orbit, NetModel())
        _NO_GC.append(commander)
        await orbit.init(port=args.port, satellites=tuple())
        await orbit.launch(
            Location(name="aux"),
            fn_name="aux",
            runtime_extra_info=spec.extra_info(),
        )
        print(
            f"network coordinator {spec.name!r} ready at "
            f"{args.advertise_host or 'localhost'}:{args.port}"
        )
        await asyncio.Future()

    launch_pad.loop.create_task(start())
    launch_pad.loop.run_forever()


def _run_satellite(args: argparse.Namespace, spec: NetworkSpec) -> None:
    placements = _validate_placements(spec, args.place, require_all=False)
    agents = _validate_agents(args.agent, placements, spec.dealer_names)
    launch_pad = _launch_pad(args)

    async def start() -> None:
        satellite = Satellite(
            launch_pad=launch_pad,
            functions_dict={},
            host_id=args.host_id,
            network_hostname=args.advertise_host,
        )
        satellite.start(orbit_addr=args.orbit)
        await satellite.ready()
        dealers = {dealer.name: dealer for dealer in spec.dealers}
        for item in placements:
            dealer = dealers[item.dealer_name]
            kwargs = _dealer_arguments(spec, dealer, args.container)
            kwargs["device_str"] = item.device
            satellite.launch(
                target_fn=_dealer_function(dealer),
                proc=item.process,
                thread=item.thread,
                node_name=dealer.name,
                **kwargs,
            )
        placement_by_dealer = {
            item.dealer_name: item for item in placements
        }
        for agent in agents:
            dealer_placement = placement_by_dealer[agent.dealer_name]
            satellite.launch(
                target_fn=agent_node,
                proc=dealer_placement.process,
                thread=dealer_placement.thread,
                node_name=agent.agent_name,
                path_search_redundancy=args.agent_path_search_redundancy,
            )
        print(
            f"satellite {args.host_id!r} attached to {args.orbit}; "
            f"launched {[item.dealer_name for item in placements]} and "
            f"Agents {[item.agent_name for item in agents]}"
        )
        await asyncio.Future()

    launch_pad.loop.create_task(start())
    launch_pad.loop.run_forever()


def main() -> None:
    args = parse_args()
    spec = NetworkSpec.load(args.network)
    print(f"network spec: {spec.name} ({spec.spec_hash()})")
    if args.command == "local":
        _run_local(args, spec)
    elif args.command == "orbit":
        _run_orbit(args, spec)
    elif args.command == "satellite":
        _run_satellite(args, spec)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
