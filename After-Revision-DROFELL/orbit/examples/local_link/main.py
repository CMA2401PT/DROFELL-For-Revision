import os
from typing import Any, Callable, Dict

import extend_asyncio as asyncio

from orbit import Context, LaunchPad, Location, Orbit, OrbitName, SimpleLogger
from orbit.link import ContextNetExtender, OrbitCPNCommander
from orbit.link.cpn import NetModel


def receiver_node(ctx: Context) -> None:
    async def run() -> None:
        ContextNetExtender(ctx)
        ctx.report_ready()
        source, api, payload = await receive_api(ctx, "identity")
        assert source == "sender"
        assert api == "identity"
        ctx.send((source, "identity", payload))

    asyncio.create_task(run())


def sender_node(ctx: Context) -> None:
    async def run() -> None:
        ContextNetExtender(ctx)
        ctx.report_ready()
        assert await ctx.try_connect("receiver")

        callback = lambda: "not pickleable"
        payload: Dict[str, Any] = {"callback": callback}
        ctx.send(("receiver", "identity", payload))

        source, api, returned = await receive_api(ctx, "identity")
        assert source == "receiver"
        assert api == "identity"
        assert returned is payload
        assert returned["callback"] is callback
        assert returned["callback"]() == "not pickleable"
        ctx.send((OrbitName, "local-link-test-passed", None))

    asyncio.create_task(run())


FUNCTIONS: Dict[str, Callable[[Context], None]] = {
    "receiver": receiver_node,
    "sender": sender_node,
}


async def receive_api(ctx: Context, expected_api: str):
    while True:
        packet = await ctx.recv()
        if packet[1] == expected_api:
            return packet


def initialize_process() -> asyncio.Loop:
    def on_exception(_: asyncio.Loop, error: Exception) -> None:
        raise error

    asyncio.config_impl("golang", False)
    loop = asyncio.get_event_loop()
    loop.on_except(on_exception)
    return loop


def main() -> None:
    launch_pad = LaunchPad(initialize_process)

    async def run() -> None:
        orbit = Orbit(
            launch_pad=launch_pad,
            functions_dict=FUNCTIONS,
            logger=SimpleLogger(),
        )
        commander = OrbitCPNCommander(orbit, NetModel())
        orbit.append_no_gc(commander)
        await orbit.init(port=0, satellites=tuple())
        await orbit.launch(Location(name="receiver"), "receiver")
        await orbit.launch(Location(name="sender"), "sender")
        source, api, data = await orbit.recv()
        assert source == "sender"
        assert api == "local-link-test-passed"
        assert data is None
        print(
            "local link bypassed serialization and preserved object identity",
            flush=True,
        )
        os._exit(0)

    launch_pad.loop.create_task(run())
    launch_pad.loop.run_forever()


if __name__ == "__main__":
    main()
