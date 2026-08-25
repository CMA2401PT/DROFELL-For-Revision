import os
import extend_asyncio as asyncio
from orbit import *
from orbit.satellite import SimpleSatellite


def init_fn():
    def on_except(_: asyncio.Loop, e: Exception):
        raise e
        os._exit(-1)
    asyncio.config_impl("golang", True)
    loop = asyncio.get_event_loop()
    loop.on_except(on_except)
    return loop


if __name__ == "__main__":
    launch_pad = LaunchPad(init_fn)
    s = SimpleSatellite(launch_pad=launch_pad, functions_dict={}, host_id="satellite")

    async def amain():
        print("ping: ", await s.ping("localhost:2401"))

    launch_pad.loop.create_task(amain())
    launch_pad.loop.run_forever()
