import multiprocessing
import multiprocessing.connection
import time
import extend_asyncio as asyncio

asyncio.config_impl("py", False)
FrameConn = asyncio.FrameConn
Frames = asyncio.Frames


def reader_process(read_conn: multiprocessing.connection.Connection):
    print("reader start")

    async def async_reader():
        loop = asyncio.get_event_loop()
        pr = loop.pipe_reader(read_conn)

        while True:
            frames = await pr.recv()
            if frames is None:
                break
            print(f"[Reader] Got: {frames}")
        print("[Reader] EOF reached.")
    asyncio.run(async_reader())
    print("reader exit")


def writer_process(write_conn: multiprocessing.connection.Connection):
    print("writer start")

    async def async_writer():
        loop = asyncio.get_event_loop()
        pw = loop.pipe_writer(write_conn)
        for i in range(5):
            await asyncio.sleep(0.01)
            message = f"Message {i}"
            pw.send([message.encode("utf-8")], None)
        # pw.close()
        # os._exit(0)
        # parent_conn.close()
        print("writer close!")
    asyncio.run(async_writer())
    print("writer exit")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    child_conn, parent_conn = multiprocessing.Pipe()

    pr = multiprocessing.Process(target=reader_process, args=(child_conn,))
    pw = multiprocessing.Process(target=writer_process, args=(parent_conn,))

    # del child_conn
    pr.start()
    # writer_process(parent_conn)
    # del parent_conn
    pw.start()
    time.sleep(1)
    parent_conn.close()
    child_conn.close()
    pw.join()
    pr.join()
