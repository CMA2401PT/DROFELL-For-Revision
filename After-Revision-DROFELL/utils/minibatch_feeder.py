from typing import Callable, Generic, List, Optional, Tuple, TypeVar
import extend_asyncio as asyncio


MinibatchData = TypeVar("MinibatchData")


class MinibatchFeeder(Generic[MinibatchData]):
    def __init__(
            self,
            get_minibatch: Callable[[], Optional[MinibatchData]],
            batch_size: int,
            init_version: int = 0) -> None:
        self.minibatch_idx = 0
        self.batch_size = batch_size
        self.current_version = init_version
        self.current_batch_ids: List[str] = []
        self.current_batch: List[Tuple[int, str, MinibatchData]] = []
        self.get_sample = get_minibatch

    @property
    def _next_idx(self):
        minibatch_id = f"{self.minibatch_idx//self.batch_size}-{self.minibatch_idx}"
        self.minibatch_idx += 1
        return minibatch_id

    def _gen_batch(self):
        assert self.current_batch.__len__() == 0
        self.current_batch_ids = []
        while len(self.current_batch) < self.batch_size:
            minibatch_id = self._next_idx
            sample = self.get_sample()
            if sample is None:
                break
            self.current_batch.append((self.current_version, minibatch_id, sample))
            self.current_batch_ids.append(minibatch_id)
        if len(self.current_batch_ids) != 0:
            self.current_version += 1

    def get_next(self):
        if self.current_batch.__len__() == 0:
            self._gen_batch()
        if self.current_batch.__len__() == 0:
            return None
        minibatch = self.current_batch.pop(0)
        return self.current_batch_ids, *minibatch


class AsyncMinibatchFeeder(Generic[MinibatchData]):
    def __init__(self, get_sample: Callable[[], Optional[MinibatchData]], batch_size: int, max_concurrent: int) -> None:
        self.base_feeder = MinibatchFeeder(get_minibatch=get_sample, batch_size=batch_size)
        self.sample_queue: asyncio.Queue[Optional[Tuple[List[str], int, str, MinibatchData]]] = asyncio.Queue()
        self.ctrl: asyncio.Queue[None] = asyncio.Queue()
        for _ in range(max_concurrent):
            self.ctrl.put_nowait(None)

        async def feed_loop():
            while True:
                await self.ctrl.get()
                sample = self.base_feeder.get_next()
                self.sample_queue.put_nowait(sample)
        asyncio.create_task(feed_loop())

    async def get_next(self, wait: bool = True):
        if not wait and self.sample_queue.empty():
            return self.base_feeder.get_next()
        return await self.sample_queue.get()

    def consume_one(self):
        self.ctrl.put_nowait(None)
