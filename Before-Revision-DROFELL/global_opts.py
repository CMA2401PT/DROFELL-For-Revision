import os
import sys
import time
from typing import Literal

project_dir = os.path.dirname(__file__)
sys.path.append(project_dir)
output_dir = os.path.join(project_dir, "outputs")
data_dir = os.path.join(project_dir, "data")
cache_dir = os.path.join(project_dir, "cache")
models_dir = os.path.join(project_dir, "split_qllm_models")
trained_models_dir = os.path.join(data_dir, "trained_model")
start_time = time.time_ns()

# cuda sync debug mode: torch.cuda.set_sync_debug_mode
CUDA_SYNC_DEBUG_MODE: int = 0  # 0: no-op 1: warn 2: error

# print generating/prefill tasks selected by cerebullem
PRINT_GENERATING_SELECTION = False
PRINT_PREFILL_SELECTION = False

# print adviser node suggestion for infer/train task
PRINT_ADVISER_SUGGESTION = True

# print base model load stage
PRINT_MODEL_LOAD = False

# decide how may cache blocks can cache engine holds
CACHE_ENGINE_CACHE_BLOCKS_NUM = 16

# for each TRAIN_RETRIEVE_PERIOD steps, do user model weight retrieve
TRAIN_RETRIEVE_PERIOD = 50

# whether retrieve infer block in inference
INFER_RETRIEVE = False
# when a block is full, retrieve will postpone INFER_RETRIEVE_POSTPONE generating steps
INFER_RETRIEVE_POSTPONE = 8
# postpone block retrieve, then do batch block retrieve after MERGE_BLOCK_RETRIEVE are full
MERGE_BLOCK_RETRIEVE = 0

# fast pickle tensor lazy load
FAST_TENSOR_PICKLE_LAZY_LOAD = True

# force infer/train api use multi thread/asyncio
# -> (True, True) force thread
# -> (True, False) force asyncio
# -> (False,...) not force
FORCE_INFER_CEREBELLUM_BLOCKING_INFER = (True, False)
FORCE_INFER_CEREBELLUM_BLOCKING_TRAIN = (True, True)
# if use thread for api, whether print warning when an api consumes too long time
PRINT_CHECK_CEREBELLUM_BLOCKING = False

# infer intercept window time
INTERCEPT_WINDOW_TIME = 0.01

# max infer intercept size
MAX_INTERCEPT_SIZE = 8

# whether to use rectifier
DISABLE_RECTIFIER = False

# wrap endpoint for analysis
ENDPOINT_OVERRIDE = None

# wrap memory alloactor for analysis
MEMORY_ALLOCATOR_OVERRIDE = None

# allow exllama cache share between base model decoders groups
ALLOW_CACHE_SHARE_BETWEEN_GROUPS = True

# in train backward, reuse reconstruct cache in forward(checkpoint) and backward in backward
REUSE_RECONSTRUCT_CACHE_WHEN_GRAD = False

# before training, dequantlize weight to float16/bfloat16,
# this will cause significant memory use and model is actually run in float16/bfloa16
# only for debug
RECONSTRUCTE_WEIGTH_FLOAT16 = False
RECONSTRUCTE_WEIGTH_BFLOAT16 = False

# in train, decide the policy of forwad and backward
USER_MODEL_TRAIN_METHOD = "seed"  # "grad" "seed"
DEOCDERS_TRAIN_METHOD = "first.grad"  # "grad"  "each_decoder" "first.grad" "first.2pass"

# whether to use multi cuda stream to parallel handle multi prefill in inference
MULTI_PREFILL_STREAM = False

# override max sequence length in rope config
MAX_TOKEN_LEN_OVERRIDE = 128*1024

# if TRUE, do not use readonly state in infer blocks, just cpu/gpu state
NO_RO_STATE = False
# if False, do not pre swap block when it is in readonly state in inference
SCALABLE_CACHE_ENGINE_PRE_SWAP = False

# if False, when modeling cache engine, batch cache blocks is not considered,
# also, when free blocks, reclaim blocks are not considered
MODELING_CACHE_ENGINE_CACHE = False

# if False, return as precise memory usage as possible
# otherwise some memory is pre allocated in modeling (e.g. grad memory)
MODELING_WITH_PRE_ALLOCATED = True

# infer block size (BLOCK_SIZE slot in one block)
BLOCK_SIZE = 32

# layers of decoder in decoders group
DECODER_GROUP_SIZE = 8
COMPONENT_TYPE_DECODER_GROUP_TYPE = Literal[8]
# even if total decoder groups cannot fully devided by DECODER_GROUP_SIZE, still continue
DECODER_GROUP_SIZE_ROUND_UP = True


# whether to print infer progress in infer anchor or not
PRINT_INFER_PROGRESS_IN_ANCHOR = False

# emulate anchor sampler
EMULATE_ANCHOR_SAMPLER = False

SIMULATE_NET_OVERHEAD = False
SIMULATE_NET_DELAY = 3
SIMULATE_NET_BANDWIDTH = 100  # M

# def _record_transmit(src: str, dst: str, api: str, type: str, size: List[int]):
#     now_time = time.time_ns()
#     now_time -= start_time
#     s = f"{now_time/(10**6):8.1f}ms {src}-{type}->{dst} {api}: {' '.join([f'{d/(2**10):.4f}kb' for d in size])}\n"
#     transmit_record_fp.write(s)


# record_transmit = _record_transmit
# record_transmit = None


# class _EventHandler(ABC):
#     @abstractmethod
#     def mark_event_happen(self, hint: str) -> Callable[[], None]:
#         raise NotImplementedError

#     @contextmanager
#     def mark_event(self, hint: str):
#         callback = self.mark_event_happen(hint)
#         yield
#         callback()


# class NOPHandler(_EventHandler):
#     def mark_event_happen(self, hint: str) -> Callable[[], None]:
#         return self._mark_event_terminated

#     def _mark_event_terminated(self) -> None:
#         pass


# class PrintOnlyHandler(_EventHandler):
#     def mark_event_happen(self, hint: str) -> Callable[[], None]:
#         print(f"{hint} start")
#         st = time.time_ns()
#         return lambda: self._mark_event_terminated(hint, st)

#     def _mark_event_terminated(self, hint: str, start_time: int) -> None:
#         print(f"{hint} finished {(time.time_ns()-start_time)/(10**6):.4f}ms")


# class TypedCriticalHandler(_EventHandler):
#     def __init__(self) -> None:
#         self.state: Optional[Tuple[str, Set[int]]] = None

#     def mark_event_happen(self, hint: str) -> Callable[[], None]:
#         assert (self.state is None) or (self.state[0] == hint), f"{hint} happened, but {self.state[0]},{tuple(self.state[1])} is already started"
#         sub_id = 0
#         if self.state is None:
#             self.state = (hint, set((0,)))
#         else:
#             sub_id = len(self.state[1])
#             self.state[1].add(sub_id)
#         return lambda: self._mark_event_terminated(hint, sub_id)

#     def _mark_event_terminated(self, hint: str, sub_id: int) -> None:
#         assert self.state is not None
#         self.state[1].remove(sub_id)
#         if len(self.state[1]) == 0:
#             self.state = None


# handler: _EventHandler = NOPHandler()


# def mark_event_happen(hint: str) -> Callable[[], None]:
#     return handler.mark_event_happen(hint)


# def mark_event(hint: str):
#     return handler.mark_event(hint)

debug_count = 0


def debug_hook():
    global debug_count
    debug_count += 1
    print(debug_count)
    if debug_count == 0:
        print("?")
