# cuda sync debug mode: torch.cuda.set_sync_debug_mode

from typing import Literal


CUDA_SYNC_DEBUG_MODE: int = 0  # 0: no-op 1: warn 2: error

RECONSTRUCTE_WEIGTH_FLOAT16 = False
RECONSTRUCTE_WEIGTH_BFLOAT16 = False
REUSE_RECONSTRUCT_CACHE_WHEN_GRAD = True

# in train, decide the policy of forwad and backward
USER_MODEL_ENABLE_CHECKPOTIN = True
DECODERS_TRAIN_METHOD: Literal["grad", "each_decoder", "first.grad", "first.2pass"] = "each_decoder"  # "grad"  "each_decoder" "first.grad" "first.2pass"
DECODERS_TRAIN_DOWNGRAD_METHOD: Literal["first.grad", "first.2pass"] = "first.2pass"
# whether to use multi cuda stream to parallel handle multi prefill in inference
MULTI_PREFILL_STREAM = True

# Replace flash_attn_func with PyTorch SDPA's math backend for prefill and
# finetuning attention. This is substantially slower, but is useful when exact
# repeatability matters more than FlashAttention throughput.
USE_PYTORCH_MATH_ATTENTION = False


# if >0, override max sequence length in rope config
MAX_TOKEN_LEN_OVERRIDE = 128*1024

# infer block size (BLOCK_SIZE slot in one block)
BLOCK_SIZE = 32

# ptuning_lt = time.time_ns()
# last_action: str = ""
# in_finetuning_operation = False
stage: str = "@"
# def clear_ptuning_timer():
#     global ptuning_lt
#     ptuning_lt = time.time_ns()


# async def check_priority(action: str):
#     global ptuning_lt
#     global last_action
#     ctime = time.time_ns()
#     delta_time = ctime-ptuning_lt
#     ptuning_lt = ctime
#     if delta_time > 8*(10**6):
#         print(f"{last_action}->{action}", delta_time/(10**6))
#     last_action = action
#     await asyncio.sleep(0)

# operations_record: List[Tuple[str, int, int]] = []


# @asynccontextmanager
# async def finetuning_operation(action: str):
#     action = stage+action
#     global ptuning_lt
#     global last_action
#     global in_finetuning_operation
#     assert not in_finetuning_operation, f"call {action} when {last_action} not yet finished"
#     ctime = time.time_ns()
#     delta_time = ctime-ptuning_lt
#     ptuning_lt = ctime
#     if delta_time > 0.5*(10**6):
#         print(f"not recorded: {last_action}->{action}", delta_time/(10**6))
#     last_action = action
#     in_finetuning_operation = True
#     yield

#     in_finetuning_operation = False
#     ctime = time.time_ns()
#     cpu_time = ctime-ptuning_lt
#     ptuning_lt = ctime
#     # if delta_time > 5*(10**6):
#     #     print(f"{action}", delta_time/(10**6))
#     # torch.cuda.synchronize()
#     # ctime = time.time_ns()
#     # cuda_time = ctime-ptuning_lt
#     # ptuning_lt = ctime
#     cuda_time = 0
#     operations_record.append((action, cpu_time, cuda_time))
#     await asyncio.sleep(0)


# @asynccontextmanager
# async def finetuning_stage(this_stage: str):
#     this_stage += "@"
#     global stage
#     stage = stage+this_stage
#     yield
#     stage = stage.removesuffix(this_stage)

# # PTUNING_OPERATION = finetuning_operation
# # PTUNING_STAGE = finetuning_stage
# # RESOLUTION = 0.0001


# async def wait_and_yield():
#     global ptuning_lt
#     global last_action
#     ev = torch.cuda.streams.Event(enable_timing=False, blocking=False)
#     ev.record(torch.cuda.current_stream())
#     while not ev.query():
#         # st = time.time_ns()
#         await asyncio.sleep(RESOLUTION)

#     ptuning_lt = time.time_ns()
#     last_action = "wait"
#     # actual_sleep = time.time_ns()-st
#     # if actual_sleep > (RESOLUTION*(10**9))*2:
#     #     print(f"resolution too small, actual={actual_sleep/(10**9)}")
