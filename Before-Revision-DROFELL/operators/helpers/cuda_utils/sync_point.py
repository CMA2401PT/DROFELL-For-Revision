from contextlib import contextmanager
import torch

from ... import flags

if flags.CUDA_SYNC_DEBUG_MODE != 0:  # type:ignore
    torch.cuda.set_sync_debug_mode(flags.CUDA_SYNC_DEBUG_MODE)


@contextmanager
def known_sync_point_here():
    if flags.CUDA_SYNC_DEBUG_MODE != 0:  # type:ignore
        torch.cuda.set_sync_debug_mode(0)
    yield
    if flags.CUDA_SYNC_DEBUG_MODE != 0:  # type:ignore
        torch.cuda.set_sync_debug_mode(flags.CUDA_SYNC_DEBUG_MODE)
