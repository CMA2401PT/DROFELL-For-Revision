#include <pybind11/pybind11.h>
#include "export.h"

PYBIND11_MODULE(cuda_async_load, m)
{
    bind_cuda_event_syncer(m);
    bind_cuda_memory_copier(m);
}