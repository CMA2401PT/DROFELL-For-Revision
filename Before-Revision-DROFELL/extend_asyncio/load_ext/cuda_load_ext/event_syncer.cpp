#include "export.h"
#include <thread>
#include <iostream>

CudaEventSyncer::CudaEventSyncer(uintptr_t cuda_stream_i)
{
    // printf("stream id: %u\n", cuda_stream_i);
    if (cuda_stream_i < 1000)
    {
        throw pybind11::value_error("arg is not a valid cuda stream! should use stream.cuda_stream");
    }
    cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(cuda_stream_i);
    // auto stream = at::cuda::getCurrentCUDAStream();
    // cudaStream_t stream_t = stream.stream();
    cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    cudaEventRecord(event_, stream_t);
    // polling_ = polling;
}

std::shared_ptr<CudaEventSyncer> CudaEventSyncer::create(uintptr_t cuda_stream_i)
{
    return std::make_shared<CudaEventSyncer>(cuda_stream_i);
}

void CudaEventSyncer::run_blocking_delegate()
{
    auto start = std::chrono::high_resolution_clock::now();
    // if (polling_)
    // {
    //     while (true)
    //     {
    //         cudaError_t status = cudaEventQuery(event_);
    //         if (status == cudaSuccess)
    //         {
    //             break; // 事件完成
    //         }
    //         else if (status == cudaErrorNotReady)
    //         {
    //             std::this_thread::yield(); // 让出当前线程片段
    //         }
    //         else
    //         {
    //             throw std::runtime_error(std::string("cudaEventQuery failed: ") + cudaGetErrorString(status));
    //         }
    //     }
    // }
    // else
    // {

    // }
    cudaEventSynchronize(event_);
    cudaEventDestroy(event_);
    auto sync_done = std::chrono::high_resolution_clock::now();
    sync_done_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(sync_done.time_since_epoch()).count();
}

pybind11::capsule CudaEventSyncer::get_blocking_delegate()
{
    auto closure = new std::function<void()>([this]
                                             { run_blocking_delegate(); });
    return pybind11::capsule(closure, "closure_fn", [](void *p)
                             { delete static_cast<std::function<void()> *>(p); });
}

bool CudaEventSyncer::query_finished()
{
    cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess)
    {
        auto sync_done = std::chrono::high_resolution_clock::now();
        sync_done_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(sync_done.time_since_epoch()).count();
        return true;
    }
    return false;
}

pybind11::capsule CudaEventSyncer::get_poll_delegate()
{
    auto closure = new std::function<bool()>([this]
                                             { return query_finished(); });
    return pybind11::capsule(closure, "closure_fn", [](void *p)
                             { delete static_cast<std::function<bool()> *>(p); });
}

int64_t CudaEventSyncer::retrieve_result() const
{
    return sync_done_ns_;
}

void bind_cuda_event_syncer(pybind11::module_ &m)
{
    pybind11::class_<CudaEventSyncer, std::shared_ptr<CudaEventSyncer>>(m, "CudaEventSyncer")
        .def(pybind11::init<>(&CudaEventSyncer::create))
        .def("run_blocking_delegate", &CudaEventSyncer::run_blocking_delegate)
        .def("get_blocking_delegate", &CudaEventSyncer::get_blocking_delegate)
        .def("query_finished", &CudaEventSyncer::query_finished)
        .def("get_poll_delegate", &CudaEventSyncer::get_poll_delegate)
        .def("retrieve_result", &CudaEventSyncer::retrieve_result);
}