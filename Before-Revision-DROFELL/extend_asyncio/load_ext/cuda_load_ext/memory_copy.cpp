#include "export.h"
#include <cuda_runtime.h>

CudaMemoryCopier::CudaMemoryCopier(
    size_t size,
    void *dst, bool dst_is_device,
    const void *src, bool src_is_device,
    uintptr_t cuda_stream_i)
{
    this->size_ = size;
    this->stream_ = reinterpret_cast<cudaStream_t>(cuda_stream_i);
    this->src_ = src;
    this->dst_ = dst;

    if (dst_is_device && src_is_device)
    {
        this->cpy_kind_ = cudaMemcpyDeviceToDevice;
    }
    else if ((!dst_is_device) && src_is_device)
    {
        this->cpy_kind_ = cudaMemcpyDeviceToHost;
    }
    else if ((dst_is_device) && (!src_is_device))
    {
        this->cpy_kind_ = cudaMemcpyHostToDevice;
    }
    else
    {
        throw std::runtime_error("Invalid copy kind.");
    }
}

std::shared_ptr<CudaMemoryCopier> CudaMemoryCopier::create(
    size_t size,
    uintptr_t dst, bool dst_is_device,
    uintptr_t src, bool src_is_device,
    uintptr_t cuda_stream_i)
{
    void *cdst = reinterpret_cast<void *>(dst);
    void *csrc = reinterpret_cast<void *>(src);
    return std::make_shared<CudaMemoryCopier>(size, cdst, dst_is_device, csrc, src_is_device, cuda_stream_i);
}

void CudaMemoryCopier::run_blocking_delegate()
{
    auto start = std::chrono::high_resolution_clock::now();
    cudaError_t err = cudaMemcpyAsync(dst_, src_, size_, cpy_kind_, stream_);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("cudaMemcpyAsync failed: " + std::string(cudaGetErrorString(err)));
    }
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    cudaEventRecord(event, stream_);
    cudaEventSynchronize(event);
    cudaEventDestroy(event);
    auto end = std::chrono::high_resolution_clock::now();
    sync_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

pybind11::capsule CudaMemoryCopier::get_blocking_delegate()
{
    auto closure = new std::function<void()>([this]
                                             { run_blocking_delegate(); });
    return pybind11::capsule(closure, "closure_fn", [](void *p)
                             { delete static_cast<std::function<void()> *>(p); });
}

int CudaMemoryCopier::retrieve_result() const
{
    return sync_ns_;
}

void bind_cuda_memory_copier(pybind11::module_ &m)
{
    pybind11::class_<CudaMemoryCopier, std::shared_ptr<CudaMemoryCopier>>(m, "CudaMemoryCopier")
        .def(pybind11::init<>(&CudaMemoryCopier::create))
        .def("run_blocking_delegate", &CudaMemoryCopier::run_blocking_delegate)
        .def("get_blocking_delegate", &CudaMemoryCopier::get_blocking_delegate)
        .def("retrieve_result", &CudaMemoryCopier::retrieve_result);
}