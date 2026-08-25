#pragma once
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>
#include <memory>

class CudaEventSyncer
{
public:
    CudaEventSyncer(uintptr_t stream_i);
    static std::shared_ptr<CudaEventSyncer> create(uintptr_t cuda_stream_i);
    void run_blocking_delegate();
    pybind11::capsule get_blocking_delegate();
    bool query_finished();
    pybind11::capsule get_poll_delegate();
    int64_t retrieve_result() const;

private:
    cudaEvent_t event_;
    int64_t sync_done_ns_;
    // bool polling_;
};

void bind_cuda_event_syncer(pybind11::module_ &m);

class CudaMemoryCopier
{
public:
    CudaMemoryCopier(
        size_t size,
        void *dst, bool dst_is_device,
        const void *src, bool src_is_device,
        uintptr_t cuda_stream_i);
    static std::shared_ptr<CudaMemoryCopier> create(
        size_t size,
        uintptr_t dst, bool dst_is_device,
        uintptr_t src, bool src_is_device,
        uintptr_t cuda_stream_i);
    void run_blocking_delegate();
    pybind11::capsule get_blocking_delegate();
    int retrieve_result() const;

private:
    cudaMemcpyKind cpy_kind_;
    cudaStream_t stream_;
    size_t size_;
    void *dst_;
    const void *src_;
    int sync_ns_;
};

void bind_cuda_memory_copier(pybind11::module_ &m);