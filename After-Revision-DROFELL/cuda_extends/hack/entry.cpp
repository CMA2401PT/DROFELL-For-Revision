#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

torch::Tensor tensor_from_pointer(
    int64_t ptr,
    torch::Tensor template_tensor,
    std::vector<int64_t> sizes)
{
    void *raw_ptr = reinterpret_cast<void *>(ptr);

    auto options = template_tensor.options();
    auto strides = template_tensor.strides();
    auto dtype = template_tensor.scalar_type();
    auto device = template_tensor.device();

    auto tensor = torch::from_blob(
        raw_ptr,
        sizes,
        strides,
        [=](void *) { /* no-op deleter, we don't free this memory */ },
        options);

    return tensor;
}

pybind11::memoryview tensor_to_memory_view(
    torch::Tensor tensor)
{
    assert(tensor.is_cpu());
    assert(tensor.is_contiguous());
    auto elements = tensor.numel();
    auto elem_size = tensor.element_size();
    auto len = elements * elem_size;
    auto ptr = tensor.data_ptr();
    return pybind11::memoryview::from_memory(ptr, len);
}

// torch::Tensor tensor_from_memory_view(
//     py::memoryview mv,
//     std::vector<int64_t> sizes,
//     torch::ScalarType dtype,
//     c10::optional<std::vector<int64_t>> strides = c10::nullopt)
// {
//     auto ptr = (char *)PyMemoryView_GET_BUFFER(mv.ptr())->buf;
//     auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);

//     // no-op deleter, manage memory outside
//     auto deleter = [](void *) {};

//     torch::Tensor t;
//     if (strides.has_value())
//     {
//         t = torch::from_blob(ptr, sizes, *strides, deleter, options);
//     }
//     else
//     {
//         t = torch::from_blob(ptr, sizes, deleter, options);
//     }

//     return t;
// }

torch::Tensor tensor_from_memory_view(
    pybind11::memoryview mv,
    torch::Tensor template_tensor)
{
    assert(template_tensor.is_meta());
    auto options = template_tensor.options();
    auto strides = template_tensor.strides();
    auto new_options = options.device(torch::kCPU);

    auto ptr = (char *)PyMemoryView_GET_BUFFER(mv.ptr())->buf;
    auto tensor = torch::from_blob(
        ptr,
        template_tensor.sizes(),
        strides,
        [](void *) { /* no-op deleter */ },
        new_options);

    return tensor;
}

torch::Tensor
fake_device_tensor_like(
    c10::Device device,
    torch::Tensor template_tensor)
{
    auto options = template_tensor.options();
    auto strides = template_tensor.strides();

    auto new_options = options.device(device);

    auto tensor = torch::from_blob(
        0,
        template_tensor.sizes(),
        strides,
        [=](void *) { /* no-op deleter, we don't free this memory */ },
        new_options);

    return tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tensor_from_pointer", &tensor_from_pointer, "Create tensor from raw pointer");
    m.def("fake_device_tensor_like", &fake_device_tensor_like, "Create a fake tensor (ptr=0) with specific device and other options same as given one");
    m.def("tensor_to_memory_view", &tensor_to_memory_view, "Create memory view point to tensor");
    m.def("tensor_from_memory_view", &tensor_from_memory_view, "Create tensor from memory view data");
}