#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor bytes_to_cpu_tensor(py::bytes src_obj)
{
    PyBytesObject *bytesObject = (PyBytesObject *)src_obj.ptr();
    size_t head = ((uint32_t *)bytesObject->ob_sval)[0];
    size_t size_len = head & 0xff;
    uint8_t scalarType = uint8_t(head >> 8);
    auto dtype = at::scalarTypeToTypeMeta(at::ScalarType(scalarType));
    auto item_size = dtype.itemsize();
    auto stdArray = std::vector<int64_t>(size_len);
    size_t rep_num_elems = 1;
    for (size_t i = 0; i < size_len; i++)
    {
        auto _s = *((uint32_t *)(((char *)bytesObject->ob_sval) + (i + 1) * 4));
        stdArray[i] = _s;
        rep_num_elems *= _s;
    }
    auto rep_tensor = at::empty(at::IntArrayRef(stdArray), dtype);
    memcpy((void *)rep_tensor.data_ptr(), ((void *)(((char *)bytesObject->ob_sval) + (size_len + 1) * 4)), rep_num_elems * item_size);

    return rep_tensor;
}

py::object cpu_tensor_to_bytes(torch::Tensor &src_tensor)
{
    auto dtype = src_tensor.dtype();
    auto scalarType = dtype.toScalarType();

    auto numel = src_tensor.numel();
    auto item_size = dtype.itemsize();
    // printf("item size: %d\n", item_size);
    auto sizes = src_tensor.sizes();

    size_t payload_size = numel * item_size;
    PyBytesObject *bytesObject = nullptr;
    size_t py_object_head_offset = offsetof(PyBytesObject, ob_sval);
    size_t size_info_space = (sizes.size() + 1) * 4;
    size_t total_space = payload_size + size_info_space;
    bytesObject = (PyBytesObject *)PyObject_Malloc(py_object_head_offset + total_space + 1);
    *((uint32_t *)(((char *)bytesObject) + py_object_head_offset)) = (((uint32_t)sizes.size()) | ((uint32_t)scalarType) << 8);
    for (size_t i = 0; i < sizes.size(); i++)
    {
        *((uint32_t *)(((char *)bytesObject) + py_object_head_offset + (i + 1) * 4)) = (uint32_t)sizes[i];
    }

    memcpy((void *)(((char *)bytesObject) + py_object_head_offset + size_info_space), src_tensor.data_ptr(), payload_size);
    PyObject_INIT_VAR(bytesObject, &PyBytes_Type, total_space);
    bytesObject->ob_shash = -1;
    bytesObject->ob_sval[total_space] = '\0';
    return py::reinterpret_steal<py::object>((PyObject *)bytesObject);
}

#define GPU_TO_CPU 0
#define CPU_TO_GPU 1
#define GPU_TO_GPU 2

void tensor_copy_async(
    torch::Tensor &src_tensor,
    torch::Tensor &dst_tenosr,
    int direction)
{
    auto numel = src_tensor.numel();
    auto item_size = src_tensor.dtype().itemsize();
    size_t bytes_in_block = numel * item_size;
    cudaMemcpyKind memcpy_type;
    if (direction == GPU_TO_CPU)
    {
        memcpy_type = cudaMemcpyDeviceToHost;
    }
    else if (direction == CPU_TO_GPU)
    {
        memcpy_type = cudaMemcpyHostToDevice;
    }
    else if (direction == GPU_TO_GPU)
    {
        memcpy_type = cudaMemcpyDeviceToDevice;
    }
    else
    {
        TORCH_CHECK(false, "Invalid device combination");
    }

    void *src_ptr = src_tensor.data_ptr();
    void *dst_ptr = dst_tenosr.data_ptr();
    cudaMemcpyAsync(
        dst_ptr,
        src_ptr,
        bytes_in_block,
        memcpy_type,
        at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "cpu_tensor_to_bytes",
        &cpu_tensor_to_bytes,
        "serialize cpu tensor to bytes");
    m.def(
        "bytes_to_cpu_tensor",
        &bytes_to_cpu_tensor,
        "deserialize bytes to cpu tensor");
    m.def(
        "tensor_copy_async",
        &tensor_copy_async,
        "copy tensor async");
}
