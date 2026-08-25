#include <cuda_runtime.h>
#include "util.cuh"
#include <torch/extension.h>

#define STRINGIFY_(__x) #__x
#define STRINGIFY(__x) STRINGIFY_(__x)
#define TORCH_CHECK_DTYPE(__x, __dtype) TORCH_CHECK((__x).dtype() == torch::__dtype, #__x " is incorrect datatype, must be " #__dtype)
#define TORCH_CHECK_DTYPE_OPT(__x, __dtype) TORCH_CHECK((__x).device().is_meta() || (__x).dtype() == torch::__dtype, #__x " is incorrect datatype, must be " #__dtype)
#define TORCH_CHECK_SHAPES(__x, __dim_x, __y, __dim_y, __scale_y) TORCH_CHECK((__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y, #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_SHAPES_OPT(__x, __dim_x, __y, __dim_y, __scale_y) TORCH_CHECK((__x).device().is_meta() || (__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y, #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_SHAPE_MOD(__x, __dim_x, __mod) TORCH_CHECK((__x).size(__dim_x) % __mod == 0, #__x ".shape[" STRINGIFY(__dim_x) "] must be a multiple of " STRINGIFY(__mod))

void check_cuda(cudaError_t ret)
{
    switch (ret)
    {
        case cudaSuccess:
            break;

        case cudaUnspecified:
            printf(" **** Unspecified error\n");
            TORCH_CHECK(false, "CUDA error");
            break;

        default:
            printf(" **** CUDA error\n"); \
            printf(" **** %s\n", cudaGetErrorString(ret)); \
            TORCH_CHECK(false, "CUDA error"); \
            break;
    }
}
