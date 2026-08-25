#include "q_mlp.cuh"
#include "../q4_matmul.cuh"
#include "rms_norm.cuh"
#include "../util.cuh"
#include "../matrix.cuh"
#include "adapt_gemm.cuh"
// #include "lora.cuh"

const int THREADS_X = 32;
const int THREADS_Y = 4;
// const int MAX_DIMENSION = 8192;

__device__ __forceinline__ half silu(half x)
{
    half one = __float2half(1.0f);
    half neg_x = __hneg(x);
    half e = hexp(neg_x);
    half sum = __hadd(one, e);
    half r = hrcp(sum);
    half result = __hmul(x, r);
    return result;
}

__device__ __forceinline__ half2 silu(half2 x)
{
    half2 one = __float2half2_rn(1.0f);
    half2 neg_x = __hneg2(x);
    half2 e = h2exp(neg_x);
    half2 sum = __hadd2(one, e);
    half2 r = h2rcp(sum);
    half2 result = __hmul2(x, r);
    return result;
}

typedef void (*fp_silu_mul_kernel)
(
    half*,
    const half*,
    const int,
    const int
);

template <bool use_half2>
__global__ void silu_mul_kernel
(
    half* __restrict__ x,
    const half* __restrict__ y,
    const int height,
    const int width
)
{
    MatrixView_half_rw x_(x, height, width);
    MatrixView_half y_(y, height, width);

    int column = (THREADS_X * blockIdx.x + threadIdx.x); if constexpr (use_half2) column *= 2;
    int row = THREADS_Y * blockIdx.y + threadIdx.y;
    if (row >= height) return;

    // silu(x) * y

    if constexpr (use_half2)
    {
        half2 one = __half2half2(__float2half(1.0f));

        half2 x_item = x_.item_half2(row, column);
        half2 y_item = y_.item_half2(row, column);

        x_item = silu(x_item);
        x_item = __hmul2(x_item, y_item);

        x_.set_half2(row, column, x_item);
    }
    else
    {
        half one = __float2half(1.0f);

        half x_item = x_.item(row, column);
        half y_item = y_.item(row, column);

        x_item = silu(x_item);
        x_item = __hmul(x_item, y_item);

        x_.set(row, column, x_item);
    }
}

fp_silu_mul_kernel pick_silu_mul_kernel(bool use_half2)
{
    if (use_half2) return silu_mul_kernel<true>;
    else           return silu_mul_kernel<false>;
};

// const int THREADS_X = 32;
// const int THREADS_Y = 4;

void mlp_forward(
    half* layernorm,
    float norm_epsilon,
    Q4Matrix* gate,
    Q4Matrix* up,
    Q4Matrix* down,
    half* temp_rms_output,
    half* temp_gate_output,
    half* temp_up_output,
    half* temp_weight,
    half* x,
    int x_height,
    half* output,
    const int block_kn_size,
    const int block_m_size_max,
    const int switch_threshold,
    cublasHandle_t cublas_handle,
    cudaStream_t alt_stream
){
    int columns=gate->height;
    int intermediate_size=gate->width;
    // temp_rms_output = rms(x;layernorm, norm_epsilon)
    rms_norm_cuda(x, layernorm, temp_rms_output, norm_epsilon, x_height, columns,alt_stream);
    q4_matmul_auto_select(cublas_handle,temp_rms_output,gate,temp_gate_output,x_height,false,temp_weight,block_kn_size,block_m_size_max,false,switch_threshold,alt_stream);
    q4_matmul_auto_select(cublas_handle,temp_rms_output,up,temp_up_output,x_height,false,temp_weight,block_kn_size,block_m_size_max,false,switch_threshold,alt_stream);

    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = THREADS_Y;
    bool use_half2=true;
    gridDim.x = DIVIDE(up->width, THREADS_X) / (use_half2 ? 2 : 1);
    gridDim.y = DIVIDE(x_height, THREADS_Y);

    fp_silu_mul_kernel kernel = pick_silu_mul_kernel(use_half2);
    kernel<<<gridDim, blockDim,0,alt_stream>>>(temp_gate_output, temp_up_output, x_height, intermediate_size);

    q4_matmul_auto_select(cublas_handle,temp_gate_output,down,output,x_height,false,temp_weight,block_kn_size,block_m_size_max,false,switch_threshold,alt_stream);
}

// QMLP::QMLP
// (
//     half* _layernorm,
//     float _norm_epsilon,
//     Q4Matrix* _gate,
//     Q4Matrix* _up,
//     Q4Matrix* _down,
//     half* _temp_state,
//     half* _temp_a,
//     half* _temp_b,
//     half* _temp_dq,
//     int _max_rows
// ):
//     layernorm(_layernorm),
//     norm_epsilon(_norm_epsilon),
//     gate(_gate),
//     up(_up),
//     down(_down),
//     temp_state(_temp_state),
//     temp_a(_temp_a),
//     temp_b(_temp_b),
//     temp_dq(_temp_dq),
//     max_rows(_max_rows)
// {
// }

// void QMLP::forward_
// (
//     cublasHandle_t cublas_handle,
//     half* x,
//     int rows,
//     int columns,
//     const std::vector<uintptr_t>& loras,
//     half* lora_temp
// )
// {
//     bool use_half2 = true;
//     int intermediate_size = gate->width;
//     // temp_state=rms(x,layernorm)
//     rms_norm_cuda(x, layernorm, temp_state, norm_epsilon, rows, columns);
//     // temp a = temp_state @ gate
//     gemm_half_q_half_cuda(cublas_handle, temp_state, gate, temp_a, rows, intermediate_size, columns, true, temp_dq);
//     // temp b = temp_state @ up
//     gemm_half_q_half_cuda(cublas_handle, temp_state, up,   temp_b, rows, intermediate_size, columns, true, temp_dq);

//     // apply_loras_cuda(cublas_handle, gate_proj_lora, loras, gate, temp_state, temp_a, lora_temp, rows);
//     // apply_loras_cuda(cublas_handle, up_proj_lora,   loras, up,   temp_state, temp_b, lora_temp, rows);

//     dim3 blockDim, gridDim;
//     blockDim.x = THREADS_X;
//     blockDim.y = THREADS_Y;
//     gridDim.x = DIVIDE(up->width, THREADS_X) / (use_half2 ? 2 : 1);
//     gridDim.y = DIVIDE(rows, THREADS_Y);

//     fp_silu_mul_kernel kernel = pick_silu_mul_kernel(use_half2);
//     // temp_a = silu(temp_a) * temp_b
//     kernel<<<gridDim, blockDim>>>(temp_a, temp_b, rows, intermediate_size);
//     // temp_a = temp_a @ down
//     gemm_half_q_half_cuda(cublas_handle, temp_a, down, x, rows, columns, intermediate_size, false, temp_dq);

//     // apply_loras_cuda(cublas_handle, down_proj_lora, loras, down, temp_a, x, lora_temp, rows);
// }
