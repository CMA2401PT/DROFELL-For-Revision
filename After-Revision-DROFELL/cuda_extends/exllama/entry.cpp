#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

#include "util.h"
#include "q4_matrix.cuh"
#include "column_remap.cuh"
#include "q4_matmul.cuh"
#include "q4_transpose_matmul.cuh"
#include "fuse_v2/q_mlp.cuh"
#include "fuse_v2/q_attn.cuh"
#include "half_matmul.cuh"
uintptr_t make_q4(
    torch::Tensor qweight,
    torch::Tensor qzeros,
    torch::Tensor scales,
    torch::Tensor g_idx,
    torch::Tensor cuda_x_map,
    int device,
    bool use_act,
    bool v2_mode)
{
    TORCH_CHECK_DTYPE(qweight, kInt);
    TORCH_CHECK_DTYPE(qzeros, kInt);
    TORCH_CHECK_DTYPE(scales, kHalf);

    int width = qweight.size(1);
    int height = qweight.size(0) * 8;
    int groups = qzeros.size(0);

    if (use_act)
    {
        // printf("use act\n");
        TORCH_CHECK_DTYPE(g_idx, kInt);
        TORCH_CHECK_DTYPE(cuda_x_map, kInt);
        TORCH_CHECK_SHAPES(g_idx, 0, cuda_x_map, 0, 1);
        TORCH_CHECK(height == cuda_x_map.size(0), "cuda_x_map/g_idx and w have incompatible shapes")
    }
    TORCH_CHECK_SHAPES(qweight, 1, qzeros, 1, 8);
    TORCH_CHECK_SHAPES(scales, 1, qweight, 1, 1);
    TORCH_CHECK_SHAPES(qzeros, 0, scales, 0, 1);

    Q4Matrix *m = new Q4Matrix(
        height,
        width,
        groups,

        (uint32_t *)qweight.data_ptr(),
        (uint32_t *)qzeros.data_ptr(),
        (half *)scales.data_ptr(),
        use_act ? (uint32_t *)g_idx.data_ptr() : NULL,
        use_act ? (uint32_t *)cuda_x_map.data_ptr() : NULL,
        device,
        v2_mode,
        at::cuda::getCurrentCUDAStream());
    return reinterpret_cast<uintptr_t>(m);
}

void q4_to_half_matrix(
    uintptr_t w,
    torch::Tensor out_weights,
    int reconstruct_threads_x,
    int reconstruct_threads_y)
{
    Q4Matrix *wm = reinterpret_cast<Q4Matrix *>(w);
    TORCH_CHECK(!wm->v2_mode, "a exllama v1 function is called in v2 q4");
    TORCH_CHECK_DTYPE(out_weights, kHalf);
    TORCH_CHECK(wm->height == out_weights.size(0), "wm->height != out_weights.size(0)");
    TORCH_CHECK(wm->width == out_weights.size(1), "wm->width != out_weights.size(1)");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(out_weights));
    wm->reconstruct((half *)out_weights.data_ptr(), reconstruct_threads_x, reconstruct_threads_y, at::cuda::getCurrentCUDAStream());
}

void q4_to_half_matrix_v2(
    uintptr_t w,
    torch::Tensor out_weights,
    int block_kn_size)
{
    Q4Matrix *wm = reinterpret_cast<Q4Matrix *>(w);
    TORCH_CHECK(wm->v2_mode, "a exllama v2 function is called in v1 q4");
    TORCH_CHECK_DTYPE(out_weights, kHalf);
    TORCH_CHECK(wm->height == out_weights.size(0), "wm->height != out_weights.size(0)");
    TORCH_CHECK(wm->width == out_weights.size(1), "wm->width != out_weights.size(1)");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(out_weights));
    wm->reconstruct_exllama_v2((half *)out_weights.data_ptr(), block_kn_size, at::cuda::getCurrentCUDAStream());
}

void q4_to_bfloat16_matrix(
    uintptr_t w,
    torch::Tensor out_weights,
    int reconstruct_threads_x,
    int reconstruct_threads_y)
{
    Q4Matrix *wm = reinterpret_cast<Q4Matrix *>(w);
    TORCH_CHECK(!wm->v2_mode, "a exllama v1 function is called in v2 q4");
    TORCH_CHECK_DTYPE(out_weights, kBFloat16);
    TORCH_CHECK(wm->height == out_weights.size(0), "wm->height != out_weights.size(0)");
    TORCH_CHECK(wm->width == out_weights.size(1), "wm->width != out_weights.size(1)");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(out_weights));
    wm->reconstruct_nv_bfloat16((nv_bfloat16 *)out_weights.data_ptr(), reconstruct_threads_x, reconstruct_threads_y, at::cuda::getCurrentCUDAStream());
}

void q4_to_bfloat16_matrix_v2(
    uintptr_t w,
    torch::Tensor out_weights,
    int block_kn_size)
{
    Q4Matrix *wm = reinterpret_cast<Q4Matrix *>(w);
    TORCH_CHECK(wm->v2_mode, "a exllama v2 function is called in v1 q4");
    TORCH_CHECK_DTYPE(out_weights, kBFloat16);
    TORCH_CHECK(wm->height == out_weights.size(0), "wm->height != out_weights.size(0)");
    TORCH_CHECK(wm->width == out_weights.size(1), "wm->width != out_weights.size(1)");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(out_weights));
    wm->reconstruct_nv_bfloat16_exllama_v2((nv_bfloat16 *)out_weights.data_ptr(), block_kn_size, at::cuda::getCurrentCUDAStream());
}

void column_remap(
    torch::Tensor x,
    torch::Tensor x_new,
    torch::Tensor x_map,
    const int collumn_shuffle_block_x,
    const int collumn_shuffle_block_y)
{
    TORCH_CHECK((x.dtype() == torch::kHalf) || (x.dtype() == torch::kBFloat16), "input must be float16 or bfloat16");
    TORCH_CHECK((x_new.dtype() == torch::kHalf) || (x_new.dtype() == torch::kBFloat16), "output must be float16 or bfloat16");
    TORCH_CHECK_DTYPE(x_map, kInt);
    TORCH_CHECK_SHAPES(x_map, 0, x, 1, 1);

    int height = x.size(0);
    int width = x.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    column_remap_cuda(
        (half *)x.data_ptr(),
        (half *)x_new.data_ptr(),
        height,
        width,
        (uint32_t *)x_map.data_ptr(),
        collumn_shuffle_block_x,
        collumn_shuffle_block_y,
        at::cuda::getCurrentCUDAStream());
}

void column_restore(
    torch::Tensor x,
    torch::Tensor x_new,
    torch::Tensor x_map,
    const int collumn_shuffle_block_x,
    const int collumn_shuffle_block_y)
{

    TORCH_CHECK((x.dtype() == torch::kHalf) || (x.dtype() == torch::kBFloat16), "input must be float16 or bfloat16");
    TORCH_CHECK((x_new.dtype() == torch::kHalf) || (x_new.dtype() == torch::kBFloat16), "output must be float16 or bfloat16");
    TORCH_CHECK_DTYPE(x_map, kInt);
    TORCH_CHECK_SHAPES(x_map, 0, x, 1, 1);

    int height = x.size(0);
    int width = x.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    column_restore_cuda(
        (half *)x.data_ptr(),
        (half *)x_new.data_ptr(),
        height,
        width,
        (uint32_t *)x_map.data_ptr(),
        collumn_shuffle_block_x,
        collumn_shuffle_block_y,
        at::cuda::getCurrentCUDAStream());
}

void q4_matmul_cuda_api(
    torch::Tensor x,
    uintptr_t w,
    torch::Tensor out,
    torch::Tensor temp_x,
    bool no_zero,
    bool no_half2,
    bool no_fused_remap,
    const int threads_x, // 32 Block size and thread count along columns in w and out
    const int threads_y, // 1 Block size and thread count along rows in x and out
    const int block_size_z,
    const int matmul_group_step,
    const int x_collumn_shuffle_block_x,
    const int x_collumn_shuffle_block_y,
    bool skip_remap)
{
    Q4Matrix *wm = reinterpret_cast<Q4Matrix *>(w);
    TORCH_CHECK(!wm->v2_mode, "a exllama v1 function is called in v2 q4");

    TORCH_CHECK((x.dtype() == torch::kHalf) || (x.dtype() == torch::kBFloat16), "x must be float16 or bfloat16");
    TORCH_CHECK((out.dtype() == torch::kHalf) || (out.dtype() == torch::kBFloat16), "out must be float16 or bfloat16");
    TORCH_CHECK_SHAPES(x, 0, out, 0, 1);
    TORCH_CHECK(wm->height == x.size(-1), "x and w have incompatible shapes");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    int x_height = x.size(0);

    q4_matmul_cuda(
        (half *)x.data_ptr(),
        x_height,
        wm,
        (half *)out.data_ptr(),
        (half *)temp_x.data_ptr(),
        no_zero, no_half2, no_fused_remap,
        threads_x, threads_y, block_size_z, matmul_group_step,
        x_collumn_shuffle_block_x,
        x_collumn_shuffle_block_y,
        skip_remap,
        at::cuda::getCurrentCUDAStream());
}

void q4_matmul_cuda_v2_api(
    torch::Tensor x,
    uintptr_t w,
    torch::Tensor out,
    const int block_kn_size,
    const int block_m_size_max,
    bool no_zero)
{
    Q4Matrix *wm = reinterpret_cast<Q4Matrix *>(w);
    TORCH_CHECK(wm->v2_mode, "a exllama v2 function is called in v1 q4");

    TORCH_CHECK((x.dtype() == torch::kHalf) || (x.dtype() == torch::kBFloat16), "x must be float16 or bfloat16");
    TORCH_CHECK((out.dtype() == torch::kHalf) || (out.dtype() == torch::kBFloat16), "out must be float16 or bfloat16");
    TORCH_CHECK(block_kn_size == 128, "currently support kn size = 128 only")
    TORCH_CHECK_SHAPES(x, 0, out, 0, 1);
    TORCH_CHECK(wm->height == x.size(-1), "x and w have incompatible shapes");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    int x_height = x.size(0);

    q4_matmul_cuda_v2(
        (half *)x.data_ptr(),
        x_height,
        wm,
        (half *)out.data_ptr(),
        block_kn_size,
        block_m_size_max,
        no_zero,
        at::cuda::getCurrentCUDAStream());
}

void q4_matmul_blas_api(
    torch::Tensor x,
    uintptr_t w,
    torch::Tensor out,
    torch::Tensor temp_weight,
    torch::Tensor temp_x,
    bool no_zero,
    const int x_collumn_shuffle_block_x,
    const int x_collumn_shuffle_block_y,
    const int reconstruct_threads_x,
    const int reconstruct_threads_y,

    bool skip_remap,
    bool skip_reconstruct,
    bool bfloat16)
{
    Q4Matrix *wm = reinterpret_cast<Q4Matrix *>(w);
    TORCH_CHECK(!wm->v2_mode, "a exllama v1 function is called in v2 q4");

    TORCH_CHECK((x.dtype() == torch::kHalf) || (x.dtype() == torch::kBFloat16), "x must be float16 or bfloat16");
    TORCH_CHECK((out.dtype() == torch::kHalf) || (out.dtype() == torch::kBFloat16), "out must be float16 or bfloat16");
    TORCH_CHECK((x.dtype() == torch::kBFloat16) == bfloat16, "x must be bfloat16");
    TORCH_CHECK((out.dtype() == torch::kBFloat16) == bfloat16, "out must be bfloat16");
    TORCH_CHECK_SHAPES(x, 0, out, 0, 1);
    TORCH_CHECK(wm->height == x.size(-1), "x and w have incompatible shapes")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    int x_height = x.size(0);

    // printf("q4_matmul_recons_cuda\n");
    q4_matmul_cudablas(
        (half *)x.data_ptr(),
        x_height,
        wm,
        (half *)out.data_ptr(),
        (half *)temp_weight.data_ptr(),
        (half *)temp_x.data_ptr(),
        at::cuda::getCurrentCUDABlasHandle(),
        no_zero,
        x_collumn_shuffle_block_x,
        x_collumn_shuffle_block_y,
        reconstruct_threads_x,
        reconstruct_threads_y,
        skip_remap,
        skip_reconstruct,
        bfloat16,
        at::cuda::getCurrentCUDAStream());
}

void q4_matmul_blas_v2_api(
    torch::Tensor x,
    uintptr_t w,
    torch::Tensor out,
    torch::Tensor temp_weight,
    bool no_zero,
    const int block_kn_size,
    bool skip_reconstruct,
    bool bfloat16)
{
    Q4Matrix *wm = reinterpret_cast<Q4Matrix *>(w);
    TORCH_CHECK(wm->v2_mode, "a exllama v2 function is called in v1 q4");

    TORCH_CHECK((x.dtype() == torch::kHalf) || (x.dtype() == torch::kBFloat16), "x must be float16 or bfloat16");
    TORCH_CHECK((out.dtype() == torch::kHalf) || (out.dtype() == torch::kBFloat16), "out must be float16 or bfloat16");
    TORCH_CHECK((x.dtype() == torch::kBFloat16) == bfloat16, "x must be bfloat16");
    TORCH_CHECK((out.dtype() == torch::kBFloat16) == bfloat16, "out must be bfloat16");
    TORCH_CHECK_SHAPES(x, 0, out, 0, 1);
    TORCH_CHECK(wm->height == x.size(-1), "x and w have incompatible shapes")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    int x_height = x.size(0);

    // printf("q4_matmul_recons_cuda\n");
    q4_matmul_cudablas_v2(
        (half *)x.data_ptr(),
        x_height,
        wm,
        (half *)out.data_ptr(),
        (half *)temp_weight.data_ptr(),
        at::cuda::getCurrentCUDABlasHandle(),
        no_zero,
        block_kn_size,
        skip_reconstruct,
        bfloat16,
        at::cuda::getCurrentCUDAStream());
}

void q4_transpose_matmul_blas_api(
    torch::Tensor x,
    uintptr_t w,
    torch::Tensor out,
    torch::Tensor temp_weight,
    torch::Tensor temp_x,
    bool no_zero,
    const int x_collumn_shuffle_block_x,
    const int x_collumn_shuffle_block_y,
    const int reconstruct_threads_x,
    const int reconstruct_threads_y,

    bool skip_remap,
    bool skip_reconstruct,
    bool bfloat16)
{
    Q4Matrix *wm = reinterpret_cast<Q4Matrix *>(w);
    TORCH_CHECK(!wm->v2_mode, "a exllama v1 function is called in v2 q4");

    TORCH_CHECK((x.dtype() == torch::kHalf) || (x.dtype() == torch::kBFloat16), "x must be float16 or bfloat16");
    TORCH_CHECK((out.dtype() == torch::kHalf) || (out.dtype() == torch::kBFloat16), "out must be float16 or bfloat16");
    TORCH_CHECK((x.dtype() == torch::kBFloat16) == bfloat16, "x must be bfloat16");
    TORCH_CHECK((out.dtype() == torch::kBFloat16) == bfloat16, "out must be bfloat16");
    TORCH_CHECK_SHAPES(x, 0, out, 0, 1);
    TORCH_CHECK(wm->width == x.size(-1), "x and w have incompatible shapes")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    int x_height = x.size(0);

    // printf("q4_matmul_recons_cuda\n");
    q4_transpose_matmul_blas(
        (half *)x.data_ptr(),
        x_height,
        wm,
        (half *)out.data_ptr(),
        (half *)temp_weight.data_ptr(),
        (half *)temp_x.data_ptr(),
        at::cuda::getCurrentCUDABlasHandle(),
        no_zero,
        x_collumn_shuffle_block_x,
        x_collumn_shuffle_block_y,
        reconstruct_threads_x,
        reconstruct_threads_y,
        skip_remap,
        skip_reconstruct,
        bfloat16,
        at::cuda::getCurrentCUDAStream());
}

void q4_transpose_matmul_blas_v2_api(
    torch::Tensor x,
    uintptr_t w,
    torch::Tensor out,
    torch::Tensor temp_weight,
    bool no_zero,
    const int block_kn_size,
    bool skip_reconstruct,
    bool bfloat16)
{
    Q4Matrix *wm = reinterpret_cast<Q4Matrix *>(w);
    TORCH_CHECK(wm->v2_mode, "a exllama v2 function is called in v1 q4");

    TORCH_CHECK((x.dtype() == torch::kHalf) || (x.dtype() == torch::kBFloat16), "x must be float16 or bfloat16");
    TORCH_CHECK((out.dtype() == torch::kHalf) || (out.dtype() == torch::kBFloat16), "out must be float16 or bfloat16");
    TORCH_CHECK((x.dtype() == torch::kBFloat16) == bfloat16, "x must be bfloat16");
    TORCH_CHECK((out.dtype() == torch::kBFloat16) == bfloat16, "out must be bfloat16");
    TORCH_CHECK_SHAPES(x, 0, out, 0, 1);
    TORCH_CHECK(wm->width == x.size(-1), "x and w have incompatible shapes")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    int x_height = x.size(0);

    // printf("q4_matmul_recons_cuda\n");
    q4_transpose_matmul_cudablas_v2(
        (half *)x.data_ptr(),
        x_height,
        wm,
        (half *)out.data_ptr(),
        (half *)temp_weight.data_ptr(),
        at::cuda::getCurrentCUDABlasHandle(),
        no_zero,
        block_kn_size,
        skip_reconstruct,
        bfloat16,
        at::cuda::getCurrentCUDAStream());
}

void mlp_forward_v2_api(
    torch::Tensor layernorm,
    float norm_epsilon,
    uintptr_t gate,
    uintptr_t up,
    uintptr_t down,
    torch::Tensor temp_rms_output,
    torch::Tensor temp_gate_output,
    torch::Tensor temp_up_output,
    torch::Tensor temp_weight,
    torch::Tensor x,
    torch::Tensor output,
    const int block_kn_size,
    const int block_m_size_max,
    const int switch_threshold)
{
    Q4Matrix *gate_q4 = reinterpret_cast<Q4Matrix *>(gate);
    Q4Matrix *up_q4 = reinterpret_cast<Q4Matrix *>(up);
    Q4Matrix *down_q4 = reinterpret_cast<Q4Matrix *>(down);
    TORCH_CHECK(gate_q4->v2_mode, "a exllama v2 function is called in v1 q4");

    TORCH_CHECK((layernorm.dtype() == torch::kHalf), "layernorm weight must be float16 or bfloat16");
    // TORCH_CHECK((out.dtype() == torch::kHalf) || (out.dtype() == torch::kBFloat16) , "out must be float16 or bfloat16");
    // TORCH_CHECK((x.dtype() == torch::kBFloat16)==bfloat16 , "x must be bfloat16");
    // TORCH_CHECK((out.dtype() == torch::kBFloat16)==bfloat16 , "out must be bfloat16");
    // TORCH_CHECK_SHAPES(x, 0, out, 0, 1);
    // TORCH_CHECK(wm->width == x.size(-1), "x and w have incompatible shapes")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    int x_height = x.size(0);

    mlp_forward(
        (half *)layernorm.data_ptr(),
        norm_epsilon,
        gate_q4,
        up_q4,
        down_q4,
        (half *)temp_rms_output.data_ptr(),
        (half *)temp_gate_output.data_ptr(),
        (half *)temp_up_output.data_ptr(),
        (half *)temp_weight.data_ptr(),
        (half *)x.data_ptr(),
        x_height,
        (half *)output.data_ptr(),
        block_kn_size,
        block_m_size_max,
        switch_threshold,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());
}

void attention_forward_qkv_stage_v2_api(
    torch::Tensor layernorm,
    float norm_epsilon,
    uintptr_t q,
    uintptr_t k,
    uintptr_t v,
    torch::Tensor temp_rms_output,
    torch::Tensor temp_q_output,
    torch::Tensor temp_k_output,
    torch::Tensor temp_v_output,
    torch::Tensor temp_weight,
    torch::Tensor x,
    const int block_kn_size,
    const int block_m_size_max,
    const int switch_threshold,
    int q_len,
    int batch_size)
{
    Q4Matrix *q_proj = reinterpret_cast<Q4Matrix *>(q);
    Q4Matrix *k_proj = reinterpret_cast<Q4Matrix *>(k);
    Q4Matrix *v_proj = reinterpret_cast<Q4Matrix *>(v);
    TORCH_CHECK(q_proj->v2_mode, "a exllama v2 function is called in v1 q4");

    TORCH_CHECK((layernorm.dtype() == torch::kHalf), "layernorm weight must be float16 or bfloat16");
    // TORCH_CHECK((out.dtype() == torch::kHalf) || (out.dtype() == torch::kBFloat16) , "out must be float16 or bfloat16");
    // TORCH_CHECK((x.dtype() == torch::kBFloat16)==bfloat16 , "x must be bfloat16");
    // TORCH_CHECK((out.dtype() == torch::kBFloat16)==bfloat16 , "out must be bfloat16");
    // TORCH_CHECK_SHAPES(x, 0, out, 0, 1);
    // TORCH_CHECK(wm->width == x.size(-1), "x and w have incompatible shapes")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    attention_forward_qkv_stage(
        (half *)layernorm.data_ptr(),
        norm_epsilon,
        q_proj,
        k_proj,
        v_proj,
        (half *)temp_rms_output.data_ptr(),
        (half *)temp_q_output.data_ptr(),
        (half *)temp_k_output.data_ptr(),
        (half *)temp_v_output.data_ptr(),
        (half *)temp_weight.data_ptr(),
        (half *)x.data_ptr(),
        block_kn_size,
        block_m_size_max,
        switch_threshold,
        q_len, batch_size,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());
}

void attention_forward_rope_stage_v2_api(
    torch::Tensor temp_q_output,
    torch::Tensor temp_k_output,
    torch::Tensor sin,
    torch::Tensor cos,
    int q_len,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int past_len,
    torch::Tensor past_lens)
{
    attention_forward_rope_stage(
        (half *)temp_q_output.data_ptr(),
        (half *)temp_k_output.data_ptr(),
        (half *)sin.data_ptr(),
        (half *)cos.data_ptr(),
        q_len, batch_size,
        num_heads,
        num_kv_heads,
        head_dim,
        past_len,
        (uint32_t *)past_lens.data_ptr(),
        at::cuda::getCurrentCUDAStream());
}

void lora(
    bool use_half2,
    uint64_t x,
    const int seq_len, // x,q=seq_len*hidden_size
    const int hidden_size,
    const int kv_size, // k,v=seq_len*kv_size
    uint64_t q_a,
    uint64_t q_b,
    const int q_rank,
    uint64_t k_a,
    uint64_t k_b,
    const int k_rank,
    uint64_t v_a,
    uint64_t v_b,
    const int v_rank,
    uint64_t lora_temp,
    uint64_t temp_q_output,
    uint64_t temp_k_output,
    uint64_t temp_v_output)
{
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    auto stream = at::cuda::getCurrentCUDAStream();

    if (q_rank)
    {
        half_matmul_cublas_cuda(use_half2, false, (half *)x, (half *)q_a, (half *)lora_temp, seq_len, hidden_size, q_rank, handle, stream);
        half_matmul_cublas_cuda(use_half2, true, (half *)lora_temp, (half *)q_b, (half *)temp_q_output, seq_len, q_rank, hidden_size, handle, stream);
    }
    if (k_rank)
    {
        half_matmul_cublas_cuda(use_half2, false, (half *)x, (half *)k_a, (half *)lora_temp, seq_len, hidden_size, k_rank, handle, stream);
        half_matmul_cublas_cuda(use_half2, true, (half *)lora_temp, (half *)k_b, (half *)temp_k_output, seq_len, k_rank, kv_size, handle, stream);
    }
    if (v_rank)
    {
        half_matmul_cublas_cuda(use_half2, false, (half *)x, (half *)v_a, (half *)lora_temp, seq_len, hidden_size, v_rank, handle, stream);
        half_matmul_cublas_cuda(use_half2, true, (half *)lora_temp, (half *)v_b, (half *)temp_v_output, seq_len, v_rank, kv_size, handle, stream);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("make_q4", &make_q4, "make_q4");
    m.def("q4_to_half_matrix", &q4_to_half_matrix, "q4_to_half_matrix");
    m.def("q4_to_half_matrix_v2", &q4_to_half_matrix_v2, "q4_to_half_matrix_v2");
    m.def("q4_to_bfloat16_matrix", &q4_to_bfloat16_matrix, "q4_to_bfloat16_matrix");
    m.def("q4_to_bfloat16_matrix_v2", &q4_to_bfloat16_matrix_v2, "q4_to_bfloat16_matrix_v2");

    m.def("column_remap", &column_remap, "column_remap");
    m.def("column_restore", &column_restore, "column_restore");

    m.def("q4_matmul_cuda", &q4_matmul_cuda_api, "q4_matmul_cuda");
    m.def("q4_matmul_cuda_v2", &q4_matmul_cuda_v2_api, "q4_matmul_cuda_v2");
    m.def("q4_matmul_blas", &q4_matmul_blas_api, "q4_matmul_blas");
    m.def("q4_matmul_blas_v2", &q4_matmul_blas_v2_api, "q4_matmul_blas_v2");
    m.def("q4_transpose_matmul_blas", &q4_transpose_matmul_blas_api, "q4_transpose_matmul_blas");
    m.def("q4_transpose_matmul_blas_v2", &q4_transpose_matmul_blas_v2_api, "q4_transpose_matmul_blas_v2");

    m.def("mlp_forward_v2", &mlp_forward_v2_api, "mlp_forward_v2");
    m.def("attention_forward_qkv_stage_v2", &attention_forward_qkv_stage_v2_api, "attention_forward_qkv_stage_v2_api");
    m.def("attention_forward_rope_stage_v2", &attention_forward_rope_stage_v2_api, "attention_forward_rope_stage_v2_api");
    m.def("lora", &lora, "lora");
}
